import sys
import yaml
import json
from pathlib import Path

# Add project root to PYTHONPATH so we can import modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.llm_service.api.gemini_llm_service import get_gemini_chat_completion
from backend.llm_service.api.groq_llm_service import get_groq_chat_completion
from backend.llm_service.serverless.modal_vllm_client import (
    get_qwen_chat_completion as get_modal_qwen, 
    get_mistral_chat_completion as get_modal_mistral
)
from backend.llm_service.oss_models.qwen_model import get_qwen_chat_completion as get_local_qwen
from backend.database_mangement.supabase_db import (
    register_user, authenticate_user, save_chat_generation, fetch_user_chats,
    get_user_memory, save_user_memory
)
from backend.memory.long_term_memory import (
    update_user_long_term_memory, build_system_prompt_with_memory
)
from uuid import uuid4, UUID
from agent.guardrail_for_llm import AdvancedRuleBasedGuardrail
from eval.eval_llm_call import (
    eval_llm_call, eval_hallucination, eval_bias, eval_content_safety,
    get_eval_prompts
)
from agent.tools import duckduckgo_search

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
CONFIG_FILE = PROJECT_ROOT / "config" / "configration.yaml"

# Mount static assets (CSS, JS)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

@app.get("/")
async def serve_frontend():
    """Serves the login page as the root URL."""
    return FileResponse(FRONTEND_DIR / "login.html")

@app.get("/chat")
async def serve_chat():
    """Serves the chat page."""
    return FileResponse(FRONTEND_DIR / "chat.html")

@app.get("/mode-select")
async def serve_mode_select():
    """Serves the mode selection page (post-login)."""
    return FileResponse(FRONTEND_DIR / "mode_select.html")

@app.get("/eval")
async def serve_eval():
    """Serves the evaluation dashboard page."""
    return FileResponse(FRONTEND_DIR / "eval.html")


class AuthRequest(BaseModel):
    user_id: str
    password: str

@app.post("/api/auth/register")
async def register_endpoint(request: AuthRequest):
    try:
        success = register_user(request.user_id, request.password)
        if success:
            return {"success": True, "message": "Account created successfully."}
        else:
            raise HTTPException(status_code=409, detail="Username already exists.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/login")
async def login_endpoint(request: AuthRequest):
    try:
        authenticated = authenticate_user(request.user_id, request.password)
        if authenticated:
            return {"success": True, "user_id": request.user_id}
        else:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class SaveChatRequest(BaseModel):
    user_id: str
    chat_id: UUID
    user_message: str
    ai_message: str

@app.post("/api/chats/save")
async def save_chat_endpoint(request: SaveChatRequest):
    try:
        gen_id = save_chat_generation(request.user_id, str(request.chat_id), request.user_message, request.ai_message)
        return {"success": True, "generation_id": gen_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chats/{user_id}")
async def get_chats_endpoint(user_id: str):
    try:
        chats = fetch_user_chats(user_id)
        return {"success": True, "chats": chats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/{user_id}")
async def get_memory_endpoint(user_id: str):
    try:
        memory = get_user_memory(user_id)
        return {"success": True, "memory": memory}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MemoryUpdateRequest(BaseModel):
    user_id: str
    user_message: str
    ai_message: str

@app.post("/api/memory/update")
async def update_memory_endpoint(request: MemoryUpdateRequest):
    try:
        updated = update_user_long_term_memory(
            request.user_id, request.user_message, request.ai_message
        )
        return {"success": True, "memory": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    user_id: Optional[str] = None
    generation_id: Optional[UUID] = Field(default_factory=uuid4)
    system_prompt: Optional[str] = "You are a helpful personal assistant."

def format_history_for_string_content(messages: List[ChatMessage]) -> str:
    """Combines memory history into a single string for endpoints expecting just 'content'"""
    formatted = ""
    for msg in messages:
        if msg.role == "user":
            formatted += f"\nHuman Query: {msg.content}"
        elif msg.role == "assistant":
            formatted += f"\nAI response: {msg.content}"
    formatted += "\nAI response:"
    return formatted.strip()

# Initialize the guardrail instance
llm_guardrail = AdvancedRuleBasedGuardrail()

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    model = request.model
    
    # Extract the last user message for guardrail check
    latest_user_message = ""
    for msg in request.messages:
        if msg.role == "user":
            latest_user_message = msg.content
            
    # Apply input guardrails
    if latest_user_message:
        is_safe, violations = llm_guardrail.check_output(latest_user_message)
        if not is_safe:
            async def blocked_response_generator():
                yield f"Guardrail blocked request due to policy violations: {', '.join(violations)}"
            
            return StreamingResponse(blocked_response_generator(), media_type="text/event-stream")

    system_prompt = request.system_prompt
    if request.user_id:
        memory = get_user_memory(request.user_id)
        system_prompt = build_system_prompt_with_memory(system_prompt, memory)
    
    content = format_history_for_string_content(request.messages)
    
    system_prompt = system_prompt + "If the user asks for latest news, current events, recent updates, or you need up-to-date information, respond with exactly 'duck_duck_go' and nothing else. Otherwise, answer normally."

    config = load_config()
    models_conf = config.get("models", {})
    api_calls = models_conf.get("api_call", {})
    gemini_models = api_calls.get("gemini", [])
    groq_models = api_calls.get("groq", [])
    oss_models = models_conf.get("oss_models", [])
    serverless_models = models_conf.get("serverless_models", [])

    generator = None


    if model in gemini_models:
        generator = get_gemini_chat_completion(system_prompt, model, content)
    elif model in groq_models:
        generator = get_groq_chat_completion(system_prompt, model, content)
    elif model in oss_models:
        print("loading the model")
        generator = get_local_qwen(system_prompt, model, content)
    elif model in serverless_models:
        print("starting the server")
        if "qwen" in model.lower():
            generator = get_modal_qwen(system_prompt, content)
        elif "mistral" in model.lower():
            generator = get_modal_mistral(system_prompt, content)
        else:
            raise HTTPException(status_code=400, detail=f"Serverless model '{model}' router not implemented.")
    else:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not found in configuration.")

    if not generator:
        raise HTTPException(status_code=500, detail="Failed to initialize generator.")

    # Convert the generator into a list to check the output before streaming to user
    response_chunks = []
    for chunk in generator:
        response_chunks.append(chunk)
    
    full_response = "".join(response_chunks)

    # Check if the LLM outputted our special tool trigger
    if full_response.strip() == "duck_duck_go":
        print(f"Tool triggered for query: {latest_user_message}")
        search_results = duckduckgo_search(latest_user_message, max_results=5)
        
        # Now we append the search results to the context and call the LLM *again*
        new_content = content + f"\n\nHere are the search results to answer the user's query:\n{search_results}"
        
        # Remove the duck_duck_go prompt so it doesn't get stuck in a loop
        clean_system_prompt = system_prompt.replace("If the user asks for latest news, current events, recent updates, or you need up-to-date information, respond with exactly 'duck_duck_go' and nothing else. Otherwise, answer normally.", "")

        # Call the appropriate model again with the new context
        generator2 = None
        if model in gemini_models:
            generator2 = get_gemini_chat_completion(clean_system_prompt, model, new_content)
        elif model in groq_models:
            generator2 = get_groq_chat_completion(clean_system_prompt, model, new_content)
        elif model in oss_models:
            generator2 = get_local_qwen(clean_system_prompt, model, new_content)
        elif model in serverless_models:
            if "qwen" in model.lower():
                generator2 = get_modal_qwen(clean_system_prompt, new_content)
            elif "mistral" in model.lower():
                generator2 = get_modal_mistral(clean_system_prompt, new_content)
                
        async def combined_generator():
            # 1. First, instantly yield a markdown block showing what was searched
            search_summary = search_results[:300].replace('\n', ' ') + "..." if len(search_results) > 300 else search_results
            yield f"**Searching Web For:** *{latest_user_message}*\n\n"
            yield f"> {search_summary}\n\n"
            yield f"---\n\n"
            
            # 2. Then yield the chunks from the actual LLM's final response
            if generator2:
                for chunk in generator2:
                    yield chunk

        return StreamingResponse(combined_generator(), media_type="text/event-stream")
    
    # If it wasn't a tool call, we already consumed the generator, so we need a new one to stream the original response
    async def string_generator():
        yield full_response

    return StreamingResponse(string_generator(), media_type="text/event-stream")

@app.get("/api/models")
async def get_models():
    """Returns the list of available models from the config for the frontend dropdown."""
    config = load_config()
    models_conf = config.get("models", {})
    
    api_call = models_conf.get("api_call", {})
    # Filter out empty list items inside serverless config
    serverless = [m for m in models_conf.get("serverless_models", []) if m]

    return {
        "api_gemini": api_call.get("gemini", []),
        "api_groq": api_call.get("groq", []),
        "oss_local": models_conf.get("oss_models", []),
        "oss_serverless": serverless
    }


@app.get("/api/eval/prompts")
async def get_eval_prompts_endpoint():
    """Returns the pre-built evaluation test suites with prompts."""
    return {"success": True, "prompts": get_eval_prompts()}


def _collect_model_response(model: str, prompt: str) -> str:
    """
    Synchronously collect the full (non-streamed) response from any model.
    Reuses the same routing logic as the chat endpoint.
    """
    system_prompt = "You are a helpful personal assistant."
    content = f"Human Query: {prompt}\nAI response:"

    config = load_config()
    models_conf = config.get("models", {})
    api_calls = models_conf.get("api_call", {})
    gemini_models = api_calls.get("gemini", [])
    groq_models = api_calls.get("groq", [])
    oss_models = models_conf.get("oss_models", [])
    serverless_models = models_conf.get("serverless_models", [])

    generator = None

    if model in gemini_models:
        generator = get_gemini_chat_completion(system_prompt, model, content)
    elif model in groq_models:
        generator = get_groq_chat_completion(system_prompt, model, content)
    elif model in oss_models:
        generator = get_local_qwen(system_prompt, model, content)
    elif model in serverless_models:
        if "qwen" in model.lower():
            generator = get_modal_qwen(system_prompt, content)
        elif "mistral" in model.lower():
            generator = get_modal_mistral(system_prompt, content)
        else:
            raise ValueError(f"Serverless model '{model}' router not implemented.")
    else:
        raise ValueError(f"Model '{model}' is not found in configuration.")

    if not generator:
        raise ValueError("Failed to initialize generator.")

    # Collect all tokens into full text
    full_text = ""
    for chunk in generator:
        full_text += chunk
    return full_text


class EvalSingleRequest(BaseModel):
    model_a: str
    model_b: str
    prompt: str
    category: str
    ground_truth: Optional[str] = None
    expected_behavior: Optional[str] = None


@app.post("/api/eval/single")
async def eval_single_endpoint(request: EvalSingleRequest):
    """
    Evaluate a single prompt against two models.
    Gets responses from both, then scores them with category-appropriate evaluator.
    """
    try:
        # Get responses from both models
        response_a = _collect_model_response(request.model_a, request.prompt)
        response_b = _collect_model_response(request.model_b, request.prompt)

        # Score based on category
        if request.category == "hallucination":
            gt = request.ground_truth or ""
            score_a = eval_hallucination(request.prompt, response_a, gt)
            score_b = eval_hallucination(request.prompt, response_b, gt)
        elif request.category == "bias":
            eb = request.expected_behavior or ""
            score_a = eval_bias(request.prompt, response_a, eb)
            score_b = eval_bias(request.prompt, response_b, eb)
        elif request.category == "content_safety":
            eb = request.expected_behavior or ""
            score_a = eval_content_safety(request.prompt, response_a, eb)
            score_b = eval_content_safety(request.prompt, response_b, eb)
        else:
            # Fallback to general evaluator
            score_val_a = eval_llm_call(request.prompt, response_a)
            score_val_b = eval_llm_call(request.prompt, response_b)
            score_a = {"score": int(score_val_a), "reasoning": "General evaluation"}
            score_b = {"score": int(score_val_b), "reasoning": "General evaluation"}

        return {
            "success": True,
            "prompt": request.prompt,
            "category": request.category,
            "model_a": {
                "model": request.model_a,
                "response": response_a,
                "score": score_a.get("score", -1),
                "reasoning": score_a.get("reasoning", "")
            },
            "model_b": {
                "model": request.model_b,
                "response": response_b,
                "score": score_b.get("score", -1),
                "reasoning": score_b.get("reasoning", "")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/eval")
async def eval_endpoint(question: str, llm_response: str):
    """Endpoint to evaluate an LLM response using the eval_llm_call function."""
    try:
        score = eval_llm_call(question, llm_response)
        return {"success": True, "score": score}
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))