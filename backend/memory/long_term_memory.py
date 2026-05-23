import logging
import os
from groq import Groq
from backend.database_mangement.supabase_db import get_user_memory, save_user_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY_1"))

MEMORY_EXTRACTION_MODEL = "llama-3.3-70b-versatile"
MAX_MEMORY_ITEMS = 30  # cap to prevent token overflow

EXTRACTION_PROMPT = """You are a memory extraction assistant. Given a conversation exchange between a user and an AI, extract important personal facts about the USER only.

Extract things like: name, age, job, location, preferences, hobbies, technical skills, goals, family details, or any other permanent/semi-permanent facts.

Rules:
- Return ONLY a JSON array of short fact strings. Example: ["User's name is Shubham", "Works as a software engineer", "Prefers Python"]
- If no new facts are found, return an empty array: []
- Do NOT include facts about the AI or general knowledge
- Do NOT repeat facts that already exist in the existing memory
- Keep each fact concise (under 15 words)
- Return ONLY the JSON array, nothing else"""


def extract_memory_from_exchange(user_message: str, ai_message: str, existing_memory: list) -> list:
    """
    Uses an LLM to extract key user facts from a conversation exchange.
    Returns a list of new fact strings to add to memory.
    """
    try:
        existing_str = "\n".join(f"- {fact}" for fact in existing_memory) if existing_memory else "None yet"
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Existing memory about the user:\n{existing_str}\n\nNew conversation:\nUser: {user_message}\nAI: {ai_message}"}
            ],
            model=MEMORY_EXTRACTION_MODEL,
            temperature=0.1,
            max_tokens=300,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the JSON array from the response
        import json
        # Handle cases where LLM wraps in markdown code block
        if result.startswith("```"):
            result = result.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        
        new_facts = json.loads(result)
        
        if isinstance(new_facts, list):
            # Filter out empty strings and duplicates
            new_facts = [f.strip() for f in new_facts if isinstance(f, str) and f.strip()]
            # Remove facts that are already in existing memory (case-insensitive)
            existing_lower = {m.lower() for m in existing_memory}
            new_facts = [f for f in new_facts if f.lower() not in existing_lower]
            return new_facts
        
        return []
    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")
        return []


def update_user_long_term_memory(user_id: str, user_message: str, ai_message: str) -> list:
    """
    Extracts new facts from the conversation and updates the user's 
    long-term memory in the database. Returns the updated memory list.
    """
    # Get existing memory from DB
    existing_memory = get_user_memory(user_id)
    
    # Extract new facts
    new_facts = extract_memory_from_exchange(user_message, ai_message, existing_memory)
    
    if new_facts:
        # Merge and cap
        updated_memory = existing_memory + new_facts
        if len(updated_memory) > MAX_MEMORY_ITEMS:
            updated_memory = updated_memory[-MAX_MEMORY_ITEMS:]
        
        # Save to DB
        save_user_memory(user_id, updated_memory)
        logger.info(f"Added {len(new_facts)} new memory items for user {user_id}: {new_facts}")
        return updated_memory
    
    return existing_memory


def build_system_prompt_with_memory(base_prompt: str, memory: list) -> str:
    """
    Builds a system prompt that includes the user's long-term memory context.
    """
    if not memory:
        return base_prompt
    
    memory_str = "\n".join(f"• {fact}" for fact in memory)
    return f"""{base_prompt}

You have the following long-term memory about this user from previous conversations. Use these facts to personalize your responses and maintain continuity:

{memory_str}

Use this context naturally — don't explicitly mention that you have a memory system unless asked."""