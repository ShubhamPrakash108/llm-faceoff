import os
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

MODAL_QWEN_BASE_URL = os.environ.get("MODAL_VLLM_QWEN_URL")
MODAL_MISTRAL_BASE_URL = os.environ.get("MODAL_VLLM_MISTRAL_URL")

qwen_client = OpenAI(base_url=MODAL_QWEN_BASE_URL, api_key="unused")
mistral_client = OpenAI(base_url=MODAL_MISTRAL_BASE_URL, api_key="unused")


def get_qwen_chat_completion(system_prompt: str, content: str):
    """Token-by-token streaming from the Modal-hosted vLLM server for Qwen."""
    stream = qwen_client.chat.completions.create(
        model=QWEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ],
        stream=True,
        max_tokens=1024,
        temperature=0.7,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        yield token


def get_mistral_chat_completion(system_prompt: str, content: str):
    """Token-by-token streaming from the Modal-hosted vLLM server for Mistral."""
    stream = mistral_client.chat.completions.create(
        model=MISTRAL_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ],
        stream=True,
        max_tokens=1024,
        temperature=0.7,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        yield token


if __name__ == "__main__":
    print("── Testing Qwen Model ──")
    try:
        reply_qwen = get_qwen_chat_completion(
            system_prompt="You are a helpful personal assistant.",
            content="What are three tips for staying productive when working from home?",
        )
    except Exception as e:
        print(f"Skipping Qwen due to error: {e}")
        
    print("\n── Testing Mistral Model ──")
    try:
        reply_mistral = get_mistral_chat_completion(
            system_prompt="You are a helpful personal assistant.",
            content="What are three tips for staying productive when working from home?",
        )
    except Exception as e:
        print(f"Skipping Mistral due to error: {e}")
