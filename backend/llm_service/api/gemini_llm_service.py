import os
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

API_KEY_1 = os.getenv("GEMINI_API_KEY_1", "your-key-1-here")
API_KEY_2 = os.getenv("GEMINI_API_KEY_2", "your-key-2-here")

def get_gemini_chat_completion(system_prompt, model, content):
    def try_generate(api_key):
        try:
            client = genai.Client(api_key=api_key)
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=content,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            ):
                token = chunk.text or ""
                yield token
        except Exception as e:
            logger.error(f"Error in Gemini generation: {e}")
            raise e

    try:
        yield from try_generate(API_KEY_1)
    except Exception as e:
        logger.warning(f"First Gemini API failed ({e}), switching to secondary...")
        try:
            yield from try_generate(API_KEY_2)
        except Exception as e2:
            logger.error(f"Secondary Gemini API also failed processing: {e2}")
            raise e2