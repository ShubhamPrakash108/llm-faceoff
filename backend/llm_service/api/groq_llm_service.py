import os
import logging
from groq import Groq
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

API_KEY_1 = os.getenv("GROQ_API_KEY_1", "your-key-1-here")
API_KEY_2 = os.getenv("GROQ_API_KEY_2", "your-key-2-here")

def get_groq_chat_completion(system_prompt, model, content):
    def try_generate(api_key):
        try:
            client = Groq(api_key=api_key)
            stream = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model=model,
                stream=True,
            )

            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                yield token
        except Exception as e:
            logger.error(f"Error in Groq generation: {e}")
            raise e

    try:
        yield from try_generate(API_KEY_1)
    except Exception as e:
        logger.warning(f"First Groq API failed ({e}), switching to secondary...")
        try:
            yield from try_generate(API_KEY_2)
        except Exception as e2:
            logger.error(f"Secondary Groq API also failed processing: {e2}")
            raise e2