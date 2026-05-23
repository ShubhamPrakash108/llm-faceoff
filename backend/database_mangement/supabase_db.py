import os
import uuid
import logging
import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_PROJECT_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY", "")

try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    else:
        supabase = None
        logger.warning("Supabase URL or Key is missing from environment variables.")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    supabase = None


def register_user(user_id: str, password: str) -> bool:
    """
    Registers a new user in the user_info table.
    """
    if not supabase:
        logger.error("Supabase client is not initialized.")
        return False
        
    try:
        # Check if user already exists
        response = supabase.table("user_info").select("*").eq("user_id", user_id).execute()
        if response.data:
            logger.warning(f"User {user_id} already exists.")
            return False
            
        data = {
            "user_id": user_id,
            "password": password
        }
        supabase.table("user_info").insert(data).execute()
        logger.info(f"User {user_id} registered successfully.")
        return True
    except Exception as e:
        logger.error(f"Error registering user {user_id}: {e}")
        raise e


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user against the user_info table.
    """
    if not supabase:
        logger.error("Supabase client is not initialized.")
        return False
        
    try:
        response = supabase.table("user_info").select("*").eq("user_id", user_id).eq("password", password).execute()
        if response.data and len(response.data) > 0:
            logger.info(f"User {user_id} authenticated successfully.")
            return True
        else:
            logger.warning(f"Authentication failed for user {user_id}.")
            return False
    except Exception as e:
        logger.error(f"Error authenticating user {user_id}: {e}")
        raise e


def save_chat_generation(user_id: str, chat_id: str, user_message: str, ai_message: str) -> str:
    """
    Saves a user and AI message exchange to the user_generation table.
    Each call inserts a new row with a unique generation_id (the primary key).
    The chat column stores the user+assistant message pair for this exchange.
    """
    if not supabase:
        logger.error("Supabase client is not initialized.")
        return None
        
    generation_id = str(uuid.uuid4())
    
    chat_messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ai_message}
    ]
    
    try:
        row = {
            "user_id": user_id,
            "generation_id": generation_id,
            "chat": chat_messages,
            "generated_at": datetime.datetime.utcnow().isoformat()
        }
        
        supabase.table("user_generation").insert(row).execute()
        
        logger.info(f"Saved chat generation {generation_id} for user {user_id}.")
        return generation_id
    except Exception as e:
        logger.error(f"Error saving chat for user {user_id}: {e}")
        raise e


def fetch_user_chats(user_id: str) -> list:
    """
    Fetches all chat generations for a specific user.
    """
    if not supabase:
        logger.error("Supabase client is not initialized.")
        return []
        
    try:
        response = supabase.table("user_generation").select("*").eq("user_id", user_id).order("generated_at", desc=False).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching chats for user {user_id}: {e}")
        raise e


def get_user_memory(user_id: str) -> list:
    """
    Fetches the long_term_memory JSONB from user_info for the given user.
    Returns a list of memory fact strings, or an empty list if none exist.
    """
    if not supabase:
        logger.error("Supabase client is not initialized.")
        return []
    
    try:
        response = supabase.table("user_info").select("long_term_memory").eq("user_id", user_id).execute()
        if response.data and len(response.data) > 0:
            memory = response.data[0].get("long_term_memory")
            if memory and isinstance(memory, list):
                return memory
        return []
    except Exception as e:
        logger.error(f"Error fetching memory for user {user_id}: {e}")
        return []


def save_user_memory(user_id: str, memory: list) -> bool:
    """
    Saves the long_term_memory JSONB for the given user.
    Memory is a list of fact strings.
    """
    if not supabase:
        logger.error("Supabase client is not initialized.")
        return False
    
    try:
        supabase.table("user_info").update({
            "long_term_memory": memory
        }).eq("user_id", user_id).execute()
        logger.info(f"Saved {len(memory)} memory items for user {user_id}.")
        return True
    except Exception as e:
        logger.error(f"Error saving memory for user {user_id}: {e}")
        raise e

