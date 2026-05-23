import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

logger = logging.getLogger(__name__)

# Global cache to prevent reloading the model on every request
_model = None
_tokenizer = None
_loaded_model_name = None

def get_qwen_chat_completion(system_prompt, model, content):
    global _model, _tokenizer, _loaded_model_name
    
    try:
        # Lazy load the model (only loads on the first request or if the model changes)
        if _model is None or _loaded_model_name != model:
            logger.info(f"Loading local model {model} into memory. This may take a moment...")
            _tokenizer = AutoTokenizer.from_pretrained(model)
            _model = AutoModelForCausalLM.from_pretrained(
                model, 
                torch_dtype="auto", 
                device_map="auto"
            )
            _loaded_model_name = model
    except Exception as load_err:
        logger.error(f"Failed to load model {model}: {load_err}")
        raise load_err

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        # Format the input using the model's chat template
        text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = _tokenizer([text], return_tensors="pt").to(_model.device)
    except Exception as preprocess_err:
        logger.error(f"Error during prompt preprocessing for {model}: {preprocess_err}")
        raise preprocess_err
    
    # Set up the streamer
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.7,
        pad_token_id=_tokenizer.eos_token_id
    )
    
    try:
        # Generate in a separate thread so we can yield chunks from the streamer on the main thread
        thread = Thread(target=_model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
    except Exception as gen_err:
        logger.error(f"Error during Qwen token generation: {gen_err}")
        raise gen_err

