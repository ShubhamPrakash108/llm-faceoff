import modal

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "huggingface-hub==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol   = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache",        create_if_missing=True)

QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

VLLM_PORT  = 8000
MINUTES    = 60

app = modal.App("oss-personal-assistant")

@app.function(
    image=vllm_image,
    gpu="A10G",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm":        vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=50)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve_qwen():
    import subprocess

    cmd = [
        "vllm", "serve", QWEN_MODEL_NAME,
        "--served-model-name", QWEN_MODEL_NAME,
        "--host",              "0.0.0.0",
        "--port",              str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--max-model-len",     "8192",
        "--enforce-eager",
        "--uvicorn-log-level", "info",
    ]

    print("Starting vLLM Qwen Server:", " ".join(cmd))
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    gpu="A10G",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm":        vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=50)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve_mistral():
    import subprocess

    cmd = [
        "vllm", "serve", MISTRAL_MODEL_NAME,
        "--served-model-name", MISTRAL_MODEL_NAME,
        "--host",              "0.0.0.0",
        "--port",              str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--max-model-len",     "8192",
        "--enforce-eager",
        "--uvicorn-log-level", "info",
    ]

    print("Starting vLLM Mistral Server:", " ".join(cmd))
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main():
    from openai import OpenAI
    import os

    qwen_base_url = os.environ.get(
        "MODAL_VLLM_QWEN_URL",
        "https://<your-workspace>--oss-personal-assistant-serve-qwen.modal.run/v1",
    )
    mistral_base_url = os.environ.get(
        "MODAL_VLLM_MISTRAL_URL",
        "https://<your-workspace>--oss-personal-assistant-serve-mistral.modal.run/v1",
    )

    print("\n── Testing Qwen Model ──")
    client_qwen = OpenAI(base_url=qwen_base_url, api_key="unused")

    try:
        stream_qwen = client_qwen.chat.completions.create(
            model=QWEN_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful personal assistant."},
                {"role": "user",   "content": "Summarise the benefits of serverless GPU inference."},
            ],
            stream=True,
            max_tokens=64,
        )

        for chunk in stream_qwen:
            token = chunk.choices[0].delta.content or ""
            print(token, end="", flush=True)
        print()
    except Exception as e:
        print(f"Skipping Qwen due to error: {e}")

    print("\n── Testing Mistral Model ──")
    client_mistral = OpenAI(base_url=mistral_base_url, api_key="unused")

    try:
        stream_mistral = client_mistral.chat.completions.create(
            model=MISTRAL_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful personal assistant."},
                {"role": "user",   "content": "Summarise the benefits of serverless GPU inference."},
            ],
            stream=True,
            max_tokens=64,
        )

        for chunk in stream_mistral:
            token = chunk.choices[0].delta.content or ""
            print(token, end="", flush=True)
        print()
    except Exception as e:
        print(f"Skipping Mistral due to error: {e}")
