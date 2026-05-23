FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --target=/build/deps \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --target=/build/deps -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN addgroup --system app && \
    adduser --system --ingroup app --home /home/app app

COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY config/ ./config/
COPY agent/ ./agent/
COPY eval/ ./eval/

RUN mkdir -p /home/app/.cache/huggingface && \
    chown -R app:app /home/app /app

ENV HF_HOME=/home/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/app/.cache/huggingface

USER app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.model_apis_endpoint.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
