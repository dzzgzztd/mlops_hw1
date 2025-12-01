FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir "dvc[s3]"

COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

RUN mkdir -p logs saved_models app/grpc/generated

RUN python scripts/generate_grpc_code.py

RUN git config --global user.email "ml-service@example.com" && \
    git config --global user.name "ML Service"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]