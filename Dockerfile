FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.lock

COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

COPY .dvc/ ./.dvc/
COPY .dvcignore ./.dvcignore

RUN dvc config core.no_scm true

COPY clearml.conf /app/clearml.conf
ENV CLEARML_CONFIG_FILE=/app/clearml.conf

RUN mkdir -p logs saved_models app/grpc/generated

RUN python -m pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1

RUN python scripts/generate_grpc_code.py

RUN git config --global user.email "ml-service@example.com" && \
    git config --global user.name "ML Service"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]