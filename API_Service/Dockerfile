# Dockerfile

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

ENV MODEL_PATH="models/final_model.joblib"
ENV MODEL_VERSION="v1"
ENV PORT=8080

CMD ["uvicorn", "ml_src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]