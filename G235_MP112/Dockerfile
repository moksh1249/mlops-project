FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# dependencies first, so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# then the code and the champion model
COPY src/ ./src/
COPY app/ ./app/
COPY models/model.joblib ./models/model.joblib
COPY params.yaml .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
