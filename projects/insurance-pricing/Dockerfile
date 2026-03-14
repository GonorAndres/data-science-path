FROM python:3.11-slim
WORKDIR /app

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/pricing_backend/ ./pricing_backend/
COPY backend/data/ ./data/

EXPOSE 8080
CMD ["uvicorn", "pricing_backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
