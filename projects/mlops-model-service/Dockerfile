# syntax=docker/dockerfile:1.6

FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim AS runtime

RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ ./app/
COPY models/ ./models/

USER appuser

EXPOSE 8080

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/best

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
