#!/usr/bin/env bash
# Deploy the service to Cloud Run.
# Required env vars: PROJECT, SERVICE, REGION.

set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${SERVICE:=nlp-reviews}"
: "${REGION:=us-central1}"

IMAGE="gcr.io/${PROJECT}/${SERVICE}:$(git rev-parse --short HEAD 2>/dev/null || date +%s)"

echo "Building image ${IMAGE}..."
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT}"

echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --project "${PROJECT}" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 60s \
  --port 8080

URL=$(gcloud run services describe "${SERVICE}" \
  --region "${REGION}" \
  --project "${PROJECT}" \
  --format 'value(status.url)')

echo
echo "Deployed."
echo "Service URL: ${URL}"
echo "Test:"
echo "  curl ${URL}/health"
echo "  curl -X POST ${URL}/predict -H 'Content-Type: application/json' -d '{\"text\":\"Me encanto!\"}'"
