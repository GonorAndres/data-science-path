#!/usr/bin/env bash
# Simple load test using `hey`. Install: brew install hey OR go install github.com/rakyll/hey@latest

set -euo pipefail

: "${URL:?URL is required (e.g. http://localhost:8080)}"

echo "Sending 200 requests, 10 concurrent, to ${URL}/predict ..."

hey -n 200 -c 10 \
  -m POST \
  -H "Content-Type: application/json" \
  -d '{"text": "El producto es excelente, lo recomiendo."}' \
  "${URL}/predict"
