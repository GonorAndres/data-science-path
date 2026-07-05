"""Prometheus metrics for the service."""

from prometheus_client import Counter, Histogram, generate_latest


REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    ["endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions made.",
    ["label"],
)

INPUT_LENGTH = Histogram(
    "input_length_chars",
    "Input length in characters.",
    buckets=(50, 100, 200, 400, 800, 1600, 3200),
)


def render_metrics() -> bytes:
    return generate_latest()
