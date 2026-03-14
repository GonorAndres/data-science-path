"""FastAPI application for insurance pricing dashboard."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pricing_backend.routers import overview, eda, models

app = FastAPI(
    title="Insurance Pricing API",
    description="Pre-computed actuarial model results for the insurance pricing dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3060",
        "http://127.0.0.1:3060",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(overview.router, prefix="/api/v1")
app.include_router(eda.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
