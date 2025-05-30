"""
FastAPI application entry point for Nautilus API.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.router import api_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Nautilus API",
    description="API for Nautilus Jules backend. See /docs for OpenAPI documentation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", tags=["root"])
def root():
    """
    Root endpoint for health check.
    """
    return {"message": "Welcome to the Nautilus API. See /docs for documentation."}
