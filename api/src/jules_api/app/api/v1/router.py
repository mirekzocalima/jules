"""
API v1 router for all endpoints.
"""
from fastapi import APIRouter
from jules_api.app.api.v1.endpoints import models, operability, rao, cables, vessels, auth

api_router = APIRouter()

api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(operability.router, prefix="/operability", tags=["operability"])
api_router.include_router(rao.router, tags=["rao"])
api_router.include_router(cables.router, tags=["cables"])
api_router.include_router(vessels.router, tags=["vessels"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
