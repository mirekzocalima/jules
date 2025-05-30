"""
Models API endpoints for cable types and predictions.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from backend.jules_core import list_available_cable_types, NautilusModel

router = APIRouter()
logger = logging.getLogger("models")

class CableTypeResponse(BaseModel):
    cable_types: List[str]

class PredictionRequest(BaseModel):
    features: Dict[str, float]
    outcomes: Optional[Dict[str, float]] = None

class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
    truth: Optional[Dict[str, float]] = None

@router.get("/cables", response_model=CableTypeResponse, summary="List available cable types", tags=["models"])
def get_cable_types():
    """
    List all available cable types.
    """
    try:
        cable_types = list_available_cable_types()
        logger.info("Cable types listed.")
        return {"cable_types": cable_types}
    except Exception as e:
        logger.error(f"Error listing cable types: {e}")
        raise HTTPException(status_code=500, detail="Failed to list cable types")

@router.post("/cables/{cable_type}/predict", response_model=PredictionResponse, summary="Predict outcomes for given features", tags=["models"])
def predict_cable(cable_type: str, req: PredictionRequest):
    """
    Predict outcomes for a given cable type and features.
    """
    try:
        model = NautilusModel(version="2025-05-23", cable_type=cable_type)
        result = model.predict(req.features, req.outcomes)
        logger.info(f"Prediction for cable {cable_type} done.")
        return {"predictions": result, "truth": req.outcomes}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
