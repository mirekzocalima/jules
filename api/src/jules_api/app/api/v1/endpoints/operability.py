"""
Operability API endpoints for heatmap and matrix.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Tuple
import logging
from backend.jules_core import get_operability_heatmap, get_operability_matrix

router = APIRouter()
logger = logging.getLogger("operability")

class OperabilityRequest(BaseModel):
    model_version: str
    cable_type: str
    vessel: str
    vessel_type: str
    draught: str
    point: Tuple[float, float, float]
    features: Dict[str, Any]
    mbr_min: float
    tension_tdp_min: float
    tension_tdp_max: float

class OperabilityResponse(BaseModel):
    matrix: Dict[str, Any]
    heatmap: Dict[str, Any]

@router.post("/heatmap", response_model=OperabilityResponse, summary="Get operability heatmap", tags=["operability"])
def get_heatmap(req: OperabilityRequest):
    """
    Get operability heatmap for given parameters.
    """
    try:
        heatmap = get_operability_heatmap(
            req.model_version, req.cable_type, req.vessel, req.vessel_type, req.draught,
            req.point, req.features, req.mbr_min, req.tension_tdp_min, req.tension_tdp_max
        )
        logger.info("Operability heatmap generated.")
        return {"matrix": {}, "heatmap": heatmap}
    except Exception as e:
        logger.error(f"Heatmap error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate heatmap")

@router.post("/matrix", response_model=OperabilityResponse, summary="Get operability matrix", tags=["operability"])
def get_matrix(req: OperabilityRequest):
    """
    Get operability matrix for given parameters.
    """
    try:
        matrix = get_operability_matrix(
            req.model_version, req.cable_type, req.vessel, req.vessel_type, req.draught,
            req.point, req.features, req.mbr_min, req.tension_tdp_min, req.tension_tdp_max
        )
        logger.info("Operability matrix generated.")
        return {"matrix": matrix, "heatmap": {}}
    except Exception as e:
        logger.error(f"Matrix error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate matrix")
