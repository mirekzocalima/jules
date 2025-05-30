"""
Vessels API endpoints for listing vessels, types, and draughts.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from backend.jules_core import NautilusVessels

router = APIRouter()
logger = logging.getLogger("vessels")

class VesselResponse(BaseModel):
    name: str
    type: str
    draughts: List[str]

@router.get("/vessels", response_model=List[str], summary="List available vessels", tags=["vessels"])
def list_vessels():
    """
    List all available vessels.
    """
    try:
        vessels = NautilusVessels()
        vessel_names = vessels.list_vessels()
        logger.info("Vessels listed.")
        return vessel_names
    except Exception as e:
        logger.error(f"Error listing vessels: {e}")
        raise HTTPException(status_code=500, detail="Failed to list vessels")

@router.post("/vessels/types", response_model=List[str], summary="List vessel types for a vessel", tags=["vessels"])
def list_vessel_types(vessel: str):
    """
    List vessel types for a given vessel.
    """
    try:
        vessels = NautilusVessels()
        types = vessels.list_vessel_types(vessel)
        logger.info(f"Vessel types for {vessel} listed.")
        return types
    except Exception as e:
        logger.error(f"Error listing vessel types: {e}")
        raise HTTPException(status_code=500, detail="Failed to list vessel types")

@router.post("/vessels/draughts", response_model=List[str], summary="List draughts for a vessel and type", tags=["vessels"])
def list_draughts(vessel: str, vessel_type: str):
    """
    List draughts for a given vessel and type.
    """
    try:
        vessels = NautilusVessels()
        draughts = vessels.list_draughts(vessel, vessel_type)
        logger.info(f"Draughts for {vessel} ({vessel_type}) listed.")
        return draughts
    except Exception as e:
        logger.error(f"Error listing draughts: {e}")
        raise HTTPException(status_code=500, detail="Failed to list draughts")
