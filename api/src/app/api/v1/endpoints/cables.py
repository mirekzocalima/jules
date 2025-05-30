"""
Cables API endpoints for listing and retrieving cable records.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
from backend.jules_core import NautilusCables

router = APIRouter()
logger = logging.getLogger("cables")

class CableResponse(BaseModel):
    name: str
    type: str
    data: Dict[str, Any]

@router.get("/cables", response_model=Dict[str, list], summary="List available cables", tags=["cables"])
def list_cables():
    """
    List all available cables.
    """
    try:
        cables = NautilusCables()
        cable_names = cables.cable_names()
        logger.info("Cables listed.")
        return {"cables": cable_names}
    except Exception as e:
        logger.error(f"Error listing cables: {e}")
        raise HTTPException(status_code=500, detail="Failed to list cables")

@router.post("/cables/{cable_type}", response_model=CableResponse, summary="Get cable record", tags=["cables"])
def get_cable_record(cable_type: str):
    """
    Get cable record for a given cable type.
    """
    try:
        cables = NautilusCables()
        record = cables.get_record(cable_type, kind="raw")
        logger.info(f"Cable record for {cable_type} retrieved.")
        return {"name": cable_type, "type": "raw", "data": record}
    except Exception as e:
        logger.error(f"Error retrieving cable record: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cable record")
