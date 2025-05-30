"""
RAO API endpoints for calculation and plotting.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, Tuple
import logging
from backend.jules_core import RAOCalculator

router = APIRouter()
logger = logging.getLogger("rao")

class RAORequest(BaseModel):
    vessel: str
    vessel_type: str
    draught: str
    wave_direction: float
    wave_period: float
    wave_amplitude: float
    point: Optional[Tuple[float, float, float]] = None

class RAOResponse(BaseModel):
    data: Dict[str, Any]
    plot: Optional[Dict[str, Any]] = None

@router.post("/raos", response_model=RAOResponse, summary="Calculate RAO values", tags=["rao"])
def calculate_rao(req: RAORequest):
    """
    Calculate RAO values for given vessel and wave parameters.
    """
    try:
        rao_calc = RAOCalculator(req.vessel, req.vessel_type, req.draught)
        data = rao_calc.calculate(
            req.wave_direction, req.wave_period, req.wave_amplitude, req.point
        )
        logger.info("RAO calculation done.")
        return {"data": data}
    except Exception as e:
        logger.error(f"RAO calculation error: {e}")
        raise HTTPException(status_code=500, detail="RAO calculation failed")

@router.post("/raos/plot", response_model=RAOResponse, summary="Generate RAO plot", tags=["rao"])
def plot_rao(req: RAORequest):
    """
    Generate RAO plot for given vessel and wave parameters.
    """
    try:
        rao_calc = RAOCalculator(req.vessel, req.vessel_type, req.draught)
        plot = rao_calc.plot(
            req.wave_direction, req.wave_period, req.wave_amplitude, req.point
        )
        logger.info("RAO plot generated.")
        return {"data": {}, "plot": plot}
    except Exception as e:
        logger.error(f"RAO plot error: {e}")
        raise HTTPException(status_code=500, detail="RAO plot failed")
