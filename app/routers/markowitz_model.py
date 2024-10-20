from fastapi import APIRouter, Depends, HTTPException
from .helpers import markowitz_helper
from pydantic import BaseModel
from typing import List

class PortfolioParams(BaseModel):
    stocks: List[str]
    start_date: str
    end_date: str

router = APIRouter(
    prefix="/markowitz_model",
    tags=["markowitz_model"],
    # dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)


@router.post("/")  # Change to POST
async def getOptimalPortfolio(params: PortfolioParams):
    # Convert Pydantic model to dictionary
    params_dict = params.model_dump()
    result = markowitz_helper.start_model(params_dict)
    return {"result": result}
