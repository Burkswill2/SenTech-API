from fastapi import APIRouter, Depends, HTTPException
from .helpers import markowitz_helper

# from ..dependencies import get_token_header

router = APIRouter(
    prefix="/markowitz_model",
    tags=["markowitz_model"],
    # dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)


fake_items_db = {"plumbus": {"name": "Plumbus"}, "gun": {"name": "Portal Gun"}}


@router.get("/")
async def getOptimalPortfolio():
    result = markowitz_helper.start_model()
    print(result)
    return {"result": result}
