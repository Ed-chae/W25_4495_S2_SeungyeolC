from fastapi import APIRouter
from app.services.predictive_analytics import analyze_revenue_trends, predict_future_revenue

router = APIRouter()

@router.get("/revenue-trends/")
async def get_revenue_trends():
    return analyze_revenue_trends()

@router.get("/predict-revenue/")
async def get_revenue_forecast():
    return predict_future_revenue()
