from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Response
from sqlalchemy.orm import Session
from pathlib import Path

from db import SessionLocal, SalesData, RestaurantOrder
from file_processing import process_sales_data
from sentiment_analysis import analyze_sentiment
from revenue_forecasting import forecast_revenue
from weather_analysis import predict_revenue_impact
from customer_segmentation import segment_customers
from demand_forecasting import forecast_demand
from anomaly_detection import detect_sales_anomalies
from product_recommendation import recommend_products
from market_basket_analysis import market_basket_analysis

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/")
def root():
    return {"message": "Welcome to the Intelligent Business Analytics API!"}


# -----------------------------------
# 🌐 CORS Preflight Handler
# -----------------------------------
@router.options("/{rest_of_path:path}", include_in_schema=False)
async def preflight_handler(rest_of_path: str, response: Response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return Response(status_code=200)


# -----------------------------------
# 📂 File Upload API
# -----------------------------------
@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel files are allowed.")

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        df = process_sales_data(file_path)
        return {"message": "File uploaded and processed successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# -----------------------------------
# 🧼 Reset Database APIs
# -----------------------------------
@router.delete("/reset-sales-data/")
def reset_sales_data():
    db = SessionLocal()
    db.query(SalesData).delete()
    db.commit()
    db.close()
    return {"message": "All sales data deleted."}


@router.delete("/reset-restaurant-orders/")
def reset_restaurant_orders():
    db = SessionLocal()
    db.query(RestaurantOrder).delete()
    db.commit()
    db.close()
    return {"message": "All restaurant order data deleted."}


# -----------------------------------
# 📊 Sentiment Analysis API
# -----------------------------------
@router.get("/sentiment-results/")
def get_sentiment_results(db: Session = Depends(get_db)):
    """Returns reviews and sentiment for each item (standard + restaurant)."""
    sales_reviews = db.query(SalesData).all()
    restaurant_reviews = db.query(RestaurantOrder).all()

    results = []

    for r in sales_reviews:
        results.append({
            "item": r.product,
            "review": r.review,
            "sentiment": analyze_sentiment(r.review)["label"]
        })

    for r in restaurant_reviews:
        results.append({
            "item": r.menu_item,
            "review": r.review,
            "sentiment": analyze_sentiment(r.review)["label"]
        })

    # Optional summary per item
    summary = {}
    for entry in results:
        item = entry["item"]
        sentiment = entry["sentiment"]
        if item not in summary:
            summary[item] = {"positive": 0, "negative": 0}
        if sentiment == "POSITIVE":
            summary[item]["positive"] += 1
        else:
            summary[item]["negative"] += 1

    summary_table = [
        {
            "item": item,
            "positive": data["positive"],
            "negative": data["negative"],
            "summary": f"{item} - {round((data['negative'] / (data['positive'] + data['negative']) * 100) if (data['positive'] + data['negative']) > 0 else 0)}% negative"
        }
        for item, data in summary.items()
    ]

    return {
        "details": results,
        "summary": summary_table
    }


# -----------------------------------
# 📈 Revenue Forecasting
# -----------------------------------
@router.get("/revenue-forecast/")
def get_revenue_forecast():
    return forecast_revenue()


# -----------------------------------
# ⛅ Weather Impact on Sales
# -----------------------------------
@router.get("/weather-impact/")
def get_weather_impact(city: str = "Vancouver"):
    return predict_revenue_impact(city)


# -----------------------------------
# 🛍️ Customer Segmentation
# -----------------------------------
@router.get("/customer-segmentation/")
def get_customer_segments():
    return segment_customers()


# -----------------------------------
# 📊 Demand Forecasting
# -----------------------------------
@router.get("/demand-forecast/")
def get_demand_forecast():
    return forecast_demand()


# -----------------------------------
# 🚨 Anomaly Detection
# -----------------------------------
@router.get("/sales-anomalies/")
def get_sales_anomalies():
    return detect_sales_anomalies()


# -----------------------------------
# 🎯 Product Recommendation
# -----------------------------------
@router.get("/product-recommendations/")
def get_product_recommendations(user_id: int):
    return recommend_products(user_id)


# -----------------------------------
# 🛒 Market Basket Analysis
# -----------------------------------
@router.get("/market-basket/")
def get_market_basket_analysis():
    return market_basket_analysis()
