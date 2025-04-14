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
from product_recommendation import recommend_products
from market_basket_analysis import market_basket_analysis
from menu_category_analysis import categorize_menu_items_hf
from collections import defaultdict, Counter


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
def get_sentiment_summary(db: Session = Depends(get_db)):
    """Returns summarized sentiment per item, including most common review."""
    sales_reviews = db.query(SalesData).all()
    restaurant_reviews = db.query(RestaurantOrder).all()

    summary = defaultdict(lambda: {
        "positive": 0,
        "negative": 0,
        "reviews": []
    })

    # Analyze sentiment and gather reviews
    for r in sales_reviews:
        item = r.product
        if item and r.review:
            sentiment = analyze_sentiment(r.review)["label"]
            summary[item]["reviews"].append(r.review.strip())
            if sentiment == "POSITIVE":
                summary[item]["positive"] += 1
            else:
                summary[item]["negative"] += 1

    for r in restaurant_reviews:
        item = r.menu_item
        if item and r.review:
            sentiment = analyze_sentiment(r.review)["label"]
            summary[item]["reviews"].append(r.review.strip())
            if sentiment == "POSITIVE":
                summary[item]["positive"] += 1
            else:
                summary[item]["negative"] += 1

    result = []
    for item, stats in summary.items():
        total = stats["positive"] + stats["negative"]
        negative_percent = (stats["negative"] / total * 100) if total else 0
        most_common_review = Counter(stats["reviews"]).most_common(1)
        result.append({
            "item": item,
            "positive": stats["positive"],
            "negative": stats["negative"],
            "total_reviews": total,
            "most_common_review": most_common_review[0][0] if most_common_review else "",
            "summary": f"{round(negative_percent, 1)}% negative"
        })

    return {"summary": result}


# -----------------------------------
# 📈 Revenue Forecasting
# -----------------------------------
@router.get("/revenue-forecast/")
def get_revenue_forecast():
    try:
        return forecast_revenue()
    except Exception as e:
        return {"prophet_forecast": [], "lstm_forecast": [], "message": str(e)}


# -----------------------------------
# ⛅ Weather Impact on Sales
# -----------------------------------
@router.get("/weather-impact/")
def get_weather_impact(city: str = "Vancouver"):
    try:
        return predict_revenue_impact(city)
    except Exception as e:
        return {"error": str(e)}


# -----------------------------------
# 🛍️ Customer Segmentation
# -----------------------------------
@router.get("/customer-segmentation/")
def get_customer_segments():
    try:
        return segment_customers()
    except Exception as e:
        return {"segments": [], "message": str(e)}


# -----------------------------------
# 📊 Demand Forecasting
# -----------------------------------
@router.get("/demand-forecast/")
def get_demand_forecast():
    try:
        return forecast_demand()
    except Exception as e:
        return {"forecast": [], "message": str(e)}



# -----------------------------------
# 🎯 Product Recommendation
# -----------------------------------
@router.get("/product-recommendations/")
def get_product_recommendations(user_id: int):
    try:
        return recommend_products(user_id)
    except Exception as e:
        return {"svd_recommendations": [], "nn_recommendations": [], "message": str(e)}


# -----------------------------------
# 🛒 Market Basket Analysis
# -----------------------------------
@router.get("/market-basket/")
def get_market_basket_analysis():
    try:
        return market_basket_analysis()
    except Exception as e:
        return {"rules": [], "message": str(e)}
    
# -----------------------------------
# 🛒 Menu Sales chart
# -----------------------------------
@router.get("/menu-category/")
def get_menu_categories():
    try:
        categorized = categorize_menu_items_hf()
        return {"categories": categorized}
    except Exception as e:
        return {"categories": {}, "error": str(e)}