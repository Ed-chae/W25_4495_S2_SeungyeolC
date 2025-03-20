from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pathlib import Path
from db import SessionLocal, SalesData
from file_processing import process_sales_data
from sentiment_analysis import analyze_sentiment
from revenue_forecasting import forecast_revenue
from weather_analysis import predict_revenue_impact
from customer_segmentation import segment_customers
from demand_forecasting import forecast_demand
from anomaly_detection import detect_sales_anomalies
from product_recommendation import recommend_products
from market_basket_analysis import market_basket_analysis

app = FastAPI(title="Intelligent Business Analytics API")

# CORS settings to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change "*" to specific URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # Ensure upload folder exists

def get_db():
    """Dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "Welcome to the Intelligent Business Analytics API!"}

# -----------------------------------
# 📂 File Upload API
# -----------------------------------
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handles file upload, processes data, and stores results."""
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
# 📊 Sentiment Analysis API
# -----------------------------------
@app.get("/sentiment-results/")
def get_sentiment_results(db: Session = Depends(get_db)):
    """Fetch sentiment analysis results."""
    results = db.query(SalesData).all()
    return [{"product": r.product, "review": r.review, "sentiment": analyze_sentiment(r.review)["label"]} for r in results]

# -----------------------------------
# 📈 Revenue Forecasting API
# -----------------------------------
@app.get("/revenue-forecast/")
def get_revenue_forecast():
    """Fetches AI-powered revenue predictions (Prophet & LSTM)."""
    return forecast_revenue()

# -----------------------------------
# ⛅ Weather Impact on Sales API
# -----------------------------------
@app.get("/weather-impact/")
def get_weather_impact(city: str = "Vancouver"):
    """Fetches weather conditions & predicts sales impact."""
    return predict_revenue_impact(city)

# -----------------------------------
# 🛍️ Customer Segmentation API
# -----------------------------------
@app.get("/customer-segmentation/")
def get_customer_segments():
    """Fetches AI-driven customer segments."""
    return segment_customers()

# -----------------------------------
# 📊 Demand Forecasting API
# -----------------------------------
@app.get("/demand-forecast/")
def get_demand_forecast():
    """Fetches AI-powered demand predictions (LSTM & XGBoost)."""
    return forecast_demand()

# -----------------------------------
# 🚨 Anomaly Detection API
# -----------------------------------
@app.get("/sales-anomalies/")
def get_sales_anomalies():
    """Fetches detected anomalies in sales trends."""
    return detect_sales_anomalies()

# -----------------------------------
# 🎯 Product Recommendation API
# -----------------------------------
@app.get("/product-recommendations/")
def get_product_recommendations(user_id: int):
    """Fetches AI-based product recommendations for a given user."""
    return recommend_products(user_id)

# -----------------------------------
# 🛒 Market Basket Analysis API
# -----------------------------------
@app.get("/market-basket/")
def get_market_basket_analysis():
    """Fetches Market Basket Analysis results (frequent itemsets & association rules)."""
    return market_basket_analysis()
