from fastapi import FastAPI, APIRouter, File, UploadFile
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import io
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import logging
import requests

# ✅ Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Database Configuration
DATABASE_URL = "postgresql://postgres:4495@localhost/business_analytics"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ Define Order Model
class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    time = Column(Time)
    order_id = Column(String, index=True)
    menu_item = Column(String)
    quantity = Column(Integer)
    total_price = Column(Float)
    payment_method = Column(String)
    customer_review = Column(String)
    weather_condition = Column(String)
    sentiment_score = Column(Float)

# ✅ Create Tables
Base.metadata.create_all(bind=engine)

# ✅ FastAPI App
app = FastAPI()

# ✅ Enable CORS for Frontend Communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ❗ Allow all origins (For Development Only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Start Flask Forecasting API Automatically
subprocess.Popen(["python", "revenue_forecasting.py"])

router = APIRouter()

# ✅ Upload File Endpoint (Redirects to Flask API)
@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        response = requests.post("http://127.0.0.1:5050/upload", files=files)
        return response.json()
    except Exception as e:
        logger.error(f"Upload Error: {str(e)}")
        return {"error": str(e)}

# ✅ Analytics Endpoint
@router.get("/analytics/")
def get_analytics():
    db = SessionLocal()
    data = db.query(Order).all()
    db.close()

    if not data:
        return {"error": "No data available"}

    df = pd.DataFrame([{column.name: getattr(d, column.name) for column in Order.__table__.columns} for d in data])
    df["date"] = pd.to_datetime(df["date"])

    # ✅ Revenue Trends
    revenue_trends = df.groupby("date")["total_price"].sum().reset_index()

    # ✅ Best-Selling Menu Items
    best_sellers = df.groupby("menu_item")["quantity"].sum().reset_index().sort_values(by="quantity", ascending=False)

    return {
        "salesData": {
            "labels": revenue_trends["date"].dt.strftime("%Y-%m-%d").tolist(),
            "datasets": [
                {"label": "Revenue", "data": revenue_trends["total_price"].tolist(), "backgroundColor": "rgba(75,192,192,0.6)"}
            ],
        },
        "customerData": {
            "labels": best_sellers["menu_item"].tolist(),
            "datasets": [
                {"data": best_sellers["quantity"].tolist(), "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56"]}
            ],
        },
    }

# ✅ Root Endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Intelligent Business Analytics System"}

# ✅ Include Router
app.include_router(router)

# ✅ Start FastAPI Backend
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
