from fastapi import FastAPI, APIRouter, File, UploadFile, Depends
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import io
from datetime import datetime, timedelta
from textblob import TextBlob
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# ✅ PostgreSQL Database Configuration
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

# ✅ FastAPI App Setup with Auto-Cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Cleans up database when app shuts down. """
    yield
    db = SessionLocal()
    db.query(Order).delete()
    db.commit()
    db.close()

app = FastAPI(lifespan=lifespan)

# ✅ CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.1.76:3000"],  # Allow frontend access
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

router = APIRouter()

# ✅ Process Excel File
def process_excel(contents):
    df = pd.read_excel(io.BytesIO(contents))
    if df.empty:
        return {"error": "Uploaded file is empty"}
    if len(df.columns) < 2:
        return {"error": "File must have at least two columns"}
    return df

# ✅ Upload File Endpoint
@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = process_excel(contents)

        if isinstance(df, dict) and "error" in df:
            return df  # Return error response if processing failed

        # ✅ Convert Date & Time with Specified Format
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce").dt.date
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.time

        # ✅ Sentiment Analysis
        df["Sentiment Score"] = df["Customer Review"].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if x else 0
        )

        # ✅ Store Data in Database
        db = SessionLocal()
        for _, row in df.iterrows():
            order = Order(
                date=row["Date"],
                time=row["Time"],
                order_id=row["Order ID"],
                menu_item=row["Menu Item"],
                quantity=row["Quantity"],
                total_price=row["Total Price"],
                payment_method=row["Payment Method"],
                customer_review=row["Customer Review"],
                weather_condition=row["Weather Condition"],
                sentiment_score=row["Sentiment Score"],
            )
            db.add(order)
        db.commit()
        db.close()

        return {"message": "File processed and data stored successfully."}
    except Exception as e:
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
    df["date"] = df["date"].astype(str)

    # ✅ Revenue Trends
    revenue_trends = df.groupby("date")["total_price"].sum().reset_index()

    # ✅ Best-Selling Menu Items
    best_sellers = df.groupby("menu_item")["quantity"].sum().reset_index().sort_values(by="quantity", ascending=False)

    # ✅ Sentiment Analysis
    df["sentiment"] = df["customer_review"].apply(lambda x: TextBlob(str(x)).sentiment.polarity if x else 0)
    sentiment_scores = df.groupby("menu_item")["sentiment"].mean().reset_index()

    # ✅ Best & Worst Menu Items
    best_menu = sentiment_scores.loc[sentiment_scores["sentiment"].idxmax(), "menu_item"]
    worst_menu = sentiment_scores.loc[sentiment_scores["sentiment"].idxmin(), "menu_item"]

    # ✅ Weather Impact on Sales
    weather_impact = df.groupby("weather_condition")["total_price"].sum().reset_index()

    # ✅ Predict Revenue for Next 7 Days (Corrected Formula)
    daily_revenue = df.groupby("date")["total_price"].sum().mean()

    future_revenue = []
    last_date = pd.to_datetime(df["date"]).max()

    for i in range(1, 8):
        next_date = last_date + timedelta(days=i)
        for weather in weather_impact["weather_condition"].unique():
            weather_factor = (
                weather_impact[weather_impact["weather_condition"] == weather]["total_price"].sum()
                / weather_impact["total_price"].sum()
                if not weather_impact["total_price"].sum() == 0
                else 1
            )
            predicted_revenue = daily_revenue * weather_factor
            future_revenue.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "weather_condition": weather,
                "predicted_revenue": round(predicted_revenue, 2),
            })

    return {
        "revenue_trends": revenue_trends.to_dict(orient="records"),
        "best_sellers": best_sellers.to_dict(orient="records"),
        "sentiment_analysis": sentiment_scores.to_dict(orient="records"),
        "weather_impact": weather_impact.to_dict(orient="records"),
        "future_revenue": future_revenue,
        "best_menu": best_menu,
        "worst_menu": worst_menu,
    }

# ✅ Root Endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Intelligent Business Analytics System"}

# ✅ Include Router
app.include_router(router)
