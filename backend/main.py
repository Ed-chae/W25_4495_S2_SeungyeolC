import io
from fastapi import FastAPI, APIRouter, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import asynccontextmanager
import pandas as pd
from textblob import TextBlob

# ✅ Database Configuration (PostgreSQL)
DATABASE_URL = "postgresql://postgres:4495@localhost/business_analytics"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ Define Order Table
class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date)
    time = Column(Time)
    order_id = Column(String)
    menu_item = Column(String)
    quantity = Column(Integer)
    total_price = Column(Float)
    payment_method = Column(String)
    customer_review = Column(String)
    weather_condition = Column(String)

# ✅ Create the Table if it Doesn't Exist
Base.metadata.create_all(bind=engine)

# ✅ Auto-delete data on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run the app as usual
    db = SessionLocal()
    db.execute("DELETE FROM orders;")  # Delete all records on shutdown
    db.commit()
    db.close()
    print("✅ All data removed on shutdown.")

# ✅ Initialize FastAPI
app = FastAPI(lifespan=lifespan)
router = APIRouter()

# ✅ CORS Middleware (Allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.1.76:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Process Uploaded Excel File
def process_excel(contents):
    try:
        df = pd.read_excel(io.BytesIO(contents))

        # Convert 'Time' to proper format
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.time

        # Handle missing values
        df["Customer Review"] = df["Customer Review"].fillna("")
        df["Customer Review"] = df["Customer Review"].astype(str).str.replace('"', '', regex=False)

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
            )
            db.add(order)

        db.commit()
        db.close()

        print("✅ Data successfully saved to the database.")
        return {"message": "File processed successfully", "columns": df.columns.tolist()}
    
    except Exception as e:
        print("❌ Error processing Excel:", str(e))
        return {"error": str(e)}

# ✅ Upload API Endpoint
@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return process_excel(contents)

# ✅ Get Analytics Data
@router.get("/analytics/")
def get_analytics():
    db = SessionLocal()
    data = db.query(Order).all()
    db.close()
    
    if not data:
        return {"error": "No data available"}
    
    df = pd.DataFrame([{column.name: getattr(d, column.name) for column in Order.__table__.columns} for d in data])

    # Convert date to string format
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)

    # Replace NaN and infinite values
    df = df.fillna(0)
    df.replace([float("inf"), float("-inf")], 0, inplace=True)

    revenue_trends = df.groupby("date")["total_price"].sum().reset_index()
    best_sellers = df.groupby("menu_item")["quantity"].sum().reset_index().sort_values(by="quantity", ascending=False)

    sentiment_scores = df.dropna(subset=["customer_review"])
    sentiment_scores["sentiment"] = sentiment_scores["customer_review"].apply(
        lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0
    )
    
    avg_sentiment = sentiment_scores.groupby("menu_item")["sentiment"].mean().reset_index()
    weather_impact = df.groupby("weather_condition")["total_price"].sum().reset_index()

    return {
        "revenue_trends": revenue_trends.to_dict(orient="records"),
        "best_sellers": best_sellers.to_dict(orient="records"),
        "sentiment_analysis": avg_sentiment.to_dict(orient="records"),
        "weather_impact": weather_impact.to_dict(orient="records"),
    }

# ✅ Register API Routes
app.include_router(router)

# ✅ Root Endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Intelligent Business Analytics System"}
