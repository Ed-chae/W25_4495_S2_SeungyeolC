from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Database setup
DATABASE_URL = "postgresql://postgres:4495@localhost:5432/sales_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ---------------------------------------
# Existing SalesData Table (keep as-is)
# ---------------------------------------
class SalesData(Base):
    __tablename__ = "sales"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date)
    product = Column(String)
    revenue = Column(Float)
    review = Column(String)

# ---------------------------------------
# NEW: RestaurantOrder Table for restaurant-specific data
# ---------------------------------------
class RestaurantOrder(Base):
    __tablename__ = "restaurant_orders"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date)
    menu_item = Column(String)
    quantity = Column(Integer)
    price = Column(Float)
    review = Column(String)
    weather = Column(String)

# Create both tables if they don't already exist
Base.metadata.create_all(bind=engine)

# ---------------------------------------
# Save Functions
# ---------------------------------------

def save_sales_data(df: pd.DataFrame):
    """Saves processed sales data to PostgreSQL (original structure)."""
    session = SessionLocal()
    for _, row in df.iterrows():
        entry = SalesData(
            date=row["Date"],
            product=row["Product"],
            revenue=row["Revenue"],
            review=row["Review"]
        )
        session.add(entry)
    session.commit()
    session.close()

def save_restaurant_orders(df: pd.DataFrame):
    """Saves processed restaurant orders to PostgreSQL (new structure)."""
    session = SessionLocal()
    for _, row in df.iterrows():
        order = RestaurantOrder(
            date=row["Date"],
            menu_item=row["Menu"],
            quantity=int(row["Quantity"]),
            price=float(row["Price"]),
            review=row["Review"],
            weather=row.get("Weather", "")  # optional
        )
        session.add(order)
    session.commit()
    session.close()
