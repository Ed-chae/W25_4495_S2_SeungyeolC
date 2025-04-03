from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Database setup
DATABASE_URL = "postgresql://postgres:4495@localhost:5432/sales_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class SalesData(Base):
    __tablename__ = "sales"
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String)
    date = Column(Date)
    product = Column(String)
    quantity = Column(Integer)
    revenue = Column(Float)
    review = Column(String)
    weather = Column(String)

class RestaurantOrder(Base):
    __tablename__ = "restaurant_orders"
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String)
    date = Column(Date)
    menu_item = Column(String)
    quantity = Column(Integer)
    price = Column(Float)
    review = Column(String)
    weather = Column(String)

Base.metadata.create_all(bind=engine)

def save_sales_data(df: pd.DataFrame):
    session = SessionLocal()
    for _, row in df.iterrows():
        entry = SalesData(
            order_id=row.get("order_id"),
            date=row["date"],
            product=row["product"],
            quantity=int(row.get("quantity", 1)),
            revenue=row["revenue"],
            review=row.get("review"),
            weather=row.get("weather", "")
        )
        session.add(entry)
    session.commit()
    session.close()

def save_restaurant_orders(df: pd.DataFrame):
    session = SessionLocal()
    for _, row in df.iterrows():
        order = RestaurantOrder(
            order_id=row.get("order_id"),
            date=row["date"],
            menu_item=row["menu_item"],
            quantity=int(row["quantity"]),
            price=float(row["price"]),
            review=row.get("review"),
            weather=row.get("weather", "")
        )
        session.add(order)
    session.commit()
    session.close()
