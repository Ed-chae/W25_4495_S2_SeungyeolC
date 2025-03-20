from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os

DATABASE_URL = "postgresql://postgres:4495@localhost:5432/sales_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Define sales data table
class SalesData(Base):
    __tablename__ = "sales"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date)
    product = Column(String)
    revenue = Column(Float)
    review = Column(String)

Base.metadata.create_all(bind=engine)  # Create table if not exists

def save_sales_data(df: pd.DataFrame):
    """Saves processed sales data to PostgreSQL."""
    session = SessionLocal()

    for _, row in df.iterrows():
        sales_entry = SalesData(
            date=row["Date"],
            product=row["Product"],
            revenue=row["Revenue"],
            review=row["Review"]
        )
        session.add(sales_entry)

    session.commit()
    session.close()
