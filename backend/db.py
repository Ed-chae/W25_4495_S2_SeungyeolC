from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

# ✅ Setup PostgreSQL Database Connection
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

# ✅ Create Table in the Database
Base.metadata.create_all(bind=engine)
