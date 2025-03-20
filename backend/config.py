import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = "postgresql://postgres:4495@localhost:5432/sales_db"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "your_openweather_api_key")

# Security & API Settings
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
