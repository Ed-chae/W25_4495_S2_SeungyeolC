import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:yourpassword@localhost/business_analytics")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://192.168.1.76:3000").split(",")
