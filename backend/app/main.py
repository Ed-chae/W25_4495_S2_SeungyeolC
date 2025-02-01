from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import data, analytics

app = FastAPI()

# ✅ Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to a specific frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.include_router(data.router, prefix="/api/data", tags=["Data Management"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])

@app.get("/")
def home():
    return {"message": "Welcome to the Intelligent Business Analytics System"}
