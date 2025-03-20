from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import app as api_routes

app = FastAPI(title="Intelligent Business Analytics API")

# CORS settings to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change "*" to specific URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.mount("/", api_routes)

@app.get("/")
def root():
    return {"message": "Welcome to the Intelligent Business Analytics System!"}
