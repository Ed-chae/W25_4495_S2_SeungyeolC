import requests
import os
import pandas as pd
from db import SessionLocal, SalesData
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sqlalchemy import func
from sklearn.linear_model import LinearRegression
import numpy as np

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def fetch_weather(city):
    """Fetches current weather data for a given city."""
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "weather_code": data["weather"][0]["id"]
    }

def fetch_historical_weather(city, date):
    """Fetches past weather data for correlation analysis (mocked for now)."""
    # In a production setting, use a weather history API or store past data.
    return {
        "temperature": np.random.uniform(-5, 35),  # Simulating temperature range
        "humidity": np.random.uniform(20, 80),
        "weather": "Clear" if np.random.rand() > 0.5 else "Rainy"
    }

def correlate_weather_sales():
    """Finds correlation between past weather and sales revenue."""
    session = SessionLocal()
    sales_data = session.query(SalesData).all()
    session.close()

    df = pd.DataFrame([{
        "date": s.date, 
        "revenue": s.revenue, 
        "weather": fetch_historical_weather("Vancouver", s.date)
    } for s in sales_data])

    if "weather" not in df.columns:
        print("Warning: 'weather' column is missing. Available columns:", df.columns)
    df["weather"] = "Unknown"
    df["humidity"] = df["weather"].apply(lambda x: x["humidity"])
    df["weather_condition"] = df["weather"].apply(lambda x: x["weather"])

    # Convert categorical weather conditions to numeric
    df["weather_code"] = df["weather_condition"].apply(lambda x: 1 if x == "Clear" else 0)

    # Train a Linear Regression model
    X = df[["temperature", "humidity", "weather_code"]]
    y = df["revenue"]

    model = LinearRegression()
    model.fit(X, y)

    return model

def predict_revenue_impact(city):
    """Predicts sales impact based on future weather conditions."""
    model = correlate_weather_sales()
    weather_data = fetch_weather(city)

    if not weather_data:
        return {"error": "Unable to fetch weather data"}

    input_features = np.array([[weather_data["temperature"], weather_data["humidity"], 1 if weather_data["weather"] == "Clear" else 0]])
    predicted_revenue = model.predict(input_features)[0]

    return {
        "city": city,
        "temperature": weather_data["temperature"],
        "humidity": weather_data["humidity"],
        "weather": weather_data["weather"],
        "predicted_revenue": round(predicted_revenue, 2)
    }
