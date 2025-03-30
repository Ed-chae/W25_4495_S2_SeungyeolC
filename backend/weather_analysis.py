import requests
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from sklearn.linear_model import LinearRegression
from db import SessionLocal, RestaurantOrder

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"


# ---------------------------------------------
# 🌤️ Live Weather Fetch (for Forecast)
# ---------------------------------------------
def fetch_weather(city):
    """Fetches current weather data for a given city using OpenWeather API."""
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "main" not in data or "weather" not in data:
        raise ValueError("❌ Failed to retrieve weather data.")

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "weather": data["weather"][0]["main"]
    }


# ---------------------------------------------
# 🕓 Historical Weather (Mocked)
# ---------------------------------------------
def fetch_historical_weather(city, date):
    """Mocked historical weather for correlation with revenue."""
    return {
        "temperature": np.random.uniform(-5, 35),
        "humidity": np.random.uniform(20, 80),
        "weather": "Clear" if np.random.rand() > 0.5 else "Rainy"
    }


# ---------------------------------------------
# 📈 Weather vs Revenue Correlation
# ---------------------------------------------
def correlate_weather_sales():
    """Correlates past weather (mocked) with restaurant revenue."""
    session = SessionLocal()
    orders = session.query(RestaurantOrder).filter(RestaurantOrder.date != None).all()
    session.close()

    if not orders:
        raise ValueError("No restaurant order data found.")

    data = []
    for o in orders:
        weather = fetch_historical_weather("Vancouver", o.date)
        data.append({
            "date": o.date,
            "revenue": o.quantity * o.price,
            "temperature": weather["temperature"],
            "humidity": weather["humidity"],
            "weather_condition": weather["weather"]
        })

    df = pd.DataFrame(data)
    df["weather_code"] = df["weather_condition"].apply(lambda x: 1 if x == "Clear" else 0)

    required_cols = ["temperature", "humidity", "weather_code", "revenue"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Missing required columns for model training.")

    X = df[["temperature", "humidity", "weather_code"]]
    y = df["revenue"]

    model = LinearRegression()
    model.fit(X, y)

    return model


# ---------------------------------------------
# 🔮 Predict Revenue Based on Weather
# ---------------------------------------------
def predict_revenue_impact(city):
    """Predicts expected restaurant revenue based on current weather."""
    model = correlate_weather_sales()
    weather_data = fetch_weather(city)

    input_features = np.array([[
        weather_data["temperature"],
        weather_data["humidity"],
        1 if weather_data["weather"] == "Clear" else 0
    ]])

    predicted_revenue = model.predict(input_features)[0]

    return {
        "city": city,
        "temperature": weather_data["temperature"],
        "humidity": weather_data["humidity"],
        "weather": weather_data["weather"],
        "predicted_revenue": round(predicted_revenue, 2)
    }
