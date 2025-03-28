import requests
import os
import pandas as pd
from db import SessionLocal, RestaurantOrder
from dotenv import load_dotenv
from datetime import datetime, timedelta
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
        "weather": data["weather"][0]["main"]
    }


def fetch_historical_weather(city, date):
    """Fetches past weather data for correlation analysis (mocked for now)."""
    return {
        "temperature": np.random.uniform(-5, 35),
        "humidity": np.random.uniform(20, 80),
        "weather": "Clear" if np.random.rand() > 0.5 else "Rainy"
    }


def correlate_weather_sales():
    """Finds correlation between past weather and restaurant order revenue."""
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

    # Check for missing columns before training
    required_cols = ["temperature", "humidity", "weather_code", "revenue"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing one of required columns: {required_cols}")

    X = df[["temperature", "humidity", "weather_code"]]
    y = df["revenue"]

    model = LinearRegression()
    model.fit(X, y)

    return model


def predict_revenue_impact(city):
    """Predicts revenue impact based on weather forecast for a city."""
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
