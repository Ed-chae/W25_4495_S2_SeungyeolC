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
    """Mocks historical weather for correlation training."""
    return {
        "temperature": np.random.uniform(-5, 35),
        "humidity": np.random.uniform(20, 80),
        "weather": "Clear" if np.random.rand() > 0.5 else "Rainy"
    }


def correlate_weather_sales():
    """Train a model to find relationship between weather and restaurant revenue."""
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

    X = df[["temperature", "humidity", "weather_code"]]
    y = df["revenue"]

    model = LinearRegression()
    model.fit(X, y)

    return model


def predict_revenue_impact(city):
    """Predicts revenue for the next 7 days using weather forecasts."""
    model = correlate_weather_sales()
    future_predictions = []

    today = datetime.today()
    for i in range(7):
        future_date = today + timedelta(days=i)
        weather = fetch_historical_weather(city, future_date)  # Replace with real 7-day API if available

        temp = weather["temperature"]
        humid = weather["humidity"]
        code = 1 if weather["weather"] == "Clear" else 0

        X = np.array([[temp, humid, code]])
        predicted_revenue = model.predict(X)[0]

        future_predictions.append({
            "date": future_date.strftime("%Y-%m-%d"),
            "temperature": round(temp, 1),
            "humidity": round(humid, 1),
            "weather": weather["weather"],
            "predicted_revenue": round(predicted_revenue, 2)
        })

    return {
        "city": city,
        "forecast": future_predictions
    }
