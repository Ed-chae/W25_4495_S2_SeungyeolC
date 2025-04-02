import pandas as pd
from prophet import Prophet
from db import SessionLocal, RestaurantOrder
from datetime import datetime, timedelta

# -----------------------------
# 📈 Forecast Restaurant Revenue
# -----------------------------
def forecast_revenue():
    session = SessionLocal()
    orders = session.query(RestaurantOrder).filter(RestaurantOrder.date != None).all()
    session.close()

    if not orders:
        return {
            "prophet_forecast": [],
            "message": "No data available. Please upload a file first."
        }

    df = pd.DataFrame([{
        "ds": o.date,
        "y": o.quantity * o.price
    } for o in orders])

    if df.empty or "ds" not in df.columns or "y" not in df.columns:
        return {
            "prophet_forecast": [],
            "message": "Data format error: missing 'ds' or 'y' columns."
        }

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    # Prophet Forecast
    prophet_model = Prophet()
    prophet_model.fit(df)
    future = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future)
    prophet_output = forecast[["ds", "yhat"]].tail(30).to_dict(orient="records")

    return {
        "prophet_forecast": prophet_output,
        "message": "Prophet forecast generated."
    }
