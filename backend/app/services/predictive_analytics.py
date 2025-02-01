import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

from app.services.data_ingestion import load_cleaned_data

def analyze_revenue_trends():
    df = load_cleaned_data()
    if df is None:
        return {"status": "error", "message": "No data available"}

    # Aggregate revenue by month
    df["YearMonth"] = df["OrderDate"].dt.to_period("M")
    revenue_trend = df.groupby("YearMonth")["Revenue"].sum().reset_index()
    revenue_trend["YearMonth"] = revenue_trend["YearMonth"].astype(str)

    return revenue_trend.to_dict(orient="records")


def predict_future_revenue():
    df = load_cleaned_data()
    if df is None:
        return {"status": "error", "message": "No data available"}

    # Prepare data for forecasting
    df["YearMonth"] = df["OrderDate"].dt.to_period("M").astype(str)
    revenue_trend = df.groupby("YearMonth")["Revenue"].sum().reset_index()

    # Convert time periods to numeric values for regression
    revenue_trend["MonthIndex"] = np.arange(len(revenue_trend))

    # Train a simple linear regression model
    X = revenue_trend[["MonthIndex"]]
    y = revenue_trend["Revenue"]

    model = LinearRegression()
    model.fit(X, y)

    # Predict next 3 months
    future_months = np.array([[len(revenue_trend)], [len(revenue_trend) + 1], [len(revenue_trend) + 2]])
    predictions = model.predict(future_months)

    future_dates = [
        str(pd.Period(revenue_trend["YearMonth"].iloc[-1]) + i)
        for i in range(1, 4)
    ]

    return {"future_predictions": dict(zip(future_dates, predictions.round(2)))}
