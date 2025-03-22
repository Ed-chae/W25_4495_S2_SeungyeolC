import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from db import SessionLocal, SalesData
from datetime import datetime, timedelta

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def fetch_sales_data():
    """Fetches historical sales data from the database."""
    session = SessionLocal()
    sales_data = session.query(SalesData).all()
    session.close()

    df = pd.DataFrame([{"ds": s.date, "y": s.revenue} for s in sales_data])
    if "ds" not in df.columns:
        df.rename(columns={df.columns[0]: "ds"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])
    return df

def train_lstm_model(df):
    """Trains an LSTM model for demand forecasting."""
    df = df.set_index("ds").resample("D").sum().reset_index()
    values = df["y"].values
    values = (values - values.min()) / (values.max() - values.min())  # Normalize

    x_train, y_train = [], []
    for i in range(len(values) - 10):
        x_train.append(values[i : i + 10])
        y_train.append(values[i + 10])

    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    # Train Model
    model = LSTMModel(input_size=1, hidden_size=50, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    return model, values

def predict_lstm(model, values):
    """Generates demand predictions using LSTM."""
    inputs = torch.tensor(values[-10:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    outputs = []
    
    for _ in range(30):  # Predict next 30 days
        pred = model(inputs).item()
        outputs.append(pred)
        inputs = torch.cat([inputs[:, 1:, :], torch.tensor([[[pred]]])], dim=1)

    return outputs

def train_xgboost_model(df):
    """Trains an XGBoost model for demand forecasting."""
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month

    X = df[["day_of_week", "month"]]
    y = df["y"]

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y)

    return model

def predict_xgboost(model):
    """Generates demand predictions using XGBoost."""
    future_dates = pd.date_range(start=datetime.today(), periods=30, freq="D")
    future_data = pd.DataFrame({"ds": future_dates})
    future_data["day_of_week"] = future_data["ds"].dt.dayofweek
    future_data["month"] = future_data["ds"].dt.month

    predictions = model.predict(future_data[["day_of_week", "month"]])
    return [{"ds": future_dates[i].strftime("%Y-%m-%d"), "yhat": predictions[i]} for i in range(30)]

def forecast_demand():
    """Runs both LSTM and XGBoost models to predict demand."""
    df = fetch_sales_data()

    # Train LSTM
    lstm_model, values = train_lstm_model(df)
    lstm_forecast = predict_lstm(lstm_model, values)

    # Train XGBoost
    xgb_model = train_xgboost_model(df)
    xgb_forecast = predict_xgboost(xgb_model)

    return {
        "lstm_forecast": [{"ds": (datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d"), "yhat": lstm_forecast[i]} for i in range(30)],
        "xgboost_forecast": xgb_forecast
    }
