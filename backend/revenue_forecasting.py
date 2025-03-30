import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from prophet import Prophet
from db import SessionLocal, RestaurantOrder
from datetime import datetime, timedelta

# -----------------------------
# 📦 LSTM Model Definition
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# -----------------------------
# 🧠 Train LSTM Model
# -----------------------------
def train_lstm(data):
    data = data[["ds", "y"]].dropna()
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.set_index("ds").resample("D").sum().reset_index()

    values = data["y"].values
    if len(values) <= 10:
        raise ValueError("Not enough data to train LSTM.")

    values = (values - values.min()) / (values.max() - values.min())

    x_train, y_train = [], []
    for i in range(len(values) - 10):
        x_train.append(values[i:i + 10])
        y_train.append(values[i + 10])

    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

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

# -----------------------------
# 🔮 Predict with LSTM
# -----------------------------
def predict_lstm(model, values):
    inputs = torch.tensor(values[-10:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    outputs = []
    for _ in range(30):
        pred = model(inputs).item()
        outputs.append(pred)
        inputs = torch.cat([inputs[:, 1:, :], torch.tensor([[[pred]]])], dim=1)
    return outputs

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
            "lstm_forecast": [],
            "message": "No data available. Please upload a file first."
        }

    df = pd.DataFrame([{
        "ds": o.date,
        "y": o.quantity * o.price
    } for o in orders])

    if df.empty or "ds" not in df.columns or "y" not in df.columns:
        return {
            "prophet_forecast": [],
            "lstm_forecast": [],
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

    # LSTM Forecast
    lstm_model, values = train_lstm(df.copy())
    lstm_output = predict_lstm(lstm_model, values)
    today = datetime.today()

    lstm_forecast = [
        {"ds": (today + timedelta(days=i)).strftime("%Y-%m-%d"), "yhat": float(lstm_output[i])}
        for i in range(30)
    ]

    return {
        "prophet_forecast": prophet_output,
        "lstm_forecast": lstm_forecast
    }
