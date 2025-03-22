import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from prophet import Prophet
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

def train_lstm(data):
    """Trains an LSTM model for revenue forecasting."""
    # Prepare Data
    data = data[["ds", "y"]].dropna()
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.set_index("ds").resample("D").sum().reset_index()
    
    values = data["y"].values
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
    """Generates LSTM revenue predictions."""
    inputs = torch.tensor(values[-10:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    outputs = []
    
    for _ in range(30):  # Predict next 30 days
        pred = model(inputs).item()
        outputs.append(pred)
        inputs = torch.cat([inputs[:, 1:, :], torch.tensor([[[pred]]])], dim=1)

    return outputs

def forecast_revenue():
    """Runs Prophet and LSTM to predict future revenue."""
    session = SessionLocal()
    sales_data = session.query(SalesData).all()
    session.close()

    df = pd.DataFrame([{"ds": s.date, "y": s.revenue} for s in sales_data])
    
    # Prophet Forecasting
    prophet_model = Prophet()
    if "ds" not in df.columns or "y" not in df.columns:
        print("Warning: Prophet DataFrame is missing required columns. Available columns:", df.columns)
    df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])

    future = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future)

    # Train LSTM
    lstm_model, values = train_lstm(df)
    lstm_forecast = predict_lstm(lstm_model, values)

    # Convert results to JSON
    return {
        "prophet_forecast": forecast[["ds", "yhat"]].tail(30).to_dict(orient="records"),
        "lstm_forecast": [{"ds": (datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d"), "yhat": lstm_forecast[i]} for i in range(30)]
    }
