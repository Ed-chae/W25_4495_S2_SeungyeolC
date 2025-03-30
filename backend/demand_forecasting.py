import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from db import SessionLocal, RestaurantOrder

# -----------------------------
# 🧠 LSTM Model for Demand Forecast
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
# 📈 Demand Forecast Function
# -----------------------------
def forecast_demand():
    """Forecasts item-level demand for the next 7 days using historical order quantity data."""
    session = SessionLocal()
    orders = session.query(RestaurantOrder).filter(RestaurantOrder.date != None).all()
    session.close()

    if not orders:
        return {
            "forecast": [],
            "message": "No restaurant order data found. Please upload data first."
        }

    df = pd.DataFrame([
        {"date": o.date, "menu_item": o.menu_item, "quantity": o.quantity}
        for o in orders
    ])

    df["date"] = pd.to_datetime(df["date"])

    forecast_summary = []

    for item, group in df.groupby("menu_item"):
        daily_sales = group.set_index("date").resample("D").sum().fillna(0)
        y = daily_sales["quantity"].values

        if len(y) < 10:
            continue

        x_train, y_train = [], []
        for i in range(len(y) - 10):
            x_train.append(y[i:i+10])
            y_train.append(y[i+10])

        x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

        model = LSTMModel(1, 32, 1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for _ in range(50):
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        recent = torch.tensor(y[-10:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        predictions = []
        for _ in range(7):
            next_val = model(recent).item()
            predictions.append(max(0, round(next_val)))
            recent = torch.cat([recent[:, 1:, :], torch.tensor([[[next_val]]])], dim=1)

        forecast_summary.append({
            "product": item,
            "forecast_next_7_days": int(sum(predictions))
        })

    return {
        "forecast": forecast_summary,
        "message": "Demand forecast generated successfully."
    }
