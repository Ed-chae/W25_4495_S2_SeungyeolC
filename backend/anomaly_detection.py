import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import torch.optim as optim
from db import SessionLocal, RestaurantOrder

# -----------------------------
# 🧠 Autoencoder Model
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# -----------------------------
# 📥 Fetch Data from DB
# -----------------------------
def fetch_order_data():
    session = SessionLocal()
    data = session.query(RestaurantOrder).filter(RestaurantOrder.date != None).all()
    session.close()

    df = pd.DataFrame([{
        "date": o.date,
        "menu_item": o.menu_item,
        "quantity": o.quantity
    } for o in data])

    df["date"] = pd.to_datetime(df["date"])
    return df


# -----------------------------
# 🌲 Isolation Forest
# -----------------------------
def detect_anomalies_isolation_forest(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly_score"] = model.fit_predict(df[["quantity"]])
    df["is_anomaly"] = df["anomaly_score"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
    return df


# -----------------------------
# 🤖 Autoencoder Detection
# -----------------------------
def train_autoencoder(df):
    values = df["quantity"].values
    values = (values - values.min()) / (values.max() - values.min())  # Normalize
    values = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    model = Autoencoder(input_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(200):
        optimizer.zero_grad()
        output = model(values)
        loss = criterion(output, values)
        loss.backward()
        optimizer.step()

    return model


def detect_anomalies_autoencoder(df, model):
    values = df["quantity"].values
    values = (values - values.min()) / (values.max() - values.min())  # Normalize
    values = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    reconstructed = model(values).detach().numpy()
    errors = np.abs(values.numpy() - reconstructed.flatten())

    df["autoencoder_score"] = errors
    threshold = np.percentile(errors, 95)
    df["is_anomaly_autoencoder"] = df["autoencoder_score"].apply(lambda x: "Anomaly" if x > threshold else "Normal")

    return df


# -----------------------------
# 🚨 Main Anomaly Function
# -----------------------------
def detect_sales_anomalies():
    df = fetch_order_data()

    if df.empty or "quantity" not in df.columns:
        raise ValueError("No valid quantity data found.")

    df = detect_anomalies_isolation_forest(df)
    autoencoder_model = train_autoencoder(df)
    df = detect_anomalies_autoencoder(df, autoencoder_model)

    return df.to_dict(orient="records")
