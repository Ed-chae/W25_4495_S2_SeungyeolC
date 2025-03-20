import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import torch.optim as optim
from db import SessionLocal, SalesData

# Define Autoencoder Model
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

def fetch_sales_data():
    """Fetch sales data from the database."""
    session = SessionLocal()
    sales_data = session.query(SalesData).all()
    session.close()

    df = pd.DataFrame([{"date": s.date, "revenue": s.revenue} for s in sales_data])
    df["date"] = pd.to_datetime(df["date"])
    return df

def detect_anomalies_isolation_forest(df):
    """Uses Isolation Forest to detect anomalies in sales data."""
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly_score"] = model.fit_predict(df[["revenue"]])
    df["is_anomaly"] = df["anomaly_score"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
    return df

def train_autoencoder(df):
    """Trains an Autoencoder for anomaly detection."""
    values = df["revenue"].values
    values = (values - values.min()) / (values.max() - values.min())  # Normalize
    values = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    model = Autoencoder(input_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        optimizer.zero_grad()
        output = model(values)
        loss = criterion(output, values)
        loss.backward()
        optimizer.step()

    return model

def detect_anomalies_autoencoder(df, model):
    """Uses trained Autoencoder to detect anomalies."""
    values = df["revenue"].values
    values = (values - values.min()) / (values.max() - values.min())  # Normalize
    values = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    reconstructed = model(values).detach().numpy()
    errors = np.abs(values.numpy() - reconstructed.flatten())

    df["autoencoder_score"] = errors
    threshold = np.percentile(errors, 95)
    df["is_anomaly_autoencoder"] = df["autoencoder_score"].apply(lambda x: "Anomaly" if x > threshold else "Normal")

    return df

def detect_sales_anomalies():
    """Runs both Isolation Forest & Autoencoder models to detect anomalies."""
    df = fetch_sales_data()

    # Isolation Forest
    df = detect_anomalies_isolation_forest(df)

    # Autoencoder
    autoencoder_model = train_autoencoder(df)
    df = detect_anomalies_autoencoder(df, autoencoder_model)

    return df.to_dict(orient="records")
