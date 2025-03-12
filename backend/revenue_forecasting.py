import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load data from uploaded file
def load_data(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["Date", "Total Price"])  # Return empty dataframe if file is missing
    
    df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")
    df["Date"] = pd.to_datetime(df["Date"])  # Convert Date column to datetime
    df = df.groupby("Date")["Total Price"].sum().reset_index()  # Aggregate revenue per day
    return df

# LSTM Model for Revenue Forecasting
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Prophet Model for Forecasting
def train_prophet(df):
    df.rename(columns={"Date": "ds", "Total Price": "y"}, inplace=True)
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]

# Flask API Endpoint for Forecasting
@app.route("/forecast", methods=["GET"])
def forecast():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_data.xlsx")  
    if not os.path.exists(file_path):
        return jsonify({"error": "Uploaded file not found"}), 400

    df = load_data(file_path)
    if df.empty:
        return jsonify({"error": "Uploaded file is empty or invalid"}), 400

    prophet_forecast = train_prophet(df)
    return jsonify({"prophet_forecast": prophet_forecast.to_dict(orient="records")})


# Flask API Endpoint for Uploading Data
@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename("uploaded_data.xlsx")  # Always save as the same file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200

if __name__ == "__main__":
    app.run(debug=True)
