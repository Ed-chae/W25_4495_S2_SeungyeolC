import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from prophet import Prophet

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for frontend communication

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Load Data from Uploaded File
def load_data(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["Date", "Total Price"])  # Return empty dataframe if file is missing
    
    df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.groupby("Date")["Total Price"].sum().reset_index()  # Aggregate revenue per day
    return df

# ✅ Prophet Model for Forecasting
def train_prophet(df):
    df.rename(columns={"Date": "ds", "Total Price": "y"}, inplace=True)
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]

# ✅ API Endpoint for Forecasting
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

# ✅ API Endpoint for File Upload
@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename("uploaded_data.xlsx")  # Save as a fixed file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
