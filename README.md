#  Intelligent Business Analytics System 

##  Project Overview
The **Intelligent Business Analytics System** is a powerful AI-driven platform designed to analyze sales data, predict revenue, detect anomalies, and provide insights into customer sentiment, weather impact, and product recommendations. The system integrates **FastAPI (backend)** and **React (frontend)** with **AI-powered analytics** using **Prophet, LSTM, K-Means, and sentiment analysis models**.

---

##  **Key Features**
### **1️ File Upload & Data Processing**  
-  Users can upload **Excel sales data** (`.xlsx`).  
-  The backend processes data and stores it in **PostgreSQL**.

### ** AI-Powered Sentiment Analysis**  
-  Classifies **customer reviews** as **positive or negative** using **DistilBERT**.  
-  **Visualized in charts** in the frontend.

### ** AI-Based Revenue Forecasting**  
-  **Forecasts future revenue trends** using **Prophet (Time Series)** and **LSTM (Deep Learning)**.  
-  Displays **predicted sales trends** in React.

### ** Weather Impact Analysis on Revenue**  
-  Analyzes the effect of **weather conditions** on sales.  
-  Uses **OpenWeatherMap API** + **Linear Regression** to predict revenue impact.

### ** AI-Powered Customer Segmentation**  
-  Clusters customers based on **purchase behavior** using **K-Means & DBSCAN**.  
-  Visualized in React for insights.

### ** AI-Based Demand Forecasting**  
-  Predicts **future sales demand trends** using **LSTM & XGBoost**.  
-  Helps optimize **inventory management**.

### ** AI-Powered Anomaly Detection**  
-  Detects unusual sales spikes & fraudulent transactions using **Isolation Forest & Autoencoders**.  
-  Highlights anomalies in sales data.

### ** AI-Based Product Recommendations**  
-  Suggests **personalized products** using **Collaborative Filtering & Deep Learning**.  
-  **Recommender System** powered by **SVD & Neural Networks**.

### ** Market Basket Analysis**  
-  Identifies frequently purchased **product combinations** using **Apriori Algorithm**.  
-  Helps with **cross-selling strategies**.

---

##  **Tech Stack**
### **Backend (FastAPI + PostgreSQL + AI Models)**
- **FastAPI** (for API endpoints)
- **PostgreSQL** (for storing sales data)
- **Pandas, NumPy** (for data processing)
- **Prophet, LSTM** (for revenue forecasting)
- **DistilBERT** (for sentiment analysis)
- **K-Means, DBSCAN** (for customer segmentation)
- **OpenWeatherMap API** (for weather impact analysis)
- **Isolation Forest, Autoencoders** (for anomaly detection)
- **Scikit-Learn, XGBoost** (for demand forecasting)
- **Apriori Algorithm** (for market basket analysis)

### **Frontend (React + Chart.js + Axios)**
- **React** (for UI components)
- **Axios** (for API calls)
- **Chart.js + react-chartjs-2** (for data visualization)

---

##  **Project Structure**
```
Intelligent_Business_Analytics_System/ │── backend/ # FastAPI Backend │ ├── api.py # Main API routes │ ├── db.py # Database connection │ ├── file_processing.py # Handles file uploads & data processing │ ├── sentiment_analysis.py # AI-powered sentiment analysis │ ├── revenue_forecasting.py # AI-based revenue forecasting │ ├── weather_analysis.py # Weather impact on revenue │ ├── customer_segmentation.py # AI-powered customer segmentation │ ├── anomaly_detection.py # Detects sales anomalies │ ├── product_recommendation.py # AI-based product recommendations │ ├── demand_forecasting.py # Predicts demand using AI │ ├── market_basket_analysis.py # Market Basket Analysis │ ├── upload.py # File upload API │ ├── config.py # Configuration settings │ ├── requirements.txt # Python dependencies │── frontend/ # React Frontend │ ├── src/ │ │ ├── components/ │ │ │ ├── FileUpload.js # Upload Excel files │ │ │ ├── SentimentChart.js # Sentiment Analysis Visualization │ │ │ ├── RevenueChart.js # Revenue Forecasting Chart │ │ │ ├── WeatherImpact.js # Weather impact on sales │ │ │ ├── CustomerSegments.js # Customer segmentation visualization │ │ │ ├── DemandForecast.js # Demand forecasting visualization │ │ │ ├── SalesAnomalies.js # Anomaly detection visualization │ │ │ ├── Recommendations.js # Product recommendations visualization │ │ │ ├── MarketBasket.js # Market Basket Analysis visualization │ │ ├── services/ │ │ │ ├── api.js # Axios API service │ │ ├── App.js # Main React App │ │ ├── index.js # React Index file │ │ ├── styles.css # CSS Styles │── tests/ # Unit Tests │ ├── test_file_processing.py │ ├── test_revenue_forecasting.py │ ├── test_sentiment_analysis.py │── DocumentsAndReports/ # Documentation │── README.md # Project Documentation
```

---

##  **Installation & Setup**
### ** Clone the Repository**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/Intelligent_Business_Analytics_System.git
cd Intelligent_Business_Analytics_System
```

### ** Set Up the Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```
API runs at: **http://127.0.0.1:8000**

### ** Set Up the Frontend**
```bash
cd frontend
npm install react react-dom axios framer-motion chart.js react-chartjs-2 html2pdf.js
npm start
```
Frontend runs at: **http://localhost:3000**

---

##  **Contributors**
- **Seungyeol Chae** - [GitHub](https://github.com/Ed-chae)

---

##  **Final Thoughts**
This **AI-powered Business Analytics System** helps businesses **understand sales trends, predict revenue, optimize inventory, and personalize customer experiences**. 

>  ** Contact me at chaes@student.douglascollege.ca**  
