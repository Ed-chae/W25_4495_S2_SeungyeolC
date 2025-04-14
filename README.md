# Intelligent Business Analytics System

## 📊 Project Overview
The **Intelligent Business Analytics System** is a comprehensive AI-powered platform designed for analyzing restaurant sales data. Built with **FastAPI** and **React**, it helps restaurant owners understand customer sentiment, predict revenue, analyze weather impacts, forecast demand, recommend products, and identify frequently paired menu items.

---

## 🚀 Key Features

### ✅ File Upload & Data Processing
- Upload Excel files (`.xlsx`) with restaurant order data.
- Backend processes and stores the data in a PostgreSQL database.

### 💬 Sentiment Analysis
- Analyzes customer reviews using **DistilBERT**.
- Shows best/worst items and sentiment breakdown per menu item.

### 📈 Revenue Forecasting
- Predicts next 30 days' revenue using **Prophet**.
- Results displayed as line charts.

### ⛅ Weather Impact Forecasting
- Uses **OpenWeatherMap API** to fetch forecast.
- Applies **Linear Regression** to estimate revenue based on weather.

### 👥 Customer Segmentation
- Clusters customers using **KMeans** based on purchase behavior.
- Displays segment behavior and size.

### 📦 Demand Forecasting
- Forecasts demand (number of units) for each item for the next 7 days.

### 🛍️ Product Recommendation
- Generates personalized recommendations using:
  - **Collaborative Filtering (SVD)**
  - **Neural Network Model**
- Removes duplicates and already purchased items from recommendations.

### 🛒 Market Basket Analysis
- Uses **Apriori Algorithm** to find frequently bought item combinations.
- Ranks association rules by Support, Confidence, and Lift.

### 🍽️ Menu Category Breakdown
- Automatically categorizes menu items (e.g., Mains, Drinks, Desserts) using a **local NLP model** (HuggingFace Transformers).
- Displays pie charts showing how many items are sold in each category.

---

## 🧠 Tech Stack

### Backend (Python - FastAPI)
- **FastAPI** for RESTful APIs
- **PostgreSQL** for data storage
- **Pandas, NumPy** for processing
- **Prophet** for revenue forecasting
- **Scikit-learn (KMeans, SVD, LinearRegression)**
- **Transformers (DistilBERT)** for sentiment
- **mlxtend** for Apriori
- **dotenv, openai, requests**

### Frontend (JavaScript - React)
- **React** with functional components
- **Framer Motion** for animation
- **Chart.js + react-chartjs-2** for visualization
- **html2pdf.js** for report export
- **TailwindCSS + Custom CSS** for styling

---

## 📁 Project Structure


```
Intelligent_Business_Analytics_System/ 
│── backend/ # FastAPI Backend 
│ ├── api.py # Main API routes 
│ ├── db.py # Database connection 
│ ├── file_processing.py # Handles file uploads & data processing 
│ ├── sentiment_analysis.py # AI-powered sentiment analysis 
│ ├── revenue_forecasting.py # AI-based revenue forecasting 
│ ├── weather_analysis.py # Weather impact on revenue 
│ ├── customer_segmentation.py # AI-powered customer segmentation 
│ ├── anomaly_detection.py # Detects sales anomalies 
│ ├── product_recommendation.py # AI-based product recommendations 
│ ├── demand_forecasting.py # Predicts demand using AI 
│ ├── market_basket_analysis.py # Market Basket Analysis 
│ ├── upload.py # File upload API │ ├── config.py # Configuration settings 
│ ├── requirements.txt # Python dependencies 
│── frontend/ # React Frontend 
│ ├── src/ 
│ │ ├── components/ 
│ │ │ ├── FileUpload.js # Upload Excel files 
│ │ │ ├── SentimentChart.js # Sentiment Analysis Visualization 
│ │ │ ├── RevenueChart.js # Revenue Forecasting Chart 
│ │ │ ├── WeatherImpact.js # Weather impact on sales 
│ │ │ ├── CustomerSegments.js # Customer segmentation visualization 
│ │ │ ├── DemandForecast.js # Demand forecasting visualization 
│ │ │ ├── SalesAnomalies.js # Anomaly detection visualization 
│ │ │ ├── Recommendations.js # Product recommendations visualization 
│ │ │ ├── MarketBasket.js # Market Basket Analysis visualization 
│ │ ├── services/ 
│ │ │ ├── api.js # Axios API service 
│ │ ├── App.js # Main React App 
│ │ ├── index.js # React Index file 
│ │ ├── styles.css # CSS Styles 
│── tests/ # Unit Tests 
│── DocumentsAndReports/ # Documentation
│── README.md # Project Documentation
```

---

##  **Installation & Setup**
### ** Clone the Repository**
```bash
git clone https://github.com/Ed-chae/W25_4495_S2_SeungyeolC.git
cd W25_4495_S2_SeungyeolC-main
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
