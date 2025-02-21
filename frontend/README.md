# **Intelligent Business Analytics System**

## **Project Overview**
The **Intelligent Business Analytics System** is a data-driven application designed to analyze sales performance, customer sentiment, and weather impact on revenue. It provides insights through interactive visualizations and predictive revenue forecasting based on historical data.

## **Features**
-  **Excel File Upload**: Upload sales data for processing and analysis.
-  **Revenue Trends**: Track daily revenue fluctuations over time.
-  **Best-Selling Items**: Identify top-selling menu items based on sales volume.
-  **Customer Sentiment Analysis**: Evaluate positive and negative reviews for each menu item.
-  **Weather Impact Analysis**: Assess how different weather conditions affect sales.
-  **Revenue Forecasting**: Predict the next 7 days' revenue based on past trends and weather conditions.

## **Tech Stack**
- **Backend**: FastAPI, PostgreSQL, SQLAlchemy, Pandas, TextBlob
- **Frontend**: React.js, Chart.js, Axios
- **Deployment**: Uvicorn, Node.js

## **Installation & Setup**
### **Backend Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/Ed-chae/W25_4495_S2_SeungyeolC.git
   cd IntelligentBusinessAnalytics/backend
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### **Frontend Setup**
1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the React development server:
   ```bash
   npm start
   ```

## **Usage**
1. Open the frontend at `http://localhost:3000/`
2. Upload an Excel file containing restaurant sales data.
3. View analytics dashboards with revenue trends, best-selling items, and sentiment analysis.
4. Check predicted revenue for the next 7 days based on weather conditions.

## **File Structure**
```
IntelligentBusinessAnalytics/
│── backend/
│   ├── main.py            # FastAPI application with analytics endpoints
│   ├── db.py              # Database configuration and connection
│   ├── upload.py          # File processing and data storage
│   ├── analytics.py       # Data analysis and forecasting logic
│── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.js   # Displays analytics data and visualizations
│   │   │   ├── FileUpload.js  # Handles file uploads
│   │   ├── services/
│   │   │   ├── api.js         # API calls to backend
│   ├── public/
│   ├── package.json
│── README.md
```

## **API Endpoints**
| Method | Endpoint       | Description |
|--------|---------------|-------------|
| `POST` | `/upload/`    | Uploads an Excel file and stores sales data. |
| `GET`  | `/analytics/` | Retrieves revenue trends, sentiment analysis, and predictions. |

## **Contributors**
- **Seungyeol Chae** - Developer

