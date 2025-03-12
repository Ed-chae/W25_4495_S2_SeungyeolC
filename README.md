# **Intelligent Business Analytics System**

## **Project Overview**
The **Intelligent Business Analytics System** is a data-driven application designed to analyze sales performance, customer sentiment, and weather impact on revenue. It provides insights through interactive visualizations and predictive revenue forecasting based on historical data.

---

## **Features**
-  **Excel File Upload**: Upload sales data for processing and analysis.
-  **Revenue Trends**: Track daily revenue fluctuations over time.
-  **Best-Selling Items**: Identify top-selling menu items based on sales volume.
-  **Customer Sentiment Analysis**: Evaluate positive and negative reviews for each menu item.
-  **Weather Impact Analysis**: Assess how different weather conditions affect sales.
-  **Revenue Forecasting**: Predict the next 7 days' revenue based on past trends and weather conditions.

---

## **Tech Stack**
- **Backend**: FastAPI, PostgreSQL, SQLAlchemy, Pandas, SpaCy, Scikit-learn
- **Frontend**: React.js, Chart.js, Axios
- **Deployment**: Uvicorn, Node.js

---

## **Installation & Setup**

### **Prerequisites:**
- **Python 3.10+**
- **Node.js 16+** and **npm**
- **PostgreSQL** database
- **Git**

### **1. Clone the Repository:**
```bash
git clone https://github.com/Ed-chae/W25_4495_S2_SeungyeolC.git
cd W25_4495_S2_SeungyeolC
```

### **2. Backend Setup (FastAPI):**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### **Environment Variables:**
Create a `.env` file with the following content:
```plaintext
DATABASE_URL=postgresql://postgres:4495@localhost/business_analytics
```
Replace `<username>` and `<password>` with your PostgreSQL credentials.

#### **Run the Backend Server:**
```bash
uvicorn main:app --reload
```
API Documentation: **http://localhost:8000/docs**

---

### **3. Frontend Setup (React.js):**
```bash
cd ../frontend
npm install
npm start
```
Frontend will be available at: **http://localhost:3000**

---

### **4. Testing the Application:**
1. **Upload Sample Data:** Use the **/upload/** endpoint at **http://localhost:8000/docs**.
2. **View the Dashboard:** Open **http://localhost:3000**.

---
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

---

## **API Endpoints**
| Method | Endpoint       | Description |
|--------|----------------|-------------|
| `POST` | `/upload/`     | Upload Excel file and process data |
| `GET`  | `/insights/`   | Retrieve advanced analytics insights |

---

## **Contributors**
- **Seungyeol Chae** - Developer
