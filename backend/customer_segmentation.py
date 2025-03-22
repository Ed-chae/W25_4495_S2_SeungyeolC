import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from db import SessionLocal, SalesData
from sqlalchemy.orm import sessionmaker

def fetch_customer_data():
    """Fetch sales data from the database and preprocess for segmentation."""
    session = SessionLocal()
    sales_data = session.query(SalesData).all()
    session.close()

    df = pd.DataFrame([{
        "customer_id": s.id, 
        "revenue": s.revenue,
        "purchase_count": np.random.randint(1, 15)  # Simulating purchase count
    } for s in sales_data])

    return df

def apply_kmeans_clustering(df, n_clusters=3):
    """Applies K-Means clustering on customer data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    required_columns = ["revenue", "purchase_count"]
    for col in required_columns:
        if col not in df.columns:
            print(f"Column {col} not found in DataFrame. Available columns: {df.columns}")

    return df

def apply_dbscan_clustering(df):
    """Applies DBSCAN clustering for anomaly detection in purchases."""
    dbscan = DBSCAN(eps=5, min_samples=2)
    df["dbscan_cluster"] = dbscan.fit_predict(df[["revenue", "purchase_count"]])
    return df

def segment_customers():
    """Fetches data, applies clustering, and returns customer segments."""
    df = fetch_customer_data()

    # Apply AI-based clustering
    df = apply_kmeans_clustering(df)
    df = apply_dbscan_clustering(df)

    return df.to_dict(orient="records")
