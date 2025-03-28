import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from db import SessionLocal, SalesData
from sqlalchemy.orm import sessionmaker

def fetch_customer_data():
    """
    Fetch sales data from the database and preprocess for segmentation.
    Each entry is treated as a unique customer for now (simulation).
    """
    session = SessionLocal()
    sales_data = session.query(SalesData).all()
    session.close()

    df = pd.DataFrame([{
        "customer_id": s.id,
        "revenue": s.revenue,
        "purchase_count": np.random.randint(1, 10)  # Simulated purchases
    } for s in sales_data if s.revenue is not None])

    return df

def apply_kmeans_clustering(df, n_clusters=3):
    """Applies KMeans clustering based on revenue and purchase count."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["kmeans_cluster"] = kmeans.fit_predict(df[["revenue", "purchase_count"]])
    return df

def apply_dbscan_clustering(df):
    """Applies DBSCAN to detect density-based customer patterns."""
    dbscan = DBSCAN(eps=5, min_samples=2)
    df["dbscan_cluster"] = dbscan.fit_predict(df[["revenue", "purchase_count"]])
    return df

def explain_clusters(df):
    """
    Add a simple interpretation of each cluster to help users understand results.
    """
    explanations = []
    for cluster_id in sorted(df["kmeans_cluster"].unique()):
        segment_df = df[df["kmeans_cluster"] == cluster_id]
        avg_revenue = segment_df["revenue"].mean()
        avg_count = segment_df["purchase_count"].mean()

        if avg_revenue > 100 and avg_count > 5:
            label = "💎 VIP Customers"
        elif avg_revenue > 50:
            label = "📦 Regular Buyers"
        else:
            label = "🛍️ Low-Spend Shoppers"

        explanations.append({
            "cluster_id": int(cluster_id),
            "label": label,
            "avg_revenue": round(avg_revenue, 2),
            "avg_purchase_count": round(avg_count, 2),
            "total_customers": len(segment_df)
        })

    return explanations

def segment_customers():
    """
    Main function to segment customers with AI-based clustering and explanations.
    """
    df = fetch_customer_data()

    if df.empty:
        return {"error": "No customer data available for segmentation."}

    df = apply_kmeans_clustering(df)
    df = apply_dbscan_clustering(df)
    summary = explain_clusters(df)

    return {
        "summary": summary,
        "raw_data": df.to_dict(orient="records")
    }
