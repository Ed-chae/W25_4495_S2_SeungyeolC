import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from db import SessionLocal, RestaurantOrder

# -----------------------------
# 📥 Fetch Restaurant Orders
# -----------------------------
def fetch_customer_data():
    """
    Fetch restaurant orders and simulate customer behavior using random customer IDs.
    """
    session = SessionLocal()
    orders = session.query(RestaurantOrder).filter(RestaurantOrder.price != None, RestaurantOrder.quantity != None).all()
    session.close()

    if not orders:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "customer_id": o.id,  # Simulated individual customer
            "revenue": o.quantity * o.price,
            "purchase_count": np.random.randint(1, 10)  # Simulated count
        }
        for o in orders
    ])

    return df

# -----------------------------
# 📊 Clustering Algorithms
# -----------------------------
def apply_kmeans_clustering(df, n_clusters=3):
    """Apply KMeans clustering based on revenue and purchase count."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["kmeans_cluster"] = kmeans.fit_predict(df[["revenue", "purchase_count"]])
    return df

def apply_dbscan_clustering(df):
    """Apply DBSCAN for density-based clustering (e.g., outlier detection)."""
    dbscan = DBSCAN(eps=5, min_samples=2)
    df["dbscan_cluster"] = dbscan.fit_predict(df[["revenue", "purchase_count"]])
    return df

# -----------------------------
# 🧠 Explain Cluster Segments
# -----------------------------
def explain_clusters(df):
    """
    Create human-readable labels for each KMeans cluster.
    """
    summaries = []
    for cluster_id in sorted(df["kmeans_cluster"].unique()):
        group = df[df["kmeans_cluster"] == cluster_id]
        avg_rev = group["revenue"].mean()
        avg_count = group["purchase_count"].mean()

        if avg_rev > 100 and avg_count > 5:
            label = "💎 VIP Customers"
        elif avg_rev > 50:
            label = "📦 Regular Buyers"
        else:
            label = "🛍️ Low-Spend Shoppers"

        summaries.append({
            "cluster_id": int(cluster_id),
            "label": label,
            "avg_revenue": round(avg_rev, 2),
            "avg_purchase_count": round(avg_count, 2),
            "total_customers": len(group)
        })

    return summaries

# -----------------------------
# 🚀 Main Segmentation Function
# -----------------------------
def segment_customers():
    """
    Endpoint logic for customer segmentation results.
    """
    df = fetch_customer_data()

    if df.empty:
        return {
            "summary": [],
            "raw_data": [],
            "message": "No restaurant customer data available for segmentation."
        }

    df = apply_kmeans_clustering(df)
    df = apply_dbscan_clustering(df)
    summary = explain_clusters(df)

    return {
        "summary": summary,
        "raw_data": df.to_dict(orient="records")
    }
