import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD
from db import SessionLocal, RestaurantOrder
from collections import defaultdict

# -----------------------------------
# 🧠 Neural Network Model
# -----------------------------------
class RecommendationNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=16):
        super(RecommendationNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc = nn.Sequential(
            nn.Linear(embedding_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.fc(x)

# -----------------------------------
# 📥 Fetch Restaurant Purchases
# -----------------------------------
def fetch_purchase_data():
    session = SessionLocal()
    orders = session.query(RestaurantOrder).filter(RestaurantOrder.menu_item != None).all()
    session.close()

    df = pd.DataFrame([{
        "order_id": o.order_id,
        "item": o.menu_item,
        "rating": 5 if "love" in str(o.review).lower() or "great" in str(o.review).lower() else 2,
        "customer_id": abs(hash(str(o.order_id))) % 1000  # pseudo user
    } for o in orders])

    df.columns = df.columns.str.strip().str.lower()
    if "order id" in df.columns:
        df.rename(columns={"order id": "order_id"}, inplace=True)

    return df

# -----------------------------------
# 🔢 SVD Collaborative Filtering
# -----------------------------------
def collaborative_filtering(df):
    matrix = df.groupby(["customer_id", "item"])["rating"].mean().unstack().fillna(0)

    svd = TruncatedSVD(n_components=5, random_state=42)
    item_matrix = svd.fit_transform(matrix)

    recs = {}
    for i, user_id in enumerate(matrix.index):
        top_indices = np.argsort(item_matrix[i])[-3:]
        recs[user_id] = [matrix.columns[j] for j in top_indices]

    return recs

# -----------------------------------
# 🧠 Train Neural Network
# -----------------------------------
def train_neural_network(df):
    user_ids = df["customer_id"].unique()
    item_names = df["item"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {i: j for j, i in enumerate(item_names)}

    df["user_idx"] = df["customer_id"].map(user_to_idx)
    df["item_idx"] = df["item"].map(item_to_idx)

    model = RecommendationNN(len(user_ids), len(item_names))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(50):
        user_tensor = torch.tensor(df["user_idx"].values, dtype=torch.long)
        item_tensor = torch.tensor(df["item_idx"].values, dtype=torch.long)
        rating_tensor = torch.tensor(df["rating"].values, dtype=torch.float32).unsqueeze(-1)

        optimizer.zero_grad()
        output = model(user_tensor, item_tensor)
        loss = criterion(output, rating_tensor)
        loss.backward()
        optimizer.step()

    return model, user_to_idx, item_to_idx

# -----------------------------------
# 🔮 Recommend Products
# -----------------------------------
def recommend_products(user_id: str):
    df = fetch_purchase_data()
    if df.empty:
        return {"error": "No purchase data available for recommendations."}

    try:
        user_id_int = int(user_id)
    except ValueError:
        return {"error": f"Invalid user_id '{user_id}' — must be numeric."}

    svd_recs = collaborative_filtering(df)
    model, user_to_idx, item_to_idx = train_neural_network(df)

    user_idx = user_to_idx.get(user_id_int)
    if user_idx is None:
        return {"error": f"User {user_id} not found."}

    item_scores = []
    for item, idx in item_to_idx.items():
        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        item_tensor = torch.tensor([idx], dtype=torch.long)
        score = model(user_tensor, item_tensor).item()
        item_scores.append((item, score))

    # ✅ Remove already purchased items
    purchased_items = df[df["customer_id"] == user_id_int]["item"].tolist()
    filtered_scores = [item for item in item_scores if item[0] not in purchased_items]

    # ✅ Get top 3 recommendations
    top_nn_recs = [item for item, _ in sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:3]]

    return {
        "svd_recommendations": svd_recs.get(user_id_int, []),
        "nn_recommendations": top_nn_recs
    }
