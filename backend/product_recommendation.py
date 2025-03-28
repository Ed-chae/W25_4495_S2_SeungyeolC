import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD
from db import SessionLocal, RestaurantOrder
from collections import defaultdict

# -----------------------------------
# 🧠 Neural Network Model for Recs
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
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)
        x = torch.cat([user_embedded, item_embedded], dim=1)
        return self.fc(x)

# -----------------------------------
# 📥 Fetch Restaurant Orders
# -----------------------------------
def fetch_purchase_data():
    """Fetches customer-item interactions for restaurant data."""
    session = SessionLocal()
    orders = session.query(RestaurantOrder).all()
    session.close()

    df = pd.DataFrame([
        {
            "customer_id": np.random.randint(1, 100),  # Simulated customer ID
            "item": order.menu_item,
            "rating": np.random.randint(1, 6)  # Simulated feedback
        }
        for order in orders
    ])

    return df

# -----------------------------------
# 🔢 Collaborative Filtering (SVD)
# -----------------------------------
def collaborative_filtering(df):
    """Uses SVD for collaborative filtering-based recommendations."""
    user_item_matrix = df.pivot(index="customer_id", columns="item", values="rating").fillna(0)

    svd = TruncatedSVD(n_components=5, random_state=42)
    item_factors = svd.fit_transform(user_item_matrix)

    recommendations = defaultdict(list)
    for i, customer_id in enumerate(user_item_matrix.index):
        top_items = np.argsort(item_factors[i])[-3:]
        recommendations[customer_id] = [user_item_matrix.columns[j] for j in top_items]

    return recommendations

# -----------------------------------
# 🔁 Train Neural Network Recommender
# -----------------------------------
def train_neural_network(df):
    """Trains a neural network for item recommendations."""
    num_users = df["customer_id"].nunique()
    num_items = df["item"].nunique()

    user_to_idx = {u: i for i, u in enumerate(df["customer_id"].unique())}
    item_to_idx = {i: j for j, i in enumerate(df["item"].unique())}

    df["user_idx"] = df["customer_id"].map(user_to_idx)
    df["item_idx"] = df["item"].map(item_to_idx)

    model = RecommendationNN(num_users, num_items)
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
# 🔮 Recommend Menu Items
# -----------------------------------
def recommend_products(user_id):
    """Generates menu item recommendations for a simulated restaurant user."""
    df = fetch_purchase_data()

    # SVD (Collaborative Filtering)
    svd_recs = collaborative_filtering(df)

    # Neural Network Recs
    model, user_to_idx, item_to_idx = train_neural_network(df)
    user_idx = user_to_idx.get(user_id)

    if user_idx is None:
        return {"error": f"User {user_id} not found."}

    item_scores = {}
    for item, idx in item_to_idx.items():
        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        item_tensor = torch.tensor([idx], dtype=torch.long)
        score = model(user_tensor, item_tensor).item()
        item_scores[item] = score

    top_nn_recs = sorted(item_scores, key=item_scores.get, reverse=True)[:3]

    return {
        "svd_recommendations": svd_recs.get(user_id, []),
        "nn_recommendations": top_nn_recs
    }
