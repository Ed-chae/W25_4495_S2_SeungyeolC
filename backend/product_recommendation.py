import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD
from db import SessionLocal, SalesData
from collections import defaultdict

# Define Neural Network Model
class RecommendationNN(nn.Module):
    def __init__(self, num_users, num_products, embedding_size=16):
        super(RecommendationNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.product_embedding = nn.Embedding(num_products, embedding_size)
        self.fc = nn.Sequential(
            nn.Linear(embedding_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user, product):
        user_embedded = self.user_embedding(user)
        product_embedded = self.product_embedding(product)
        x = torch.cat([user_embedded, product_embedded], dim=1)
        return self.fc(x)

def fetch_purchase_data():
    """Fetch customer-product purchase history."""
    session = SessionLocal()
    sales_data = session.query(SalesData).all()
    session.close()

    df = pd.DataFrame([{
        "customer_id": np.random.randint(1, 100),  # Simulating user IDs
        "product": s.product,
        "rating": np.random.randint(1, 6)  # Simulating ratings
    } for s in sales_data])

    return df

def collaborative_filtering(df):
    """Uses SVD for collaborative filtering-based recommendations."""
    user_product_matrix = df.pivot(index="customer_id", columns="product", values="rating").fillna(0)

    svd = TruncatedSVD(n_components=5, random_state=42)
    product_factors = svd.fit_transform(user_product_matrix)

    recommendations = defaultdict(list)
    for i, customer_id in enumerate(user_product_matrix.index):
        top_products = np.argsort(product_factors[i])[-3:]  # Top 3 recommendations
        recommendations[customer_id] = [user_product_matrix.columns[j] for j in top_products]

    return recommendations

def train_neural_network(df):
    """Trains a deep learning-based product recommendation model."""
    num_users = df["customer_id"].nunique()
    num_products = df["product"].nunique()

    user_to_idx = {user: i for i, user in enumerate(df["customer_id"].unique())}
    product_to_idx = {product: i for i, product in enumerate(df["product"].unique())}

    df["user_idx"] = df["customer_id"].map(user_to_idx)
    df["product_idx"] = df["product"].map(product_to_idx)

    model = RecommendationNN(num_users, num_products)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        user_tensor = torch.tensor(df["user_idx"].values, dtype=torch.long)
        product_tensor = torch.tensor(df["product_idx"].values, dtype=torch.long)
        rating_tensor = torch.tensor(df["rating"].values, dtype=torch.float32).unsqueeze(-1)

        optimizer.zero_grad()
        output = model(user_tensor, product_tensor)
        loss = criterion(output, rating_tensor)
        loss.backward()
        optimizer.step()

    return model, user_to_idx, product_to_idx

def recommend_products(user_id):
    """Generates product recommendations using both SVD & Neural Network models."""
    df = fetch_purchase_data()

    # Collaborative Filtering
    svd_recommendations = collaborative_filtering(df)

    # Train Neural Network
    nn_model, user_to_idx, product_to_idx = train_neural_network(df)

    user_idx = user_to_idx.get(user_id, None)
    if user_idx is None:
        return {"error": "User not found"}

    product_scores = {}
    for product, product_idx in product_to_idx.items():
        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        product_tensor = torch.tensor([product_idx], dtype=torch.long)
        score = nn_model(user_tensor, product_tensor).item()
        product_scores[product] = score

    nn_recommendations = sorted(product_scores, key=product_scores.get, reverse=True)[:3]

    return {
        "svd_recommendations": svd_recommendations.get(user_id, []),
        "nn_recommendations": nn_recommendations
    }
