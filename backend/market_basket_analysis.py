# market_basket_analysis.py

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from db import SessionLocal, RestaurantOrder

# -------------------------------
# 📥 Fetch Transactions
# -------------------------------
def fetch_transactions():
    session = SessionLocal()
    orders = session.query(RestaurantOrder).filter(RestaurantOrder.order_id != None).all()
    session.close()

    # Group by order_id to reconstruct transactions
    transactions = pd.DataFrame([{
        "order_id": o.order_id,
        "item": o.menu_item
    } for o in orders])

    return transactions

# -------------------------------
# 🧺 Prepare Basket Matrix
# -------------------------------
def prepare_basket(df):
    """Converts transactions into one-hot encoded basket format."""
    basket = df.groupby(['order_id', 'item'])['item'].count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket

# -------------------------------
# 🧠 Apply Apriori Algorithm
# -------------------------------
def apply_apriori(basket_df):
    """Generates frequent itemsets and association rules."""
    frequent_itemsets = apriori(basket_df, min_support=0.1, use_colnames=True)

    if frequent_itemsets.empty:
        return []

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_dict(orient="records")

# -------------------------------
# 🎯 Run Market Basket Analysis
# -------------------------------
def market_basket_analysis():
    df = fetch_transactions()

    if df.empty or "item" not in df.columns:
        return {
            "results": [],
            "message": "No transaction data found. Please upload valid order data first."
        }

    basket_df = prepare_basket(df)
    results = apply_apriori(basket_df)

    return {
        "results": results,
        "message": "✅ Market basket analysis completed." if results else "No strong association rules found."
    }
