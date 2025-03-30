import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from db import SessionLocal, RestaurantOrder

# -------------------------------
# 📥 Fetch Transactions
# -------------------------------
def fetch_transactions():
    session = SessionLocal()
    orders = session.query(RestaurantOrder).filter(RestaurantOrder.menu_item != None).all()
    session.close()

    # Each item is treated as a line in a transaction
    df = pd.DataFrame([
        {"transaction_id": i + 1, "item": order.menu_item}
        for i, order in enumerate(orders)
    ])

    return df

# -------------------------------
# 🧺 Prepare Basket Matrix
# -------------------------------
def prepare_basket(df):
    """Converts transaction-item pairs into one-hot encoded basket format."""
    basket = df.groupby(["transaction_id", "item"])["item"].count().unstack().fillna(0)
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
            "message": "No restaurant transaction data found. Please upload data first."
        }

    basket_df = prepare_basket(df)
    results = apply_apriori(basket_df)

    return {
        "results": results,
        "message": "Market basket analysis completed successfully." if results else "No strong association rules found."
    }
