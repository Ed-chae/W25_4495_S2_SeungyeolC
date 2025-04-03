import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from db import SessionLocal, RestaurantOrder

def fetch_transactions():
    session = SessionLocal()
    orders = session.query(RestaurantOrder).filter(RestaurantOrder.order_id != None).all()
    session.close()

    if not orders:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "order_id": o.order_id,
        "menu_item": o.menu_item
    } for o in orders if o.order_id and o.menu_item])

    return df

def prepare_basket(df):
    basket = df.groupby(['order_id', 'menu_item'])['menu_item'].count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket

def apply_apriori(basket_df):
    frequent_itemsets = apriori(basket_df, min_support=0.05, use_colnames=True)
    if frequent_itemsets.empty:
        return []

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_dict(orient="records")

def market_basket_analysis():
    df = fetch_transactions()

    if df.empty or "menu_item" not in df.columns:
        return {
            "results": [],
            "message": "⚠️ No transaction data found. Please upload valid order data first."
        }

    basket_df = prepare_basket(df)
    results = apply_apriori(basket_df)

    return {
        "results": results,
        "message": "✅ Market basket analysis completed." if results else "⚠️ No strong association rules found."
    }
