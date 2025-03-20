import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from db import SessionLocal, SalesData

def fetch_transaction_data():
    """Fetch sales transactions from the database."""
    session = SessionLocal()
    sales_data = session.query(SalesData).all()
    session.close()

    df = pd.DataFrame([{"transaction_id": np.random.randint(1, 100), "product": s.product} for s in sales_data])
    return df

def prepare_market_basket_data(df):
    """Transforms data into basket format for association rule mining."""
    basket = df.pivot_table(index="transaction_id", columns="product", aggfunc=lambda x: 1, fill_value=0)
    return basket

def apply_apriori(df, min_support=0.02):
    """Applies the Apriori algorithm to find frequent itemsets."""
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_threshold=0.5):
    """Generates association rules from frequent itemsets."""
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]]

def market_basket_analysis():
    """Runs Market Basket Analysis and returns frequent itemsets & rules."""
    df = fetch_transaction_data()
    basket_df = prepare_market_basket_data(df)
    
    # Apply Apriori Algorithm
    frequent_itemsets = apply_apriori(basket_df)
    
    # Generate Association Rules
    rules = generate_association_rules(frequent_itemsets)

    return {
        "frequent_itemsets": frequent_itemsets.to_dict(orient="records"),
        "association_rules": rules.to_dict(orient="records")
    }
