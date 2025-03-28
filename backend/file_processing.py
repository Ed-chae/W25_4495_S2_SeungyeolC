import pandas as pd
from db import save_sales_data, save_restaurant_orders
from sentiment_analysis import analyze_sentiment


def process_sales_data(file_path):
    """Processes uploaded Excel data: Sales or Restaurant orders."""
    df = pd.read_excel(file_path)

    # Case 1: Check if it's a sales data file (Product-based)
    sales_cols = ["Date", "Product", "Revenue", "Review"]
    if all(col in df.columns for col in sales_cols):
        return process_standard_sales(df)

    # Case 2: Check if it's a restaurant order file (Menu-based)
    restaurant_cols = ["Date", "Item", "Quantity", "Price", "Customer Review"]
    if all(col in df.columns for col in restaurant_cols):
        return process_restaurant_data(df)

    # If neither format matched
    raise ValueError("❌ Uploaded file format is not recognized. Missing required columns.")


def process_standard_sales(df: pd.DataFrame):
    """Process standard sales format data with sentiment analysis."""
    df["Sentiment"] = df["Review"].apply(lambda x: analyze_sentiment(str(x))["label"])
    save_sales_data(df)
    return df


def process_restaurant_data(df: pd.DataFrame):
    """Process restaurant-style order data with sentiment analysis."""
    # Rename columns to match database model
    df = df.rename(columns={
        "Item": "Menu",
        "Customer Review": "Review"
    })

    df["Sentiment"] = df["Review"].apply(lambda x: analyze_sentiment(str(x))["label"])

    # Optional: Calculate revenue if not included
    if "Revenue" not in df.columns:
        df["Revenue"] = df["Quantity"] * df["Price"]

    save_restaurant_orders(df)
    return df
