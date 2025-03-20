import pandas as pd
from db import save_sales_data
from sentiment_analysis import analyze_sentiment

def process_sales_data(file_path):
    """Processes Excel sales data, analyzes reviews, and saves to DB."""
    df = pd.read_excel(file_path)

    # Ensure required columns exist
    required_columns = ["Date", "Product", "Revenue", "Review"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns in uploaded file.")

    # Apply sentiment analysis to reviews
    df["Sentiment"] = df["Review"].apply(lambda x: analyze_sentiment(str(x))["label"])

    # Save data to database
    save_sales_data(df)

    return df
