import pandas as pd
from db import save_sales_data, save_restaurant_orders

# -----------------------------------
# 📂 File Processing Logic
# -----------------------------------
def process_sales_data(file_path: str):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower()  # Normalize headers to lowercase

    # Fix: Rename 'order id' to 'order_id' for internal logic
    df.rename(columns={"order id": "order_id"}, inplace=True)

    # ✅ Restaurant Order format
    if all(col in df.columns for col in ["order_id", "date", "menu", "quantity", "price"]):
        save_restaurant_orders(df)
        return df

    # ✅ Sales format
    elif all(col in df.columns for col in ["order_id", "date", "product", "revenue"]):
        save_sales_data(df)
        return df

    else:
        raise ValueError("❌ Uploaded file format is not recognized. Missing required columns.")
