import io
import pandas as pd
from sqlalchemy.orm import Session
from db import Order, SessionLocal
from textblob import TextBlob

def process_excel(contents):
    try:
        df = pd.read_excel(io.BytesIO(contents))

        # ✅ Convert 'Time' to proper format
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.time

        # ✅ Handle missing values
        df["Customer Review"] = df["Customer Review"].fillna("")
        df["Customer Review"] = df["Customer Review"].astype(str).str.replace('"', '', regex=False)

        # ✅ Save data to the database
        db = SessionLocal()
        for _, row in df.iterrows():
            order = Order(
                date=row["Date"],
                time=row["Time"],
                order_id=row["Order ID"],
                menu_item=row["Menu Item"],
                quantity=row["Quantity"],
                total_price=row["Total Price"],
                payment_method=row["Payment Method"],
                customer_review=row["Customer Review"],
                weather_condition=row["Weather Condition"],
            )
            db.add(order)

        db.commit()
        db.close()

        print("✅ Data successfully saved to the database.")
        return {"message": "File processed successfully", "columns": df.columns.tolist()}
    
    except Exception as e:
        print("❌ Error processing Excel:", str(e))
        return {"error": str(e)}
