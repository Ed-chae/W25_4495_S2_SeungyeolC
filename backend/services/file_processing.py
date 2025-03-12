import io
import pandas as pd
from sqlalchemy.orm import Session
from db import Order, SessionLocal
from textblob import TextBlob

def process_excel(contents):
    try:
        # ✅ Read Excel File
        df = pd.read_excel(io.BytesIO(contents))

        if df.empty:
            print("❌ Uploaded file is empty!")
            return {"error": "Uploaded file is empty"}

        # ✅ Convert 'Date' column to datetime & 'Time' to proper format
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.time

        # ✅ Handle missing values in Customer Review
        df["Customer Review"] = df["Customer Review"].fillna("")
        df["Customer Review"] = df["Customer Review"].astype(str).str.replace('"', '', regex=False)

        # ✅ Perform Sentiment Analysis on Reviews
        df["Sentiment Score"] = df["Customer Review"].apply(lambda x: TextBlob(x).sentiment.polarity)

        # ✅ Save Data to the Database
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
                sentiment_score=row["Sentiment Score"]
            )
            db.add(order)

        db.commit()
        db.close()

        print("✅ Data successfully saved to the database.")
        return {"message": "File processed successfully", "columns": df.columns.tolist()}
    
    except Exception as e:
        print("❌ Error processing Excel:", str(e))
        return {"error": str(e)}
