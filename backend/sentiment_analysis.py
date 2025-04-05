# sentiment_analysis.py

from transformers import pipeline
from db import SessionLocal, SalesData, RestaurantOrder

sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    if not text or not isinstance(text, str):
        return {"label": "NEUTRAL"}
    return sentiment_pipeline(text[:512])[0]

def sentiment_summary():
    db = SessionLocal()
    sales_reviews = db.query(SalesData).all()
    restaurant_reviews = db.query(RestaurantOrder).all()
    db.close()

    summary = {}

    for r in sales_reviews:
        if r.product and r.review:
            sentiment = analyze_sentiment(r.review)["label"]
            item = r.product
            if item not in summary:
                summary[item] = {"positive": 0, "negative": 0}
            if sentiment == "POSITIVE":
                summary[item]["positive"] += 1
            else:
                summary[item]["negative"] += 1

    for r in restaurant_reviews:
        if r.menu_item and r.review:
            sentiment = analyze_sentiment(r.review)["label"]
            item = r.menu_item
            if item not in summary:
                summary[item] = {"positive": 0, "negative": 0}
            if sentiment == "POSITIVE":
                summary[item]["positive"] += 1
            else:
                summary[item]["negative"] += 1

    result = []
    for item, stats in summary.items():
        total = stats["positive"] + stats["negative"]
        if total == 0:
            continue
        positive_pct = (stats["positive"] / total) * 100
        negative_pct = 100 - positive_pct
        result.append({
            "item": item,
            "positive": stats["positive"],
            "negative": stats["negative"],
            "positive_pct": round(positive_pct, 1),
            "negative_pct": round(negative_pct, 1)
        })

    if not result:
        return {"summary": [], "best_item": None, "worst_item": None}

    sorted_result = sorted(result, key=lambda x: x["positive_pct"], reverse=True)
    best_item = sorted_result[0]
    worst_item = sorted(result, key=lambda x: x["negative_pct"], reverse=True)[0]

    return {
        "summary": sorted_result,
        "best_item": f'{best_item["item"]} ({best_item["positive_pct"]}% positive)',
        "worst_item": f'{worst_item["item"]} ({worst_item["negative_pct"]}% negative)'
    }
