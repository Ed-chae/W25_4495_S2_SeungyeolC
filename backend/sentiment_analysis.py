from transformers import pipeline

# Load Sentiment Analysis Model (DistilBERT)
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Returns sentiment label ('POSITIVE' or 'NEGATIVE') and confidence score."""
    if not text or text.strip() == "":
        return {"label": "NEUTRAL", "score": 0.0}

    result = sentiment_pipeline(text)[0]
    return {"label": result["label"], "score": result["score"]}
