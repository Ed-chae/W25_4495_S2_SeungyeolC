from transformers import pipeline
import os

# Load once globally
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text[:512])[0]  # Truncate to 512 tokens
    return {
        "label": result["label"],  # "POSITIVE" or "NEGATIVE"
        "score": float(result["score"])  # Optional: confidence score
    }
