from transformers import pipeline
import os

# Load Hugging Face sentiment model once (BERT-based)
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Returns sentiment label and score for a given review string."""
    try:
        if not text or str(text).strip() == "":
            return {"label": "NEUTRAL", "score": 0.0}

        result = sentiment_pipeline(str(text)[:512])[0]  # Truncate to 512 tokens
        return {
            "label": result["label"].upper(),  # Ensure consistency
            "score": float(result["score"])
        }

    except Exception as e:
        # In case of model failure or input issues
        return {"label": "NEUTRAL", "score": 0.0}
