from transformers import pipeline

# Load Hugging Face sentiment model once globally (uses BERT by default)
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given review text.

    Returns:
        dict: {
            "label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
            "score": float (confidence score)
        }
    """
    try:
        if not text or str(text).strip() == "":
            return {"label": "NEUTRAL", "score": 0.0}

        result = sentiment_pipeline(str(text)[:512])[0]
        return {
            "label": result["label"].upper(),
            "score": float(result["score"])
        }

    except Exception:
        return {"label": "NEUTRAL", "score": 0.0}
