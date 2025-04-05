from transformers import pipeline

# Load sentiment model once globally
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given review text using Hugging Face's pipeline.

    Args:
        text (str): The review text.

    Returns:
        dict: {
            "label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
            "score": float (confidence score)
        }
    """
    try:
        text = str(text).strip()

        if not text:
            return {"label": "NEUTRAL", "score": 0.0}

        # Truncate to 512 tokens to avoid overflow
        result = sentiment_pipeline(text[:512])[0]

        return {
            "label": result.get("label", "NEUTRAL").upper(),
            "score": float(result.get("score", 0.0))
        }

    except Exception as e:
        print("Sentiment analysis error:", e)
        return {"label": "NEUTRAL", "score": 0.0}
