from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import (
    POSITIVE_THRESHOLD,
    NEGATIVE_THRESHOLD
)
import pandas as pd


analyzer = SentimentIntensityAnalyzer()

def calculate_sentiment(text):
    """Calculate sentiment scores for a message."""

    if not isinstance(text, str) or not text.strip():
        return {
            "positive_score": 0.0,
            "neutral_score": 0.0,
            "negative_score": 0.0,
            "compound_score": 0.0,
            "sentiment": "Neutral"
        }
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= POSITIVE_THRESHOLD:
        label = "Positive"

    elif compound <= NEGATIVE_THRESHOLD:
        label = "Negative"

    else:
        label = "Neutral"

    return {
        "positive_score": scores["pos"],
        "neutral_score": scores["neu"],
        "negative_score": scores["neg"],
        "compound_score": compound,
        "sentiment": label
    }


def add_sentiment(df):
    """Add sentiment scores to the DataFrame."""

    df = df.copy()

    results = df["cleaned_text"].apply(calculate_sentiment)

    sentiment_df = results.apply(pd.Series)

    return pd.concat(
        [df.reset_index(drop=True), sentiment_df.reset_index(drop=True)],
        axis=1
    )



