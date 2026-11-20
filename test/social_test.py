
import pandas as pd

from src.preprocessing import (
    clean_text,
    extract_hashtags,
    extract_mentions,
    standardize_platform
)

from src.sentiment import calculate_sentiment

from src.analysis import (
    sentiment_analysis,
    hashtag_analysis,
    politician_analysis
)


def test_clean_text():

    result = clean_text(
        "Hello   world https://example.com"
    )

    assert result == "Hello world"


def test_extract_hashtags():

    result = extract_hashtags(
        "Great work! #Python #DataScience"
    )

    assert result == ["#python", "#datascience"]


def test_extract_mentions():

    result = extract_mentions(
        "Hello @john and @alex"
    )

    assert result == ["@john", "@alex"]


def test_standardize_platform():

    assert standardize_platform("twitter") == "Twitter"

    assert standardize_platform("Facebook") == "Facebook"


def test_positive_sentiment():

    result = calculate_sentiment(
        "This is wonderful and amazing!"
    )

    assert result["sentiment"] == "Positive"


def test_negative_sentiment():

    result = calculate_sentiment(
        "This is terrible and disappointing."
    )

    assert result["sentiment"] == "Negative"


def test_hashtag_analysis():

    df = pd.DataFrame({
        "hashtags": [
            ["#python", "#data"],
            ["#python"],
            ["#data"]
        ]
    })

    result = hashtag_analysis(df)

    counts = dict(zip(result["hashtag"], result["count"]))

    assert counts["#python"] == 2
    assert counts["#data"] == 2


def test_sentiment_analysis():

    df = pd.DataFrame({
        "sentiment": [
            "Positive",
            "Negative",
            "Positive",
            "Neutral"
        ]
    })

    result = sentiment_analysis(df)

    assert result["count"].sum() == 4


def test_politician_analysis():

    df = pd.DataFrame({
        "author": ["John", "John", "Alex"],
        "cleaned_text": ["A", "B", "C"],
        "compound_score": [0.5, -0.2, 0.8],
        "sentiment": ["Positive", "Negative", "Positive"]
    })

    result = politician_analysis(df)

    assert len(result) == 2
    assert result.iloc[0]["author"] == "John"
    assert result.iloc[0]["total_messages"] == 2
