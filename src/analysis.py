
from collections import Counter
import pandas as pd


def politician_analysis(df):
    """Analyze activity and sentiment by politician."""

    result = df.groupby("author").agg(
        total_messages=("cleaned_text", "count"),
        average_sentiment=("compound_score", "mean"),
        positive_messages=(
            "sentiment",
            lambda x: (x == "Positive").sum()
        ),
        negative_messages=(
            "sentiment",
            lambda x: (x == "Negative").sum()
        ),
        neutral_messages=(
            "sentiment",
            lambda x: (x == "Neutral").sum()
        )
    )

    return result.sort_values(
        "total_messages",
        ascending=False
    ).reset_index()


def sentiment_analysis(df):
    """Calculate overall sentiment distribution."""

    result = df["sentiment"].value_counts().reset_index()

    result.columns = ["sentiment", "count"]

    result["percentage"] = (
        result["count"] / len(df) * 100
    ).round(2)

    return result


def hashtag_analysis(df):
    """Find the most popular hashtags."""

    all_hashtags = [
        tag
        for tags in df["hashtags"]
        for tag in tags
    ]

    counts = Counter(all_hashtags)

    result = pd.DataFrame(
        counts.items(),
        columns=["hashtag", "count"]
    )

    return result.sort_values(
        "count",
        ascending=False
    ).reset_index(drop=True)


def hashtag_sentiment_analysis(df):
    """Calculate average sentiment for each hashtag."""

    rows = []

    for _, row in df.iterrows():

        for hashtag in row["hashtags"]:

            rows.append({
                "hashtag": hashtag,
                "compound_score": row["compound_score"],
                "sentiment": row["sentiment"]
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "hashtag",
                "message_count",
                "average_sentiment",
                "positive_count",
                "negative_count"
            ]
        )

    hashtag_df = pd.DataFrame(rows)

    result = hashtag_df.groupby("hashtag").agg(
        message_count=("compound_score", "count"),
        average_sentiment=("compound_score", "mean"),
        positive_count=(
            "sentiment",
            lambda x: (x == "Positive").sum()
        ),
        negative_count=(
            "sentiment",
            lambda x: (x == "Negative").sum()
        )
    ).reset_index()

    return result.sort_values(
        "message_count",
        ascending=False
    )


def platform_analysis(df):
    """Compare activity across social media platforms."""

    return df.groupby("platform").agg(
        total_messages=("cleaned_text", "count"),
        average_sentiment=("compound_score", "mean"),
        positive_messages=(
            "sentiment",
            lambda x: (x == "Positive").sum()
        ),
        negative_messages=(
            "sentiment",
            lambda x: (x == "Negative").sum()
        )
    ).reset_index()


def generate_summary(df):
    """Generate overall project statistics."""

    return {
        "total_messages": int(len(df)),
        "total_politicians": int(
            df["author"].nunique()
        ),
        "total_hashtags": int(
            len({
                tag
                for tags in df["hashtags"]
                for tag in tags
            })
        ),
        "average_sentiment": round(
            float(df["compound_score"].mean()), 4
        ),
        "positive_messages": int(
            (df["sentiment"] == "Positive").sum()
        ),
        "negative_messages": int(
            (df["sentiment"] == "Negative").sum()
        ),
        "neutral_messages": int(
            (df["sentiment"] == "Neutral").sum()
        )
    }
