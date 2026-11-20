import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for environments without a display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path as path


def create_visualizations(
    df,
    politician_df,
    sentiment_df,
    hashtag_df,
    output_dir
):
    """Generate and save project charts."""

    output_dir = path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

#barplot for sentiment distribution

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=sentiment_df,
        x="sentiment",
        y="count"
    )

    plt.title("Social Media Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Messages")
    plt.tight_layout()

    plt.savefig(output_dir / "sentiment_distribution.png")
    plt.close()


#most popular hastags
    if not hashtag_df.empty:

        top_hashtags = hashtag_df.head(10)

        plt.figure(figsize=(10, 6))
        max_count = top_hashtags["count"].max()
        colors = [
            "#e63946" if count == max_count else "#d3d3d3"
            for count in top_hashtags["count"]
        ]

        sns.barplot(
            data=top_hashtags,
            x="count",
            y="hashtag",
            palette=colors
        )

        plt.title("Top 10 Popular Hashtags")
        plt.xlabel("Frequency")
        plt.ylabel("Hashtag")
        plt.tight_layout()

        plt.savefig(output_dir / "top_hashtags.png")
        plt.close()

        #politician activity
    if not politician_df.empty:

        top_politicians = politician_df.head(10)

        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=top_politicians,
            x="total_messages",
            y="author"
        )

        plt.title("Top 10 Politicians by Message Count")
        plt.xlabel("Messages")
        plt.ylabel("Politician")
        plt.tight_layout()

        plt.savefig(output_dir / "politician_activity.png")
        plt.close()

    if not politician_df.empty:

        sentiment_data = politician_df.sort_values(
            "average_sentiment",
            ascending=False
        ).head(10)

        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=sentiment_data,
            x="average_sentiment",
            y="author"
        )

        plt.axvline(0, linestyle="--", linewidth=1)

        plt.title("Politicians with Highest Average Sentiment")
        plt.xlabel("Average VADER Compound Score")
        plt.ylabel("Politician")
        plt.tight_layout()

        plt.savefig(output_dir / "politician_sentiment.png")
        plt.close()

    print(f"Charts saved to: {output_dir}")











