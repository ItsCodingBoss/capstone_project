import re
import pandas as pd


def clean_text(text):
    """Clean text by removing URLs, mentions, hashtags, and special characters.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_hashtags(text):
    """Extract hashtags from text and return them as a list."""
    if not isinstance(text, str):
        return []

    return[
        tag.lower()
        for tag in re.findall(r"(?<!\w)#\w+",text)
    ]


def extract_mentions(text):
    """Extract social media mentions."""

    if not isinstance(text, str):
        return []

    return [
        mention.lower()
        for mention in re.findall(r"(?<!\w)@\w+", text)
    ]


def extract_hashtag_count(tags):
    """Count hashtags."""

    return len(tags)

def standardize_platform(platform):
    """Standardize social media platform names."""
    if pd.isna(platform):
        return "unknown"

    platform = str(platform).strip().lower()
    if "twitter" in platform or "tweet" in platform:
        return "Twitter"

    if "facebook" in platform:
        return "Facebook"

    return platform.title()


def process_data(df,columns):
    """perform complete data processing on the dataset."""


    df = df.copy()

    text_column = columns["text"]

    df = df.dropna(subset=[text_column])

    df["cleaned_text"] = df[text_column].apply(clean_text)

    df = df[df["cleaned_text"] != ""].copy()

    df["hashtags"] = df["cleaned_text"].apply(extract_hashtags)

    df["mentions"] = df["cleaned_text"].apply(extract_mentions)

    df["hashtag_count"] = df["hashtags"].apply(
        extract_hashtag_count
    )


        # Standardize author.
    if "author" in columns:
        author_column = columns["author"]
        df["author"] = df[author_column].fillna("Unknown").astype(str).str.strip()
    else:
        df["author"] = "Unknown"

        #location
    if "location" in columns:
        location_column = columns["location"]
        df["location"] = df[location_column].fillna("Unknown").astype(str).str.strip()
    else:
        df["location"] = "Unknown"
        #position
    if "position" in columns:
        position_column = columns["position"]
        df["position"] = df[position_column].fillna("Unknown").astype(str).str.strip()
    else:
        df["position"] = "Unknown"
        #platform
    if "platform" in columns:
        platform_column = columns["platform"]
        df["platform"] = df[platform_column].apply(
            standardize_platform
        )
    else:
        df["platform"] = "Unknown"

    if "date" in columns:
        date_column = columns["date"]
        df["date"] = pd.to_datetime(
            df[date_column],
            errors="coerce",
            utc=True
        )
    else:
        df["date"] = pd.NaT

    df = df.drop_duplicates(
        subset=["author", "cleaned_text"]
    ).reset_index(drop=True)

    return df    
    







