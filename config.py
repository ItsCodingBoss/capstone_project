
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = BASE_DIR / "data" / "data.txt"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "enriched_social_media.csv"

CHART_DIR = BASE_DIR / "outputs" / "charts"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

# Change these if your dataset uses different column names.
COLUMN_MAPPING = {
    "text": ["text", "message", "tweet", "content", "post"],
    "author": ["author", "politician", "username", "screen_name", "name"],
    "platform": ["platform", "source", "network"],
    "date": ["date", "created_at", "timestamp", "time"],
    "location": ["location", "city", "state"],
    "position": ["position", "title", "designation", "office"]
}

# VADER compound score thresholds.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
