from config import RAW_DATA_PATH,CHART_DIR
from src.data_loader import load_data, inspect_data, identify_columns, validate_data
from config import COLUMN_MAPPING
from src.visualization import create_visualizations
from src.analysis import (
    politician_analysis,
    sentiment_analysis,
    hashtag_analysis
)

from src.preprocessing import process_data
from src.sentiment import add_sentiment
# 1. Load dataset
df = load_data(RAW_DATA_PATH)

# 2. Identify and validate columns
columns = identify_columns(df, COLUMN_MAPPING)
validate_data(df, columns)

# 3. Preprocessing
df = process_data(df, columns)

# 4. Sentiment analysis
df = add_sentiment(df)

# 5. Analysis
politician_df = politician_analysis(df)
sentiment_df = sentiment_analysis(df)
hashtag_df = hashtag_analysis(df)

# 6. Visualization
create_visualizations(
    df,
    politician_df,
    sentiment_df,
    hashtag_df,
    CHART_DIR
)