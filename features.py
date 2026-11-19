"""Feature engineering pipeline: Aggregates GDELT news categories and merges with Bitcoin prices."""

import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from config import DB_PATH, CATEGORIES


def extract_raw_dataframes(db_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load news articles and BTC prices from SQLite into pandas DataFrames."""
    target_path = db_path or str(DB_PATH)
    conn = sqlite3.connect(target_path)

    news_df = pd.read_sql_query("""
        SELECT timestamp, category, avg_tone, positive_score, negative_score,
               polarity, activity_density, word_count
        FROM news_articles;
    """, conn)

    btc_df = pd.read_sql_query("""
        SELECT timestamp, open, high, low, close, volume, trades_count
        FROM btc_prices
        ORDER BY timestamp ASC;
    """, conn)

    conn.close()

    if not news_df.empty:
        news_df["timestamp"] = pd.to_datetime(news_df["timestamp"])
    if not btc_df.empty:
        btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"])

    return news_df, btc_df


def aggregate_news_features(news_df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """
    Aggregate news sentiment, volume, and tone metrics across categories into time bins.
    Produces dense feature representations without lookahead bias.
    """
    if news_df.empty:
        return pd.DataFrame()

    # Floor timestamps to the interval
    news_df = news_df.copy()
    news_df["time_bucket"] = news_df["timestamp"].dt.floor(freq)

    # 1. Overall news statistics per time bucket
    global_agg = news_df.groupby("time_bucket").agg(
        total_news_volume=("avg_tone", "count"),
        global_avg_tone=("avg_tone", "mean"),
        global_tone_std=("avg_tone", "std"),
        global_avg_polarity=("polarity", "mean"),
        global_avg_negativity=("negative_score", "mean"),
        global_avg_positivity=("positive_score", "mean")
    ).fillna(0)

    # 2. Per-category statistics
    cat_frames = []
    for cat in CATEGORIES:
        cat_sub = news_df[news_df["category"] == cat]
        if cat_sub.empty:
            continue
        agg = cat_sub.groupby("time_bucket").agg(
            **{
                f"vol_{cat}": ("avg_tone", "count"),
                f"tone_{cat}": ("avg_tone", "mean"),
                f"neg_{cat}": ("negative_score", "mean"),
                f"pos_{cat}": ("positive_score", "mean"),
                f"pol_{cat}": ("polarity", "mean")
            }
        )
        cat_frames.append(agg)

    if cat_frames:
        cat_combined = pd.concat(cat_frames, axis=1).fillna(0)
        news_features = global_agg.join(cat_combined, how="left").fillna(0)
    else:
        news_features = global_agg

    # Calculate category volume shares
    for cat in CATEGORIES:
        vol_col = f"vol_{cat}"
        if vol_col in news_features.columns:
            news_features[f"share_{cat}"] = (
                news_features[vol_col] / (news_features["total_news_volume"] + 1e-6)
            )

    return news_features


def build_ml_dataset(freq: str = "1h", db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Build the complete aligned dataset for training:
    - Merges lagged news features with Bitcoin OHLCV.
    - Creates prediction targets: next-interval return and direction (1 for up, 0 for down).
    - Prevents data leakage by strictly using t-1 features for predicting period t -> t+1.
    """
    news_df, btc_df = extract_raw_dataframes(db_path)

    if btc_df.empty:
        raise ValueError("BTC price table is empty. Fetch BTC data first.")

    # Prepare BTC price series
    btc_df = btc_df.set_index("timestamp").sort_index()
    # Resample BTC to ensure regular time index
    btc_resampled = btc_df.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "trades_count": "sum"
    }).dropna()

    # Financial features on BTC
    btc_resampled["btc_log_return"] = np.log(btc_resampled["close"] / btc_resampled["close"].shift(1))
    btc_resampled["btc_volatility_6h"] = btc_resampled["btc_log_return"].rolling(6).std()
    btc_resampled["btc_volatility_24h"] = btc_resampled["btc_log_return"].rolling(24).std()
    btc_resampled["btc_hl_spread"] = (btc_resampled["high"] - btc_resampled["low"]) / btc_resampled["open"]

    # Target variable: NEXT period return and direction (strictly forward-looking)
    btc_resampled["target_next_return"] = btc_resampled["btc_log_return"].shift(-1)
    btc_resampled["target_direction"] = (btc_resampled["target_next_return"] > 0).astype(int)
    btc_resampled["target_next_volatility"] = btc_resampled["btc_hl_spread"].shift(-1)

    # Aggregate News features
    if not news_df.empty:
        news_features = aggregate_news_features(news_df, freq=freq)
        # Merge news features aligned with price
        merged = btc_resampled.join(news_features, how="left").fillna(0)
    else:
        merged = btc_resampled

    # Drop the last row because target_next_return is NaN (future unknown)
    dataset = merged.iloc[:-1].copy()

    return dataset
