"""Feature engineering pipeline for GDELT news and Bitcoin market data with non-leaky timing."""

import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from config import (
    DB_PATH,
    CATEGORIES,
    DEFAULT_INTERVAL,
    DEFAULT_PREDICTION_HORIZON,
    DEFAULT_TARGET_THRESHOLD
)


def extract_raw_dataframes(db_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load news articles and BTC prices from SQLite into pandas DataFrames."""
    target_path = db_path or str(DB_PATH)
    conn = sqlite3.connect(target_path)

    news_df = pd.read_sql_query("""
        SELECT timestamp, category, avg_tone, positive_score, negative_score,
               polarity, activity_density, word_count
        FROM news_articles
        ORDER BY timestamp ASC;
    """, conn)

    btc_df = pd.read_sql_query("""
        SELECT timestamp, open, high, low, close, volume, quote_volume, trades_count
        FROM btc_prices
        ORDER BY timestamp ASC;
    """, conn)

    conn.close()

    if not news_df.empty:
        news_df["timestamp"] = pd.to_datetime(news_df["timestamp"], utc=True)
    if not btc_df.empty:
        btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)

    return news_df, btc_df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index without lookahead bias."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / (loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 24) -> pd.Series:
    """
    Parkinson volatility estimator using high and low prices.
    More statistically efficient than close-to-close volatility.
    """
    hl_ratio = np.log(high / (low + 1e-9))
    factor = 1.0 / (4.0 * np.log(2.0))
    squared_hl = factor * (hl_ratio ** 2)
    return np.sqrt(np.maximum(0.0, squared_hl.rolling(window=window, min_periods=window).mean()))


def compute_garman_klass_volatility(
    open_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    close_s: pd.Series,
    window: int = 24
) -> pd.Series:
    """
    Garman-Klass volatility estimator using OHLC prices.
    Approximately 8x more efficient than standard close-to-close deviation.
    """
    log_hl = np.log(high_s / (low_s + 1e-9)) ** 2
    log_co = np.log(close_s / (open_s + 1e-9)) ** 2
    var_gk = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    return np.sqrt(np.maximum(0.0, var_gk.rolling(window=window, min_periods=window).mean()))


def aggregate_news_features(news_df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """
    Aggregate news sentiment, volume, and category metrics into time bins.
    Adds rolling and exponentially-smoothed signals to solve single-hour sparsity.
    Timing Rule:
      An article published at t in [T_open, T_close] belongs to candle bucket T.
      At T_close (when decisions are made for candle T+1), this news is fully known.
    """
    if news_df.empty:
        return pd.DataFrame()

    news_df = news_df.copy()
    news_df["time_bucket"] = news_df["timestamp"].dt.floor(freq)

    # 1. Global statistics per time bucket
    global_agg = news_df.groupby("time_bucket").agg(
        total_news_volume=("avg_tone", "count"),
        global_avg_tone=("avg_tone", "mean"),
        global_tone_std=("avg_tone", "std"),
        global_avg_polarity=("polarity", "mean"),
        global_avg_negativity=("negative_score", "mean"),
        global_avg_positivity=("positive_score", "mean")
    ).fillna(0.0)

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
                f"pos_{cat}": ("positive_score", "mean")
            }
        )
        cat_frames.append(agg)

    if cat_frames:
        cat_combined = pd.concat(cat_frames, axis=1).fillna(0.0)
        news_features = global_agg.join(cat_combined, how="left").fillna(0.0)
    else:
        news_features = global_agg

    # Category volume share
    for cat in CATEGORIES:
        vol_col = f"vol_{cat}"
        if vol_col in news_features.columns:
            news_features[f"share_{cat}"] = (
                news_features[vol_col] / (news_features["total_news_volume"] + 1e-6)
            )

    # 3. Rolling & Exponential Sentiment Decay Features
    news_features["news_vol_rolling_6h"] = news_features["total_news_volume"].rolling(6, min_periods=1).sum()
    news_features["news_vol_rolling_24h"] = news_features["total_news_volume"].rolling(24, min_periods=1).sum()

    # Exponential moving average of tone (24-hour half-life)
    news_features["global_tone_ema_24h"] = news_features["global_avg_tone"].ewm(span=24, min_periods=1).mean()

    # Net sentiment score: (Pos - Neg) / (Pos + Neg + eps)
    pos_24h = news_features["global_avg_positivity"].rolling(24, min_periods=1).mean()
    neg_24h = news_features["global_avg_negativity"].rolling(24, min_periods=1).mean()
    news_features["net_sentiment_24h"] = (pos_24h - neg_24h) / (pos_24h + neg_24h + 1e-5)

    # Key category rolling features
    key_categories = ["Crypto_Blockchain", "Monetary_Policy", "Regulation_Legal", "Macroeconomics"]
    for cat in key_categories:
        tone_col = f"tone_{cat}"
        vol_col = f"vol_{cat}"
        if tone_col in news_features.columns:
            news_features[f"{cat}_tone_ema_24h"] = news_features[tone_col].ewm(span=24, min_periods=1).mean()
        if vol_col in news_features.columns:
            news_features[f"{cat}_vol_rolling_24h"] = news_features[vol_col].rolling(24, min_periods=1).sum()

    # Binary indicator of news presence in recent window
    news_features["has_news_6h"] = (news_features["news_vol_rolling_6h"] > 0).astype(float)

    return news_features


def compute_all_market_features(btc_resampled: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive, multi-timeframe non-leaky technical indicators.
    All indicators at index t use strictly data from candle t and prior closed candles.
    """
    df = btc_resampled.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    op = df["open"]
    vol = df["volume"]

    # 1. Multi-timeframe Returns
    for lag in [1, 2, 4, 8, 12, 24, 48]:
        df[f"btc_return_{lag}h"] = np.log(close / (close.shift(lag) + 1e-9))

    # 2. Moving Average Ratios & Trend
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    df["btc_ema_ratio_21"] = (close / (ema_21 + 1e-9)) - 1.0
    df["btc_ema_ratio_50"] = (close / (ema_50 + 1e-9)) - 1.0
    df["btc_ema_cross_9_21"] = (ema_9 / (ema_21 + 1e-9)) - 1.0

    # 3. MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["btc_macd"] = macd
    df["btc_macd_signal"] = macd_signal
    df["btc_macd_hist"] = macd - macd_signal

    # 4. Volatility Estimators
    df["btc_volatility_6h"] = df["btc_return_1h"].rolling(6, min_periods=6).std()
    df["btc_volatility_24h"] = df["btc_return_1h"].rolling(24, min_periods=24).std()
    df["btc_parkinson_vol_24h"] = compute_parkinson_volatility(high, low, window=24)
    df["btc_garman_klass_vol_24h"] = compute_garman_klass_volatility(op, high, low, close, window=24)
    df["btc_hl_spread"] = (high - low) / (op + 1e-9)

    # Average True Range (Normalized)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["btc_natr_14"] = (tr.rolling(14, min_periods=14).mean() / (close + 1e-9)) * 100.0

    # 5. Oscillators: RSI, Stochastic RSI, Bollinger %b
    df["btc_rsi_14"] = compute_rsi(close, period=14)
    rsi = df["btc_rsi_14"]
    rsi_min = rsi.rolling(14, min_periods=14).min()
    rsi_max = rsi.rolling(14, min_periods=14).max()
    df["btc_stoch_rsi_14"] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-9)

    sma_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.rolling(20, min_periods=20).std()
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    df["btc_bb_pct_b_20"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df["btc_bb_width_20"] = (bb_upper - bb_lower) / (sma_20 + 1e-9)

    # 6. Volume & Liquidity Dynamics
    vol_sma_24 = vol.rolling(24, min_periods=1).mean()
    df["btc_volume_ratio_24h"] = vol / (vol_sma_24 + 1e-6)
    df["btc_volume_change_1h"] = vol.pct_change(1).fillna(0.0)

    # 7. Cyclical Time Features (capturing market session liquidity)
    hours = df.index.hour
    day_of_week = df.index.dayofweek if hasattr(df.index, "dayofweek") else df.index.day_of_week
    df["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    df["day_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    df["day_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

    return df


def build_ml_dataset(
    freq: str = DEFAULT_INTERVAL,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
    target_threshold: float = DEFAULT_TARGET_THRESHOLD,
    overlap_only: bool = False,
    db_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct non-leaky aligned dataset for model training and evaluation:
    - Target: return from Open of candle t+1 to Close of candle t+horizon (realistic trade execution).
    - Features: purely based on information available at Close of candle t.
    - Eliminates lookahead bias.
    """
    news_df, btc_df = extract_raw_dataframes(db_path)

    if btc_df.empty:
        raise ValueError("BTC price table is empty. Fetch BTC data first.")

    # 1. Clean and resample BTC prices
    btc_df = btc_df.set_index("timestamp").sort_index()
    btc_df = btc_df[~btc_df.index.duplicated(keep="first")]

    btc_resampled = btc_df.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trades_count": "sum"
    }).dropna()

    # 2. Compute rich market features
    btc_features = compute_all_market_features(btc_resampled)

    # 3. Explicit Target Variable Construction:
    # Decision is at t close. Execution enters at t+1 Open, exits at t+horizon Close.
    future_exit_close = btc_features["close"].shift(-prediction_horizon)
    next_entry_open = btc_features["open"].shift(-1)
    
    btc_features["target_future_return"] = (future_exit_close - next_entry_open) / (next_entry_open + 1e-9)
    btc_features["target_direction"] = (btc_features["target_future_return"] > target_threshold).astype(int)
    btc_features["target_close_to_close"] = (future_exit_close - btc_features["close"]) / (btc_features["close"] + 1e-9)

    # 4. Merge News Features
    news_metadata = {
        "news_articles_in_db": len(news_df),
        "news_min_date": str(news_df["timestamp"].min()) if not news_df.empty else None,
        "news_max_date": str(news_df["timestamp"].max()) if not news_df.empty else None,
    }

    if not news_df.empty:
        news_features = aggregate_news_features(news_df, freq=freq)
        merged = btc_features.join(news_features, how="left").fillna(0.0)
    else:
        merged = btc_features

    # 5. Overlap filtering if requested
    if overlap_only and not news_df.empty:
        earliest_news = news_df["timestamp"].min().floor(freq)
        latest_news = news_df["timestamp"].max().ceil(freq)
        merged = merged[(merged.index >= earliest_news) & (merged.index <= latest_news)].copy()

    # Drop the trailing rows where target is unknown due to forward shift
    dataset = merged.iloc[:-prediction_horizon].copy()

    # Drop the initial warm-up period rows with NaNs from rolling indicators
    dataset = dataset.dropna(subset=["btc_volatility_24h", "btc_rsi_14", "btc_natr_14", "target_future_return"]).copy()

    metadata = {
        "total_samples": len(dataset),
        "prediction_horizon": prediction_horizon,
        "target_threshold": target_threshold,
        "overlap_only": overlap_only,
        "date_start": str(dataset.index.min()),
        "date_end": str(dataset.index.max()),
        **news_metadata
    }

    return dataset, metadata


def diagnose_feature_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive diagnostic table of feature coverage, missing rates, and non-zero rates.
    """
    excluded_cols = ["target_future_return", "target_direction", "target_close_to_close"]
    feature_cols = [c for c in df.columns if c not in excluded_cols]

    report = []
    total_rows = len(df)

    for col in feature_cols:
        series = df[col]
        non_zero_cnt = int((series != 0).sum())
        non_zero_pct = (non_zero_cnt / total_rows) * 100.0 if total_rows > 0 else 0.0
        nan_cnt = int(series.isna().sum())
        nan_pct = (nan_cnt / total_rows) * 100.0 if total_rows > 0 else 0.0
        std_val = float(series.std()) if total_rows > 1 else 0.0

        is_news = (not col.startswith("btc_")) and any(k in col for k in ["news", "tone", "vol_", "share_", "neg_", "pos_", "has_news"])

        report.append({
            "Feature": col,
            "Type": "News" if is_news else "Market",
            "Non_Zero_Count": non_zero_cnt,
            "Non_Zero_Pct": round(non_zero_pct, 2),
            "NaN_Count": nan_cnt,
            "NaN_Pct": round(nan_pct, 2),
            "Std_Dev": round(std_val, 4)
        })

    report_df = pd.DataFrame(report).sort_values(by=["Type", "Non_Zero_Pct"], ascending=[True, False])
    return report_df


def extract_x_period_features_for_window(
    btc_df: pd.DataFrame,
    news_hourly: pd.DataFrame,
    idx_start: int,
    idx_mid: int,
    idx_prev_start: int,
    idx_prev_end: int,
    include_targets: bool = True,
    idx_end: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extract features strictly for an X-day period where:
    - Window 1: First X/2 days of this X days [idx_start, idx_mid]
    - Window 2: X/2 days of previous week [idx_prev_start, idx_prev_end]
    - Decision time: idx_mid (Close of candle idx_mid)
    - Target: Close(idx_end) - Open(idx_start)
    """
    p_start = float(btc_df['open'].iloc[idx_start])
    p_curr_close = float(btc_df['close'].iloc[idx_mid])

    # Window 1: First X/2 days of this X days
    slice_curr = btc_df.iloc[idx_start:idx_mid+1]
    p_curr_open = p_start
    ret_curr = (p_curr_close - p_curr_open) / (p_curr_open + 1e-9)
    high_curr = float(slice_curr['high'].max())
    low_curr = float(slice_curr['low'].min())
    range_curr = (high_curr - low_curr) / (p_curr_open + 1e-9)
    pos_in_range_curr = (p_curr_close - low_curr) / (high_curr - low_curr + 1e-9)
    dir_efficiency_curr = (p_curr_close - p_curr_open) / (high_curr - low_curr + 1e-9)
    vol_curr = float(slice_curr['volume'].sum())
    
    ret_1h_curr = slice_curr['ret_1h'] if 'ret_1h' in slice_curr.columns else np.log(slice_curr['close'] / slice_curr['close'].shift(1))
    std_curr = float(ret_1h_curr.std()) if len(slice_curr) > 1 else 0.0

    # Intra-window momentum (Quarter split of Window 1)
    idx_q1 = idx_start + (idx_mid - idx_start) // 2
    p_q1_close = float(btc_df['close'].iloc[idx_q1])
    ret_q1 = (p_q1_close - p_curr_open) / (p_curr_open + 1e-9)
    ret_q2 = (p_curr_close - p_q1_close) / (p_q1_close + 1e-9)
    accel_curr = ret_q2 - ret_q1
    vol_q1 = float(btc_df['volume'].iloc[idx_start:idx_q1+1].sum())
    vol_q2 = float(btc_df['volume'].iloc[idx_q1:idx_mid+1].sum())
    vol_trend_curr = (vol_q2 + 1.0) / (vol_q1 + 1.0)

    # VWAP Window 1
    pv_curr = float((slice_curr['close'] * slice_curr['volume']).sum())
    vwap_curr = pv_curr / (vol_curr + 1e-6)
    vwap_ratio_curr = (p_curr_close / (vwap_curr + 1e-9)) - 1.0

    # Window 2: X/2 days of previous week
    slice_prev = btc_df.iloc[idx_prev_start:idx_prev_end+1]
    p_prev_open = float(btc_df['open'].iloc[idx_prev_start])
    p_prev_close = float(btc_df['close'].iloc[idx_prev_end])
    ret_prev = (p_prev_close - p_prev_open) / (p_prev_open + 1e-9)
    high_prev = float(slice_prev['high'].max())
    low_prev = float(slice_prev['low'].min())
    range_prev = (high_prev - low_prev) / (p_prev_open + 1e-9)
    pos_in_range_prev = (p_prev_close - low_prev) / (high_prev - low_prev + 1e-9)
    dir_efficiency_prev = (p_prev_close - p_prev_open) / (high_prev - low_prev + 1e-9)
    vol_prev = float(slice_prev['volume'].sum())
    
    ret_1h_prev = slice_prev['ret_1h'] if 'ret_1h' in slice_prev.columns else np.log(slice_prev['close'] / slice_prev['close'].shift(1))
    std_prev = float(ret_1h_prev.std()) if len(slice_prev) > 1 else 0.0

    idx_prev_q1 = idx_prev_start + (idx_prev_end - idx_prev_start) // 2
    p_prev_q1 = float(btc_df['close'].iloc[idx_prev_q1])
    ret_prev_q1 = (p_prev_q1 - p_prev_open) / (p_prev_open + 1e-9)
    ret_prev_q2 = (p_prev_close - p_prev_q1) / (p_prev_q1 + 1e-9)
    accel_prev = ret_prev_q2 - ret_prev_q1

    # Comparative & Seasonal Differentials
    ret_diff = ret_curr - ret_prev
    accel_diff = accel_curr - accel_prev
    vol_ratio = (vol_curr + 1.0) / (vol_prev + 1.0)
    std_ratio = (std_curr + 1e-6) / (std_prev + 1e-6)
    range_ratio = (range_curr + 1e-6) / (range_prev + 1e-6)
    level_ratio_1w = (p_curr_close - p_prev_close) / (p_prev_close + 1e-9)
    level_ratio_1w_start = (p_curr_close - p_prev_open) / (p_prev_open + 1e-9)

    # Technical Indicators at decision time (idx_mid)
    rsi_now = float(btc_df['rsi_14'].iloc[idx_mid]) if 'rsi_14' in btc_df.columns else 50.0
    macd_now = float(btc_df['macd'].iloc[idx_mid]) if 'macd' in btc_df.columns else 0.0
    macd_hist_now = float(btc_df['macd_hist'].iloc[idx_mid]) if 'macd_hist' in btc_df.columns else 0.0
    ema_ratio_21 = float(btc_df['ema_ratio_21'].iloc[idx_mid]) if 'ema_ratio_21' in btc_df.columns else 0.0
    ema_ratio_50 = float(btc_df['ema_ratio_50'].iloc[idx_mid]) if 'ema_ratio_50' in btc_df.columns else 0.0
    ema_ratio_200 = float(btc_df['ema_ratio_200'].iloc[idx_mid]) if 'ema_ratio_200' in btc_df.columns else 0.0
    bb_pct_b = float(btc_df['bb_pct_b'].iloc[idx_mid]) if 'bb_pct_b' in btc_df.columns else 0.5
    natr_14 = float(btc_df['btc_natr_14'].iloc[idx_mid]) if 'btc_natr_14' in btc_df.columns else 1.5

    # News features in Window 1 and Window 2
    t_start = btc_df.index[idx_start]
    t_curr_mid = btc_df.index[idx_mid]
    t_prev_start = btc_df.index[idx_prev_start]
    t_prev_end = btc_df.index[idx_prev_end]

    news_vol_w1 = 0
    news_vol_w2 = 0
    news_tone_w1 = 0.0
    news_tone_w2 = 0.0
    news_pos_w1 = 0.0
    news_neg_w1 = 0.0

    if not news_hourly.empty:
        news_w1 = news_hourly.loc[(news_hourly.index >= t_start) & (news_hourly.index <= t_curr_mid)]
        news_w2 = news_hourly.loc[(news_hourly.index >= t_prev_start) & (news_hourly.index <= t_prev_end)]
        if not news_w1.empty:
            news_vol_w1 = int(news_w1['n_count'].sum()) if 'n_count' in news_w1 else len(news_w1)
            news_tone_w1 = float(news_w1['avg_tone'].mean()) if 'avg_tone' in news_w1 else 0.0
            news_pos_w1 = float(news_w1['pos'].mean()) if 'pos' in news_w1 else 0.0
            news_neg_w1 = float(news_w1['neg'].mean()) if 'neg' in news_w1 else 0.0
        if not news_w2.empty:
            news_vol_w2 = int(news_w2['n_count'].sum()) if 'n_count' in news_w2 else len(news_w2)
            news_tone_w2 = float(news_w2['avg_tone'].mean()) if 'avg_tone' in news_w2 else 0.0

    news_tone_diff = news_tone_w1 - news_tone_w2
    news_vol_ratio = (news_vol_w1 + 1.0) / (news_vol_w2 + 1.0)
    has_news_w1 = 1.0 if news_vol_w1 > 0 else 0.0

    feat_dict = {
        'timestamp': t_curr_mid,
        'p_start': p_start,
        'p_curr': p_curr_close,
        # Window 1 features
        'ret_curr': ret_curr,
        'ret_q1': ret_q1,
        'ret_q2': ret_q2,
        'accel_curr': accel_curr,
        'range_curr': range_curr,
        'pos_in_range_curr': pos_in_range_curr,
        'dir_efficiency_curr': dir_efficiency_curr,
        'vol_curr': np.log1p(vol_curr),
        'vol_trend_curr': vol_trend_curr,
        'vwap_ratio_curr': vwap_ratio_curr,
        'std_curr': std_curr if not np.isnan(std_curr) else 0.0,
        # Window 2 features
        'ret_prev': ret_prev,
        'ret_prev_q1': ret_prev_q1,
        'ret_prev_q2': ret_prev_q2,
        'accel_prev': accel_prev,
        'range_prev': range_prev,
        'pos_in_range_prev': pos_in_range_prev,
        'dir_efficiency_prev': dir_efficiency_prev,
        'vol_prev': np.log1p(vol_prev),
        'std_prev': std_prev if not np.isnan(std_prev) else 0.0,
        # Comparative features
        'ret_diff': ret_diff,
        'accel_diff': accel_diff,
        'vol_ratio': vol_ratio,
        'std_ratio': std_ratio,
        'range_ratio': range_ratio,
        'level_ratio_1w': level_ratio_1w,
        'level_ratio_1w_start': level_ratio_1w_start,
        # Technical indicators
        'rsi_now': rsi_now if not np.isnan(rsi_now) else 50.0,
        'macd_now': macd_now if not np.isnan(macd_now) else 0.0,
        'macd_hist_now': macd_hist_now if not np.isnan(macd_hist_now) else 0.0,
        'ema_ratio_21': ema_ratio_21 if not np.isnan(ema_ratio_21) else 0.0,
        'ema_ratio_50': ema_ratio_50 if not np.isnan(ema_ratio_50) else 0.0,
        'ema_ratio_200': ema_ratio_200 if not np.isnan(ema_ratio_200) else 0.0,
        'bb_pct_b': bb_pct_b if not np.isnan(bb_pct_b) else 0.5,
        'natr_14': natr_14 if not np.isnan(natr_14) else 1.5,
        # News features
        'news_vol_w1': news_vol_w1,
        'news_vol_w2': news_vol_w2,
        'news_tone_w1': news_tone_w1 if not np.isnan(news_tone_w1) else 0.0,
        'news_tone_w2': news_tone_w2 if not np.isnan(news_tone_w2) else 0.0,
        'news_tone_diff': news_tone_diff if not np.isnan(news_tone_diff) else 0.0,
        'news_vol_ratio': news_vol_ratio,
        'news_pos_w1': news_pos_w1 if not np.isnan(news_pos_w1) else 0.0,
        'news_neg_w1': news_neg_w1 if not np.isnan(news_neg_w1) else 0.0,
        'has_news_w1': has_news_w1
    }

    if include_targets and idx_end is not None:
        p_end = float(btc_df['close'].iloc[idx_end])
        delta_p = p_end - p_start
        delta_p_pct = delta_p / (p_start + 1e-9)
        target_dir = 1 if delta_p > 0 else 0
        feat_dict.update({
            'p_end': p_end,
            'delta_p': delta_p,
            'delta_p_pct': delta_p_pct,
            'target_dir': target_dir
        })

    return feat_dict


def build_x_period_dataset(
    x_days: float = 4.0,
    step_hours: int = 4,
    db_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct non-leaky dataset for X-day period analysis:
    - Period duration: X days = round(X * 24) hours
    - First half: X/2 days = round(X * 12) hours
    - Previous week half: [T_start - 168h, T_start - 168h + round(X * 12)h]
    - Target: Price(end of X days) - Price(start of X day) = Close(T_end) - Open(T_start)
    - Prediction time: Close of candle at T_start + round(X * 12)h
    """
    news_df, btc_df = extract_raw_dataframes(db_path)

    if btc_df.empty:
        raise ValueError("BTC price table is empty. Fetch BTC data first.")

    btc_df = btc_df.set_index("timestamp").sort_index()
    btc_df = btc_df[~btc_df.index.duplicated(keep="first")]

    H_total = int(round(x_days * 24))
    H_half = int(round(x_days * 12))
    W_offset = 168  # 7 days in hours

    # 1. Precompute technical indicators across all candles
    close = btc_df['close']
    high = btc_df['high']
    low = btc_df['low']
    op = btc_df['open']

    btc_df['ret_1h'] = np.log(close / close.shift(1))
    btc_df['rsi_14'] = compute_rsi(close, period=14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    btc_df['macd'] = ema_12 - ema_26
    btc_df['macd_sig'] = btc_df['macd'].ewm(span=9, adjust=False).mean()
    btc_df['macd_hist'] = btc_df['macd'] - btc_df['macd_sig']

    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()
    btc_df['ema_ratio_21'] = (close / ema_21) - 1.0
    btc_df['ema_ratio_50'] = (close / ema_50) - 1.0
    btc_df['ema_ratio_200'] = (close / ema_200) - 1.0

    sma_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.rolling(20, min_periods=20).std()
    btc_df['bb_pct_b'] = (close - (sma_20 - 2.0 * std_20)) / (4.0 * std_20 + 1e-9)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    btc_df['btc_natr_14'] = (tr.rolling(14, min_periods=14).mean() / (close + 1e-9)) * 100.0

    # 2. Pre-index news by hourly buckets
    if not news_df.empty:
        news_df['hour'] = news_df['timestamp'].dt.floor('1h')
        news_hourly = news_df.groupby('hour').agg(
            n_count=('avg_tone', 'count'),
            avg_tone=('avg_tone', 'mean'),
            pos=('positive_score', 'mean'),
            neg=('negative_score', 'mean'),
            pol=('polarity', 'mean')
        ).fillna(0.0)
    else:
        news_hourly = pd.DataFrame()

    # 3. Slide through candles and extract period records
    times = btc_df.index
    time_to_idx = {t: idx for idx, t in enumerate(times)}
    records = []

    for i in range(0, len(btc_df), step_hours):
        t_start = times[i]
        t_prev_start = t_start - pd.Timedelta(hours=W_offset)
        t_prev_end = t_prev_start + pd.Timedelta(hours=H_half)
        t_curr_mid = t_start + pd.Timedelta(hours=H_half)
        t_curr_end = t_start + pd.Timedelta(hours=H_total)

        if t_prev_start not in time_to_idx or t_curr_end not in time_to_idx:
            continue
        if t_prev_end not in time_to_idx or t_curr_mid not in time_to_idx:
            continue

        idx_prev_start = time_to_idx[t_prev_start]
        idx_prev_end = time_to_idx[t_prev_end]
        idx_start = i
        idx_mid = time_to_idx[t_curr_mid]
        idx_end = time_to_idx[t_curr_end]

        row = extract_x_period_features_for_window(
            btc_df=btc_df,
            news_hourly=news_hourly,
            idx_start=idx_start,
            idx_mid=idx_mid,
            idx_prev_start=idx_prev_start,
            idx_prev_end=idx_prev_end,
            include_targets=True,
            idx_end=idx_end
        )
        records.append(row)

    dataset = pd.DataFrame(records).set_index('timestamp')

    metadata = {
        "x_days": x_days,
        "total_period_hours": H_total,
        "half_period_hours": H_half,
        "prev_week_offset_hours": W_offset,
        "step_hours": step_hours,
        "total_samples": len(dataset),
        "date_start": str(dataset.index.min()) if not dataset.empty else None,
        "date_end": str(dataset.index.max()) if not dataset.empty else None,
        "up_ratio": float(dataset['target_dir'].mean()) if not dataset.empty else 0.0
    }

    return dataset, metadata
