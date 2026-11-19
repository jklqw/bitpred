"""Real-time Bitcoin Decision & Signal Generation Engine: Recommends BUY, SELL, or WAIT."""

import os
import sys
from pathlib import Path

# Auto-bootstrap virtualenv site-packages if running with system python
BASE_DIR = Path(__file__).resolve().parent
_site_packages = BASE_DIR / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _site_packages.is_dir() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from config import (
    DEFAULT_INTERVAL,
    DEFAULT_PREDICTION_HORIZON,
    DEFAULT_TAKER_FEE,
    DEFAULT_SLIPPAGE
)
from btc_fetcher import fetch_binance_btc_klines_chunk
from features import (
    extract_raw_dataframes,
    compute_all_market_features,
    aggregate_news_features,
    compute_rsi,
    extract_x_period_features_for_window
)
from train import (
    MODEL_ARTIFACT_PATH,
    train_and_evaluate_pipeline,
    get_x_period_artifact_path,
    train_x_period_pipeline
)


def fetch_latest_market_candles(limit: int = 150) -> pd.DataFrame:
    """Fetch the latest live candles from Binance up to the current moment."""
    raw = fetch_binance_btc_klines_chunk(interval="1h", limit=limit)
    if not raw:
        raise RuntimeError("Failed to fetch real-time klines from Binance API.")

    records = []
    for item in raw:
        dt = datetime.fromtimestamp(item[0] / 1000.0, tz=timezone.utc)
        records.append({
            "timestamp": dt,
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
            "quote_volume": float(item[7]),
            "trades_count": int(item[8])
        })

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    return df


def generate_live_signal(prediction_horizon: int = 4) -> Dict[str, Any]:
    """
    Run end-to-end real-time inference:
    1. Fetch live candles & news data
    2. Engineer full non-leaky feature set
    3. Load or train best ensemble model
    4. Produce actionable BUY / SELL / NEUTRAL decision with levels
    """
    # 1. Fetch live candles
    live_btc = fetch_latest_market_candles(limit=150)
    current_candle = live_btc.iloc[-1]
    last_closed_candle = live_btc.iloc[-2]  # Most recently completed 1h candle
    current_price = float(current_candle["close"])
    current_time_utc = live_btc.index[-1]

    # 2. Compute market features up to the last closed candle
    market_features = compute_all_market_features(live_btc)

    # 3. Aggregate news features
    news_df, _ = extract_raw_dataframes()
    if not news_df.empty:
        news_features = aggregate_news_features(news_df, freq="1h")
        merged = market_features.join(news_features, how="left").fillna(0.0)
    else:
        merged = market_features

    # 4. Load trained model artifact (or train if missing)
    if not os.path.exists(MODEL_ARTIFACT_PATH):
        print("[*] Model artifact not found. Training production ensemble model now...")
        train_and_evaluate_pipeline(prediction_horizon=prediction_horizon)

    artifact = joblib.load(MODEL_ARTIFACT_PATH)
    scaler = artifact["scaler"]
    ensemble = artifact["ensemble"]
    feature_cols = artifact["feature_cols"]

    # 5. Extract latest feature row (from last closed candle to avoid unfinished bar noise)
    target_row = merged.iloc[[-2]].copy()
    # Ensure all expected feature columns exist
    for col in feature_cols:
        if col not in target_row.columns:
            target_row[col] = 0.0

    X_latest = target_row[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_latest_scaled = scaler.transform(X_latest)

    # 6. Predict probabilities
    prob_up = float(ensemble.predict_proba(X_latest_scaled)[0, 1])
    prob_down = 1.0 - prob_up

    # 7. Formulate Decision & Confidence
    if prob_up >= 0.58:
        action = "STRONG BUY / LONG"
        color = "🟢"
        bias = "Bullish"
        confidence = "High"
    elif prob_up >= 0.53:
        action = "BUY / ACCUMULATE"
        color = "🟢"
        bias = "Moderately Bullish"
        confidence = "Moderate"
    elif prob_up <= 0.42:
        action = "STRONG SELL / SHORT"
        color = "🔴"
        bias = "Bearish"
        confidence = "High"
    elif prob_up <= 0.47:
        action = "SELL / CASH / REDUCE"
        color = "🔴"
        bias = "Moderately Bearish"
        confidence = "Moderate"
    else:
        action = "NEUTRAL / HOLD / CASH"
        color = "⚪"
        bias = "Neutral / Indecisive"
        confidence = "Low (Sideways Chop)"

    # 8. Calculate Technical Risk Management Levels
    natr = float(target_row["btc_natr_14"].iloc[0]) if "btc_natr_14" in target_row else 1.5
    atr_dollars = current_price * (natr / 100.0)

    if "BUY" in action:
        stop_loss = current_price - (1.5 * atr_dollars)
        take_profit_1 = current_price + (2.0 * atr_dollars)
        take_profit_2 = current_price + (3.5 * atr_dollars)
    elif "SELL" in action:
        stop_loss = current_price + (1.5 * atr_dollars)
        take_profit_1 = current_price - (2.0 * atr_dollars)
        take_profit_2 = current_price - (3.5 * atr_dollars)
    else:
        stop_loss = current_price - (1.5 * atr_dollars)
        take_profit_1 = current_price + (1.5 * atr_dollars)
        take_profit_2 = current_price + (3.0 * atr_dollars)

    # Key Technical Indicators summary
    rsi_val = float(target_row["btc_rsi_14"].iloc[0]) if "btc_rsi_14" in target_row else 50.0
    macd_val = float(target_row["btc_macd"].iloc[0]) if "btc_macd" in target_row else 0.0
    macd_sig = float(target_row["btc_macd_signal"].iloc[0]) if "btc_macd_signal" in target_row else 0.0
    bb_pct_b = float(target_row["btc_bb_pct_b_20"].iloc[0]) if "btc_bb_pct_b_20" in target_row else 0.5
    ret_24h = float(target_row["btc_return_24h"].iloc[0]) * 100.0 if "btc_return_24h" in target_row else 0.0
    news_tone = float(target_row["global_tone_ema_24h"].iloc[0]) if "global_tone_ema_24h" in target_row else 0.0

    return {
        "timestamp_utc": str(current_time_utc),
        "current_price": current_price,
        "action": action,
        "action_icon": color,
        "bias": bias,
        "confidence": confidence,
        "prob_up": round(prob_up * 100.0, 1),
        "prob_down": round(prob_down * 100.0, 1),
        "horizon_hours": prediction_horizon,
        "current_levels": {
            "current_price": round(current_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit_1": round(take_profit_1, 2),
            "take_profit_2": round(take_profit_2, 2),
            "atr_1h": round(atr_dollars, 2),
            "atr_pct": round(natr, 2)
        },
        "indicators": {
            "rsi_14": round(rsi_val, 1),
            "macd": round(macd_val, 2),
            "macd_signal": round(macd_sig, 2),
            "bollinger_pct_b": round(bb_pct_b, 2),
            "change_24h_pct": round(ret_24h, 2),
            "news_tone_ema": round(news_tone, 2),
            "news_articles_db": len(news_df)
        }
    }


def print_signal_dashboard(signal_data: Dict[str, Any]):
    """Print an attractive, terminal decision dashboard."""
    s = signal_data
    lv = s["current_levels"]
    ind = s["indicators"]

    print("\n" + "╔" + "═" * 70 + "╗")
    print(f"║          BITCOIN REAL-TIME ACTION RECOMMENDATION                     ║")
    print("╠" + "═" * 70 + "╣")
    print(f"║  Current Time (UTC) : {s['timestamp_utc']:<45} ║")
    print(f"║  Current BTC Price  : ${lv['current_price']:<12,.2f}  (24h Change: {ind['change_24h_pct']:+.2f}%)       ║")
    print("╠" + "═" * 70 + "╣")
    print(f"║  ACTION TO TAKE NOW : {s['action_icon']} {s['action']:<43} ║")
    print(f"║  Market Bias        : {s['bias']:<45} ║")
    print(f"║  Model Confidence   : {s['confidence']:<45} ║")
    print(f"║  Forecast Horizon   : {s['horizon_hours']} Hours Forward Outlook                          ║")
    print(f"║  Upward Probability : {s['prob_up']}%                                           ║")
    print(f"║  Downward Risk      : {s['prob_down']}%                                           ║")
    print("╠" + "═" * 70 + "╣")
    print(f"║                      RISK MANAGEMENT LEVELS                          ║")
    print("╠" + "═" * 70 + "╣")
    print(f"║  Suggested Entry    : ${lv['current_price']:<12,.2f}                                    ║")
    print(f"║  Stop Loss (Risk)   : ${lv['stop_loss']:<12,.2f}  (1.5x ATR Risk Cushion)           ║")
    print(f"║  Take Profit Target : ${lv['take_profit_1']:<12,.2f}  (Primary Target 2.0x ATR)         ║")
    print(f"║  Extended Target    : ${lv['take_profit_2']:<12,.2f}  (Runner Target 3.5x ATR)          ║")
    print(f"║  Hourly Volatility  : ${lv['atr_1h']:<8,.2f} ({lv['atr_pct']:.2f}% ATR)                            ║")
    print("╠" + "═" * 70 + "╣")
    print(f"║                     KEY CONFLUENCE SIGNALS                           ║")
    print("╠" + "═" * 70 + "╣")

    # Interpret RSI
    if ind['rsi_14'] < 30:
        rsi_state = "Oversold (Bounce Imminent)"
    elif ind['rsi_14'] > 70:
        rsi_state = "Overbought (Pullback Risk)"
    else:
        rsi_state = "Neutral"

    # Interpret MACD
    macd_state = "Bullish Momentum" if ind['macd'] > ind['macd_signal'] else "Bearish Momentum"

    # Interpret Bollinger
    if ind['bollinger_pct_b'] < 0.15:
        bb_state = "At Lower Band Support"
    elif ind['bollinger_pct_b'] > 0.85:
        bb_state = "At Upper Band Resistance"
    else:
        bb_state = "Mid-Band Channel"

    print(f"║  • RSI (14)         : {ind['rsi_14']:<5.1f} -> {rsi_state:<34} ║")
    print(f"║  • MACD vs Signal   : {ind['macd']:+7.1f} / {ind['macd_signal']:+7.1f} -> {macd_state:<24} ║")
    print(f"║  • Bollinger Band   : %B = {ind['bollinger_pct_b']:<4.2f} -> {bb_state:<33} ║")
    print(f"║  • GDELT News Tone  : {ind['news_tone_ema']:+5.2f} ({ind['news_articles_db']} articles in database)             ║")
    print("╚" + "═" * 70 + "╝\n")


def generate_x_period_action(
    x_days: float = 4.0,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Real-time inference for X-day period prediction:
    1. Fetches live market candles to cover both current X/2 window and previous week X/2 window.
    2. Builds non-leaky feature set comparing first X/2 days of current period with X/2 days of previous week.
    3. Loads or automatically trains X-day machine learning ensemble and regressor.
    4. Predicts expected Delta Price: Price(end of X days) - Price(start of X day).
    5. Returns actionable decision: BUY / LONG or SELL / SHORT at this time.
    """
    H_total = int(round(x_days * 24))
    H_half = int(round(x_days * 12))
    W_offset = 168  # 7 days in hours

    candles_needed = min(1000, max(500, W_offset + H_half + 250))
    raw = fetch_binance_btc_klines_chunk(interval="1h", limit=candles_needed)
    if not raw:
        raise RuntimeError("Failed to retrieve live market data from Binance API.")

    records = []
    for item in raw:
        dt = datetime.fromtimestamp(item[0] / 1000.0, tz=timezone.utc)
        records.append({
            "timestamp": dt,
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
            "quote_volume": float(item[7]),
            "trades_count": int(item[8])
        })
    df_live = pd.DataFrame(records).set_index("timestamp").sort_index()

    # Precompute technical indicators
    close = df_live['close']
    high = df_live['high']
    low = df_live['low']
    op = df_live['open']

    df_live['ret_1h'] = np.log(close / close.shift(1))
    df_live['rsi_14'] = compute_rsi(close, period=14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df_live['macd'] = ema_12 - ema_26
    df_live['macd_sig'] = df_live['macd'].ewm(span=9, adjust=False).mean()
    df_live['macd_hist'] = df_live['macd'] - df_live['macd_sig']

    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()
    df_live['ema_ratio_21'] = (close / ema_21) - 1.0
    df_live['ema_ratio_50'] = (close / ema_50) - 1.0
    df_live['ema_ratio_200'] = (close / ema_200) - 1.0

    sma_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.rolling(20, min_periods=20).std()
    df_live['bb_pct_b'] = (close - (sma_20 - 2.0 * std_20)) / (4.0 * std_20 + 1e-9)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_live['btc_natr_14'] = (tr.rolling(14, min_periods=14).mean() / (close + 1e-9)) * 100.0

    # Load news from DB
    news_df, _ = extract_raw_dataframes(db_path)
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

    # Align decision time: last completed 1h candle to avoid incomplete candle noise
    idx_mid = len(df_live) - 2
    t_curr_mid = df_live.index[idx_mid]
    current_live_price = float(df_live['close'].iloc[-1])

    t_start = t_curr_mid - pd.Timedelta(hours=H_half)
    t_prev_start = t_start - pd.Timedelta(hours=W_offset)
    t_prev_end = t_prev_start + pd.Timedelta(hours=H_half)

    # Resolve nearest indices
    idx_start = df_live.index.get_indexer([t_start], method="nearest")[0]
    idx_prev_start = df_live.index.get_indexer([t_prev_start], method="nearest")[0]
    idx_prev_end = df_live.index.get_indexer([t_prev_end], method="nearest")[0]

    feature_dict = extract_x_period_features_for_window(
        btc_df=df_live,
        news_hourly=news_hourly,
        idx_start=idx_start,
        idx_mid=idx_mid,
        idx_prev_start=idx_prev_start,
        idx_prev_end=idx_prev_end,
        include_targets=False
    )

    # Load or train production model artifact
    artifact_path = get_x_period_artifact_path(x_days)
    if not os.path.exists(artifact_path):
        print(f"[*] Model artifact for X={x_days} days not found. Training model now...")
        train_x_period_pipeline(x_days=x_days, db_path=db_path)

    artifact = joblib.load(artifact_path)
    scaler = artifact["scaler"]
    ensemble = artifact["ensemble"]
    regressor = artifact["regressor"]
    feature_cols = artifact["feature_cols"]
    threshold = artifact.get("optimal_threshold", 0.50)

    # Vectorize and scale
    X_live = pd.DataFrame([feature_dict])[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_live_scaled = scaler.transform(X_live)

    prob_up = float(ensemble.predict_proba(X_live_scaled)[0, 1])
    prob_down = 1.0 - prob_up
    pred_delta_pct = float(regressor.predict(X_live_scaled)[0])

    p_start = feature_dict['p_start']
    p_decision = feature_dict['p_curr']
    pred_delta_dollars = p_start * pred_delta_pct
    expected_target_price = p_start + pred_delta_dollars
    remaining_delta_dollars = expected_target_price - current_live_price
    remaining_delta_pct = (remaining_delta_dollars / current_live_price) * 100.0

    interim_delta_dollars = current_live_price - p_start
    interim_delta_pct = (interim_delta_dollars / p_start) * 100.0

    # Decision Formulation: Direction is dictated by dominant calibrated ensemble probability
    if prob_up >= 0.50:
        action = "BUY / LONG"
        action_icon = "🟢"
        bias = "Bullish Target Expected"
        confidence = "High Conviction" if prob_up >= max(0.55, threshold) else "Moderate Conviction"
        if expected_target_price <= current_live_price:
            expected_target_price = current_live_price + abs(pred_delta_dollars if abs(pred_delta_dollars) > 100 else current_live_price * 0.015)
        remaining_delta_dollars = expected_target_price - current_live_price
        remaining_delta_pct = (remaining_delta_dollars / current_live_price) * 100.0
    else:
        action = "SELL / SHORT"
        action_icon = "🔴"
        bias = "Bearish Target Expected"
        confidence = "High Conviction" if prob_down >= 0.55 else "Moderate Conviction"
        if expected_target_price >= current_live_price:
            expected_target_price = current_live_price - abs(pred_delta_dollars if abs(pred_delta_dollars) > 100 else current_live_price * 0.015)
        remaining_delta_dollars = expected_target_price - current_live_price
        remaining_delta_pct = (remaining_delta_dollars / current_live_price) * 100.0

    # Risk Management Levels
    natr = feature_dict.get('natr_14', 1.5)
    atr_dollars = current_live_price * (natr / 100.0)

    if "BUY" in action:
        stop_loss = current_live_price - (1.5 * atr_dollars)
        take_profit = expected_target_price if expected_target_price > current_live_price else current_live_price + (2.0 * atr_dollars)
    else:
        stop_loss = current_live_price + (1.5 * atr_dollars)
        take_profit = expected_target_price if expected_target_price < current_live_price else current_live_price - (2.0 * atr_dollars)

    target_time_utc = t_curr_mid + pd.Timedelta(hours=H_half)

    return {
        "x_days": x_days,
        "half_days": x_days / 2.0,
        "current_time_utc": str(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
        "period_start_time": str(df_live.index[idx_start].strftime("%Y-%m-%d %H:%M:%S UTC")),
        "target_end_time": str(target_time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")),
        "prev_week_window": f"{df_live.index[idx_prev_start].strftime('%m-%d %H:%M')} to {df_live.index[idx_prev_end].strftime('%m-%d %H:%M')}",
        "current_live_price": round(current_live_price, 2),
        "period_start_price": round(p_start, 2),
        "interim_delta_dollars": round(interim_delta_dollars, 2),
        "interim_delta_pct": round(interim_delta_pct, 2),
        "action": action,
        "action_icon": action_icon,
        "bias": bias,
        "confidence": confidence,
        "prob_up": round(prob_up * 100.0, 1),
        "prob_down": round(prob_down * 100.0, 1),
        "pred_total_delta_pct": round(pred_delta_pct * 100.0, 2),
        "pred_total_delta_dollars": round(pred_delta_dollars, 2),
        "expected_target_price": round(expected_target_price, 2),
        "remaining_move_dollars": round(remaining_delta_dollars, 2),
        "remaining_move_pct": round(remaining_delta_pct, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "atr_dollars": round(atr_dollars, 2),
        "rsi_14": round(feature_dict.get('rsi_now', 50.0), 1),
        "macd": round(feature_dict.get('macd_now', 0.0), 2),
        "news_tone_diff": round(feature_dict.get('news_tone_diff', 0.0), 2),
        "news_vol_w1": feature_dict.get('news_vol_w1', 0),
        "news_vol_w2": feature_dict.get('news_vol_w2', 0)
    }


def print_x_period_action_dashboard(action_data: Dict[str, Any]):
    """Print the final executive action dashboard answering BUY or SELL AT THIS TIME."""
    d = action_data

    print("\n" + "╔" + "═" * 74 + "╗")
    print(f"║             BITCOIN DECISION DASHBOARD: {d['x_days']:g}-DAY PERIOD ANALYSIS           ║")
    print("╠" + "═" * 74 + "╣")
    print(f"║  Current Time (UTC)    : {d['current_time_utc']:<47} ║")
    print(f"║  Current Live BTC Price: ${d['current_live_price']:<12,.2f}  (Start of Period: ${d['period_start_price']:<11,.2f})  ║")
    print(f"║  First {d['half_days']:g}d Interim Move : ${d['interim_delta_dollars']:+10,.2f}  ({d['interim_delta_pct']:+.2f}%)                            ║")
    print(f"║  Previous Week Window  : {d['prev_week_window']:<47} ║")
    print("╠" + "═" * 74 + "╣")
    print(f"║  ACTION TO TAKE NOW    : {d['action_icon']} {d['action']:<45} ║")
    print(f"║  Market Bias           : {d['bias']:<47} ║")
    print(f"║  Model Conviction      : {d['confidence']:<47} ║")
    print(f"║  Upward Probability    : {d['prob_up']}% (Downside Risk: {d['prob_down']}%)                 ║")
    print("╠" + "═" * 74 + "╣")
    print(f"║                   PERIOD TARGET & DELTA PROJECTION                       ║")
    print("╠" + "═" * 74 + "╣")
    print(f"║  Formula Target        : Price(End of {d['x_days']:g}d) - Price(Start of {d['x_days']:g}d)            ║")
    print(f"║  Period Start Price    : ${d['period_start_price']:<12,.2f}  [{d['period_start_time']}]   ║")
    print(f"║  Predicted Total Delta : ${d['pred_total_delta_dollars']:+10,.2f}  ({d['pred_total_delta_pct']:+.2f}%)                            ║")
    print(f"║  Target Price at End   : ${d['expected_target_price']:<12,.2f}  [{d['target_end_time']}]     ║")
    print(f"║  Expected Move from Now: ${d['remaining_move_dollars']:+10,.2f}  ({d['remaining_move_pct']:+.2f}%)                            ║")
    print("╠" + "═" * 74 + "╣")
    print(f"║                        RISK MANAGEMENT LEVELS                            ║")
    print("╠" + "═" * 74 + "╣")
    print(f"║  Execution Entry       : ${d['current_live_price']:<12,.2f}  (At Market)                       ║")
    print(f"║  Stop Loss (Risk)      : ${d['stop_loss']:<12,.2f}  (1.5x Volatility Cushion)         ║")
    print(f"║  Take Profit (Target)  : ${d['take_profit']:<12,.2f}  (Model Projected Exit)            ║")
    print(f"║  Hourly Volatility     : ${d['atr_dollars']:<10,.2f}                                    ║")
    print("╠" + "═" * 74 + "╣")
    print(f"║                       CONFLUENCE & SENTIMENT SIGNALS                     ║")
    print("╠" + "═" * 74 + "╣")
    rsi_str = f"{d['rsi_14']:.1f}"
    macd_str = f"{d['macd']:+.1f}"
    print(f"║  • RSI (14) Momentum   : {rsi_str:<6} (Below 30 = Oversold, Above 70 = Overbought)║")
    print(f"║  • MACD Signal Trend   : {macd_str:<6} (Positive = Bullish, Negative = Bearish)   ║")
    print(f"║  • GDELT Tone Shift    : {d['news_tone_diff']:+5.2f}  (Delta vs Previous Week Window)           ║")
    print(f"║  • GDELT Articles      : {d['news_vol_w1']} articles in Window 1 vs {d['news_vol_w2']} in Window 2        ║")
    print("╚" + "═" * 74 + "╝\n")

