"""Bitcoin market data fetcher (OHLCV) using Binance Public API with pagination, retry, and gap detection."""

import json
import time
import urllib.request
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from config import (
    USER_AGENT,
    DEFAULT_INTERVAL,
    DEFAULT_LOOKBACK_CANDLES,
    BINANCE_CHUNK_SIZE,
    BINANCE_RATE_LIMIT_SLEEP,
    TRAINING_HISTORY_YEARS
)
from database import insert_btc_prices, init_db


def _interval_to_seconds(interval: str) -> int:
    """Map Binance interval string to seconds."""
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "12h": 43200,
        "1d": 86400,
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}. Supported: {list(mapping.keys())}")
    return mapping[interval]


def _drop_unclosed_klines(klines: List[list]) -> List[list]:
    """Drop the still-forming candle (close time at/after now), unless it would leave <=1 candle."""
    if len(klines) <= 1:
        return klines
    now_ms = int(time.time() * 1000)
    closed = [item for item in klines if int(item[6]) < now_ms]
    return closed if len(closed) > 1 else klines


def fetch_binance_btc_klines_chunk(
    interval: str = "1h",
    limit: int = 1000,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
    max_retries: int = 4
) -> List[list]:
    """
    Fetch a single chunk of klines from Binance API with exponential backoff for retries and rate limits.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
    if start_time_ms:
        url += f"&startTime={start_time_ms}"
    if end_time_ms:
        url += f"&endTime={end_time_ms}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return _drop_unclosed_klines(data)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if attempt == max_retries - 1:
                    raise
                wait_time = (attempt + 1) * 3.0
                print(f"  [!] Binance API rate-limit (429). Backing off for {wait_time:.1f}s...")
                time.sleep(wait_time)
            elif e.code in (500, 502, 503, 504):
                if attempt == max_retries - 1:
                    raise
                wait_time = (attempt + 1) * 2.0
                print(f"  [!] Binance server error ({e.code}). Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"  [!] Binance HTTP error {e.code}: {e.reason}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1.0)
        except Exception as e:
            print(f"  [!] Network error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))

    return []


def fetch_paginated_btc_klines(
    interval: str = DEFAULT_INTERVAL,
    lookback_candles: int = DEFAULT_LOOKBACK_CANDLES,
    end_time_ms: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Paginate backwards from end_time_ms to retrieve lookback_candles.
    Deduplicates timestamps, detects data gaps, and validates continuity.
    """
    print(f"[*] Requesting {lookback_candles} candles ({interval}) via Binance pagination (chunk size={BINANCE_CHUNK_SIZE})...")

    all_raw_klines = []
    current_end_time = end_time_ms
    retrieved_count = 0

    while retrieved_count < lookback_candles:
        needed = lookback_candles - retrieved_count
        chunk_limit = min(BINANCE_CHUNK_SIZE, needed)

        chunk = fetch_binance_btc_klines_chunk(
            interval=interval,
            limit=chunk_limit,
            end_time_ms=current_end_time
        )

        if not chunk:
            print("  [*] No further historical klines returned by API.")
            break

        all_raw_klines = chunk + all_raw_klines
        retrieved_count += len(chunk)

        earliest_open_time = chunk[0][0]
        current_end_time = earliest_open_time - 1

        print(f"  -> Retrieved {len(chunk)} candles. Progress: {len(all_raw_klines)}/{lookback_candles}")

        # The first unbounded request includes the still-forming candle, which
        # the fetcher drops, so a full page can legitimately be chunk_limit-1.
        if len(chunk) < chunk_limit - 1:
            print("  [*] Reached the earliest available historical data on exchange.")
            break

        time.sleep(BINANCE_RATE_LIMIT_SLEEP)

    # Deduplicate by open_time
    seen_timestamps = {}
    for item in all_raw_klines:
        open_time = item[0]
        seen_timestamps[open_time] = item

    sorted_open_times = sorted(seen_timestamps.keys())
    deduped_raw = [seen_timestamps[ts] for ts in sorted_open_times]

    # Convert to structured dictionary
    results = []
    for item in deduped_raw:
        dt = datetime.fromtimestamp(item[0] / 1000.0, tz=timezone.utc)
        iso_str = dt.strftime("%Y-%m-%d %H:%M:%S")

        results.append({
            "timestamp": iso_str,
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
            "quote_volume": float(item[7]),
            "trades_count": int(item[8])
        })

    # Gap detection
    step_ms = _interval_to_seconds(interval) * 1000
    detected_gaps = []
    for i in range(len(sorted_open_times) - 1):
        diff = sorted_open_times[i + 1] - sorted_open_times[i]
        if diff != step_ms:
            missing_count = int(diff // step_ms) - 1
            t_gap_start = datetime.fromtimestamp(sorted_open_times[i] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            t_gap_end = datetime.fromtimestamp(sorted_open_times[i + 1] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            detected_gaps.append({
                "from": t_gap_start,
                "to": t_gap_end,
                "missing_candles": missing_count
            })

    date_range = (results[0]["timestamp"], results[-1]["timestamp"]) if results else (None, None)
    summary = {
        "requested_candles": lookback_candles,
        "actual_candles": len(results),
        "raw_retrieved": len(all_raw_klines),
        "duplicates_removed": len(all_raw_klines) - len(results),
        "interval": interval,
        "date_start": date_range[0],
        "date_end": date_range[1],
        "gaps_count": len(detected_gaps),
        "gaps": detected_gaps[:5]  # first 5 gaps if any
    }

    if detected_gaps:
        print(f"  [!] Warning: Detected {len(detected_gaps)} time gaps in candlestick sequence.")
        for g in detected_gaps[:3]:
            print(f"      Gap: {g['from']} -> {g['to']} ({g['missing_candles']} missing candles)")
    else:
        print("  [+] Data verification passed: 0 gaps detected in historical candle sequence.")

    return results, summary


def ingest_btc_market_data(
    interval: str = DEFAULT_INTERVAL,
    lookback_candles: int = DEFAULT_LOOKBACK_CANDLES,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch and ingest Bitcoin price candles into SQLite with full pagination.
    Returns metadata summary.
    """
    klines, summary = fetch_paginated_btc_klines(interval=interval, lookback_candles=lookback_candles)
    inserted = insert_btc_prices(klines, db_path=db_path)
    summary["inserted_or_updated"] = inserted
    print(f"[*] Ingested {inserted} candles into database. Date range: {summary['date_start']} to {summary['date_end']}")
    return summary


def ingest_btc_history(
    years: float = TRAINING_HISTORY_YEARS,
    interval: str = DEFAULT_INTERVAL,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Backfill enough hourly candles to cover the requested training history."""
    init_db(db_path=db_path)
    candles = int(float(years) * 365.25 * 24)
    print(f"[*] Backfilling {years:g} years of {interval} candles ({candles} bars)...")
    return ingest_btc_market_data(interval=interval, lookback_candles=candles, db_path=db_path)
