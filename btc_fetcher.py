"""Bitcoin market data fetcher (OHLCV) using Binance Public API."""

import json
import urllib.request
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from config import USER_AGENT
from database import insert_btc_prices


def fetch_binance_btc_klines(
    interval: str = "1h",
    limit: int = 500,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Fetch BTC/USDT candlestick data from Binance public API.
    intervals supported: '15m', '1h', '4h', '1d'
    Returns structured list of OHLCV dictionaries.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
    if start_time_ms:
        url += f"&startTime={start_time_ms}"
    if end_time_ms:
        url += f"&endTime={end_time_ms}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = []
    for item in data:
        # Binance kline format:
        # [0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume, 6: close_time, 7: quote_volume, 8: trades, ...]
        timestamp_ms = item[0]
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
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

    return results


def ingest_btc_market_data(interval: str = "1h", limit: int = 500) -> int:
    """
    Fetch and ingest recent Bitcoin price candles into SQLite.
    Returns total rows inserted.
    """
    print(f"[*] Fetching last {limit} BTC/USDT candles ({interval})...")
    klines = fetch_binance_btc_klines(interval=interval, limit=limit)
    inserted = insert_btc_prices(klines)
    print(f"[*] Successfully inserted/updated {inserted} BTC price candles.")
    return inserted
