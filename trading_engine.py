"""Multi-period live automated trading engine with 4x leverage, fee deduction, and liquidation tracking."""

import os
import sys
import json
import time
import math
import asyncio
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

# Ensure project root and .venv site-packages are in sys.path
BASE_DIR = Path(__file__).resolve().parent
_site_packages = BASE_DIR / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _site_packages.is_dir() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from config import (
    DATA_DIR, USER_AGENT, MIN_ENTRY_HIT_PROB, EXIT_HIT_PROB,
    MODEL_RETRAIN_INTERVAL_HOURS
)
from signal_engine import generate_x_period_action
from database import get_connection, insert_btc_prices, insert_news_batch
from btc_fetcher import fetch_binance_btc_klines_chunk
from gdelt_fetcher import get_recent_gkg_urls, fetch_and_process_gkg_zip

STATE_FILE = DATA_DIR / "live_trading_state.json"
_last_sync_timestamp = 0.0

# Suppress repeated alerts with the same signature (period/type/direction pair)
ALERT_COOLDOWN_SECONDS = 600.0


def sync_latest_market_and_news_data(db_path: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """
    Sync latest market candles from Binance and newest 15-minute news batch from GDELT into SQLite.
    Throttled to run at most once every 60 seconds unless force=True.
    """
    global _last_sync_timestamp
    now_ts = time.time()
    if not force and (now_ts - _last_sync_timestamp < 60.0):
        return {"status": "throttled", "cached": True}

    stats = {
        "candles_inserted": 0,
        "news_inserted": 0,
        "sync_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    # 1. Update Binance 1h candles into SQLite btc_prices
    try:
        raw = fetch_binance_btc_klines_chunk(interval="1h", limit=50)
        if raw:
            records = [{
                "timestamp": datetime.fromtimestamp(item[0] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "quote_volume": float(item[7]),
                "trades_count": int(item[8])
            } for item in raw]
            stats["candles_inserted"] = insert_btc_prices(records, db_path=db_path)
    except Exception as e:
        print(f"[!] Warning: Failed to sync Binance candles: {e}")

    # 2. Update latest GDELT news batch into SQLite news_articles
    try:
        urls = get_recent_gkg_urls(num_batches=1)
        if urls:
            articles = fetch_and_process_gkg_zip(urls[-1], filter_crypto_finance_only=False)
            if articles:
                stats["news_inserted"] = insert_news_batch(articles, db_path=db_path)
    except Exception as e:
        print(f"[!] Warning: Failed to sync GDELT news: {e}")

    _last_sync_timestamp = time.time()
    return stats

PERIOD_CONFIGS = {
    "3h": {
        "name": "3 Hours",
        "hours": 3.0,
        "x_days": 3.0 / 24.0,  # 0.125 days
        "fee_rate": 0.001,      # 0.1%
        "leverage": 4.0,
    },
    "6h": {
        "name": "6 Hours",
        "hours": 6.0,
        "x_days": 6.0 / 24.0,  # 0.25 days
        "fee_rate": 0.001,      # 0.1%
        "leverage": 4.0,
    },
    "12h": {
        "name": "12 Hours",
        "hours": 12.0,
        "x_days": 12.0 / 24.0, # 0.5 days
        "fee_rate": 0.001,      # 0.1%
        "leverage": 4.0,
    },
    "1d": {
        "name": "1 Day (24h)",
        "hours": 24.0,
        "x_days": 1.0,         # 1.0 day
        "fee_rate": 0.001,      # 0.1%
        "leverage": 4.0,
    },
    "3d": {
        "name": "3 Days (72h)",
        "hours": 72.0,
        "x_days": 3.0,         # 3.0 days
        "fee_rate": 0.003,      # 0.3% specifically for 3-day mode
        "leverage": 4.0,
    }
}


_last_valid_live_btc_price: float = 63700.0

PRICE_PROVIDERS = [
    # 1. Coinbase (Official Public Spot API)
    (
        "Coinbase",
        "https://api.coinbase.com/v2/prices/BTC-USD/spot",
        lambda data: float(data["data"]["amount"]),
    ),
    # 2. Kraken (Official Public Ticker API)
    (
        "Kraken",
        "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
        lambda data: float(data["result"]["XXBTZUSD"]["c"][0]),
    ),
    # 3. Binance Vision (Official high-capacity public data mirror)
    (
        "Binance Vision",
        "https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT",
        lambda data: float(data["price"]),
    ),
    # 4. Binance Main API
    (
        "Binance Main",
        "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        lambda data: float(data["price"]),
    ),
]


def fetch_live_btc_price_fast() -> float:
    """
    Fetch real-time BTC price with multi-provider failover and sticky memory cache.
    Tries Coinbase, Kraken, Binance Vision, and Binance Main.
    Guarantees zero artificial price drops: if any endpoint experiences a timeout
    or network rate-limit, it immediately fails over to the next; if all fail,
    it returns the last known verified price rather than jumping to a hardcoded value.
    """
    global _last_valid_live_btc_price

    for name, url, parser in PRICE_PROVIDERS:
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                price = parser(data)
                if price and price > 1000.0:
                    _last_valid_live_btc_price = price
                    return price
        except Exception:
            continue

    # Return sticky last verified live price if all endpoints temporarily failed
    return _last_valid_live_btc_price


def _parse_state_datetime(value: Any) -> Optional[datetime]:
    """Load an ISO timestamp from persisted state without failing on legacy/null values."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (TypeError, ValueError):
        return None


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF via bisection (no scipy dependency)."""
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def period_model_signal(period_hours: float) -> Dict[str, Any]:
    """
    Model signal whose forecast horizon equals the trade period.
    The X-day model predicts X/2 ahead of decision time, so a period of P hours
    uses the model with x_days = 2*P/24 (e.g. 3h trade -> 6h model, 1d -> 2d model).
    """
    return generate_x_period_action(x_days=2.0 * period_hours / 24.0, live_candle=True)


def trade_end_hit_prob(model_result: Dict[str, Any], direction: str, entry_price: float,
                       live_price: float, hours_left: float, period_hours: float) -> Dict[str, float]:
    """
    Probability that price at the trade's own end_time is on the winning side of entry.
    Model gives p_up for the full period; its implied standardized drift z is scaled to the
    remaining time, and the move already made (live vs entry) is added:
        P(up at end) = Phi( ln(live/entry)/(sigma_1h*sqrt(R)) + z*sqrt(R/L) )
    As R -> 0 this converges to 1 or 0 depending on where price sits vs entry.
    """
    R = max(hours_left, 1.0 / 60.0)
    L = max(period_hours, 1e-6)
    sigma = max(float(model_result.get("sigma_1h") or 0.0), 1e-4)
    z = _norm_ppf(float(model_result.get("prob_up", 50.0)) / 100.0)
    d = math.log(live_price / entry_price)
    p_up = _norm_cdf(d / (sigma * math.sqrt(R)) + z * math.sqrt(min(R, L) / L))
    p_up *= 100.0
    return {"prob_up": round(p_up, 1), "prob_down": round(100.0 - p_up, 1)}


class PeriodStrategyTracker:
    """
    Manages live automated 4x leveraged execution for one specific time period.
    Initial balance: 1,000.00 USDT.
    """
    def __init__(self, period_id: str, config: Dict[str, Any], manager: Optional[Any] = None):
        self.period_id = period_id
        self.name = config["name"]
        self.hours = config["hours"]
        self.x_days = config["x_days"]
        self.fee_rate = config["fee_rate"]
        self.leverage = config["leverage"]
        self.manager = manager

        self.initial_usdt = 1000.0
        self.balance_usdt = 1000.0
        self.total_realized_pnl_usdt = 0.0
        self.total_fees_paid_usdt = 0.0

        self.active_trade: Optional[Dict[str, Any]] = None
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_history: List[Dict[str, Any]] = []

        self.last_signal_direction: Optional[str] = None
        self.last_signal_hit_prob: Optional[float] = None
        self.active_alert: Optional[Dict[str, Any]] = None
        self.last_alert_signature: Optional[tuple] = None
        self.last_alert_time: Optional[datetime] = None
        self.recent_alert_times: Dict[tuple, datetime] = {}
        self.reentry_blocked_until: Optional[datetime] = None

        self.live_hit_prob: Optional[float] = None
        self.live_prob_up: Optional[float] = None
        self.live_prob_down: Optional[float] = None
        self.live_confidence: Optional[str] = None
        self.last_live_eval_time: Optional[datetime] = None

        self.is_liquidated = False

    def evaluate_alert(self, new_direction: str, new_hit_prob: float, live_price: float,
                       current_time: datetime, live_hit_prob: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Check whether the freshly evaluated signal changed direction or conviction.

        Alerts fire ONLY when:
        1. The direction flips relative to the LAST EVALUATED SIGNAL and the new
           direction has high hit probability (>= 60%). Severity is CRITICAL at
           >= 65%, otherwise WARNING.
        2. Or the hit probability of the same direction shifts by >= 25% and the
           new probability is >= 60%.

        The baseline is `last_signal_*` (not the entry signal) so an unchanged
        reversal cannot fire again on every 30-second evaluation. A short
        cooldown also suppresses flapping between the same two states.
        """
        if self.last_signal_direction is not None:
            old_dir = self.last_signal_direction
            old_prob = self.last_signal_hit_prob if self.last_signal_hit_prob is not None else 50.0
        elif self.active_trade:
            old_dir = self.active_trade["direction"]
            old_prob = self.active_trade.get("hit_probability_pct", 50.0)
        else:
            self.last_signal_direction = new_direction
            self.last_signal_hit_prob = new_hit_prob
            return None

        reversed_dir = new_direction != old_dir
        # For a reversal, compare conviction in the NEW direction rather than
        # subtracting two hit probabilities that describe opposite outcomes.
        prob_diff = (new_hit_prob - (100.0 - old_prob)) if reversed_dir else (new_hit_prob - old_prob)
        prob_diff = round(max(-100.0, min(100.0, prob_diff)), 1)

        alert_type = None
        severity = None
        if reversed_dir and new_hit_prob >= 60.0:
            alert_type = "DIRECTION_REVERSAL"
            severity = "CRITICAL" if new_hit_prob >= 65.0 else "WARNING"
        elif not reversed_dir and abs(prob_diff) >= 25.0 and new_hit_prob >= 60.0:
            alert_type = "PROBABILITY_SHIFT"
            severity = "WARNING"

        signature = (self.period_id, alert_type, old_dir, new_direction) if alert_type else None
        previous_time = self.recent_alert_times.get(signature) if signature else None
        in_cooldown = (
            previous_time is not None
            and (current_time - previous_time).total_seconds() < ALERT_COOLDOWN_SECONDS
        )

        alert = None
        if alert_type and not in_cooldown:
            live_note = ""
            if live_hit_prob is not None:
                live_note = f" (live position win prob {live_hit_prob:.1f}%)"
            if alert_type == "DIRECTION_REVERSAL":
                title = f"🚨 {self.name} Direction Reversal Alert!"
                message = (
                    f"{self.name} signal flipped from {old_dir} ({old_prob:.1f}%) to high-conviction "
                    f"{new_direction} ({new_hit_prob:.1f}%){live_note}. Significant directional reversal detected."
                )
            else:
                title = f"⚠️ {self.name} Hit Prob Swing Alert!"
                message = (
                    f"{self.name} {new_direction} hit probability shifted by {prob_diff:+.1f}% "
                    f"({old_prob:.1f}% → {new_hit_prob:.1f}%){live_note}."
                )
            alert = {
                "alert_id": f"alert-{self.period_id}-{int(current_time.timestamp() * 1000)}",
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "period_id": self.period_id,
                "period_name": self.name,
                "type": alert_type,
                "severity": severity,
                "dismissed": False,
                "old_direction": old_dir,
                "new_direction": new_direction,
                "old_hit_prob": round(old_prob, 1),
                "new_hit_prob": round(new_hit_prob, 1),
                "prob_change_pct": prob_diff,
                "live_hit_prob": round(live_hit_prob, 1) if live_hit_prob is not None else None,
                "title": title,
                "message": message,
                "price_at_alert": round(live_price, 2)
            }
            self.last_alert_signature = signature
            self.last_alert_time = current_time
            self.recent_alert_times[signature] = current_time
            # Keep the cooldown map small; stale entries can never fire again.
            if len(self.recent_alert_times) > 32:
                cutoff = current_time - timedelta(seconds=ALERT_COOLDOWN_SECONDS)
                self.recent_alert_times = {
                    sig: when for sig, when in self.recent_alert_times.items() if when > cutoff
                }

        # Update tracked signal state AFTER the baseline has been used.
        self.last_signal_direction = new_direction
        self.last_signal_hit_prob = new_hit_prob

        if alert:
            self.active_alert = alert
            print(f"\n[ALERT] 🚨 {alert['message']}\n")
        else:
            self.active_alert = None

        return alert

    def to_dict(self, live_price: float) -> Dict[str, Any]:
        """Generate comprehensive live status payload for API & dashboard."""
        unrealized_usdt = 0.0
        unrealized_pct = 0.0
        current_equity_usdt = self.balance_usdt

        trade_info = None
        if self.active_trade:
            t = self.active_trade
            direction = t["direction"]
            entry_price = t["entry_price"]
            pos_qty_btc = t["position_qty_btc"]
            margin_usdt = t.get("margin_usdt", t.get("margin_usd", self.balance_usdt))

            # Real-time mark price incorporating closing slippage (0.01%)
            if direction == "LONG":
                exit_price_now = live_price * 0.9999
                price_move_pct = (exit_price_now - entry_price) / entry_price
                unrealized_pct = price_move_pct * self.leverage * 100.0
                unrealized_usdt = pos_qty_btc * (exit_price_now - entry_price)
            else:  # SHORT
                exit_price_now = live_price * 1.0001
                price_move_pct = (entry_price - exit_price_now) / entry_price
                unrealized_pct = price_move_pct * self.leverage * 100.0
                unrealized_usdt = pos_qty_btc * (entry_price - exit_price_now)

            current_equity_usdt = max(0.0, self.balance_usdt + unrealized_usdt)

            # Time calculations
            now_dt = datetime.now(timezone.utc)
            start_dt = datetime.fromisoformat(t["start_time"])
            end_dt = datetime.fromisoformat(t["end_time"])
            total_sec = max(1.0, (end_dt - start_dt).total_seconds())
            elapsed_sec = max(0.0, (now_dt - start_dt).total_seconds())
            progress_pct = min(100.0, max(0.0, (elapsed_sec / total_sec) * 100.0))
            time_left_sec = max(0, int((end_dt - now_dt).total_seconds()))

            entry_hit_prob = t.get("entry_hit_probability_pct", t.get("hit_probability_pct", round(t["prob_up"] if direction == "LONG" else t["prob_down"], 1)))
            live_hit_prob = self.live_hit_prob if self.live_hit_prob is not None else entry_hit_prob
            delta_prob = round(live_hit_prob - entry_hit_prob, 1)

            trade_info = {
                "trade_id": t["trade_id"],
                "direction": direction,
                "direction_icon": "🟢 LONG" if direction == "LONG" else "🔴 SHORT",
                "leverage": f"{int(self.leverage)}x",
                "entry_market_price": round(t["market_price_at_entry"], 2),
                "entry_price": round(entry_price, 2),
                "liquidation_price": round(t["liquidation_price"], 2),
                "target_price": round(t["target_price"], 2),
                "position_qty_btc": round(pos_qty_btc, 5),
                "position_notional_usdt": round(pos_qty_btc * entry_price, 2),
                "margin_usdt": round(margin_usdt, 2),
                "fee_deducted_usdt": round(t.get("fee_usdt", 0.0), 2),
                "fee_pct": round(self.fee_rate * 100.0, 2),
                "unrealized_pnl_usdt": round(unrealized_usdt, 2),
                "unrealized_pnl_pct": round(unrealized_pct, 2),
                "confidence": t["confidence"],
                "live_confidence": self.live_confidence or t["confidence"],
                "prob_up": t["prob_up"],
                "prob_down": t["prob_down"],
                "live_prob_up": self.live_prob_up if self.live_prob_up is not None else t["prob_up"],
                "live_prob_down": self.live_prob_down if self.live_prob_down is not None else t["prob_down"],
                "entry_hit_probability_pct": round(entry_hit_prob, 1),
                "hit_probability_pct": round(entry_hit_prob, 1),
                "live_hit_probability_pct": round(live_hit_prob, 1),
                "live_hit_prob_delta": delta_prob,
                "expected_delta_usd": round(t.get("expected_delta_usd", 0.0), 2),
                "start_time": t["start_time"],
                "end_time": t["end_time"],
                "opened_at_exact": t.get("opened_at_exact", t["start_time"]),
                "opened_at_local": t.get("opened_at_local", t["start_time"]),
                "end_at_exact": t.get("end_at_exact", t["end_time"]),
                "elapsed_str": str(timedelta(seconds=int(elapsed_sec))),
                "progress_pct": round(progress_pct, 1),
                "time_remaining_str": str(timedelta(seconds=time_left_sec)),
                "time_remaining_sec": time_left_sec,
            }

        # Win Rate & Stats from Trade History
        closed_trades = [h for h in self.trade_history if h.get("status") in ("CLOSED", "LIQUIDATED")]
        total_closed = len(closed_trades)
        winning_trades = sum(1 for h in closed_trades if h.get("net_pnl_usdt", 0) > 0)
        win_rate = (winning_trades / total_closed * 100.0) if total_closed > 0 else 0.0

        total_return_pct = ((current_equity_usdt - self.initial_usdt) / self.initial_usdt) * 100.0

        cur_live_hit = self.live_hit_prob
        if cur_live_hit is None and trade_info:
            cur_live_hit = trade_info["live_hit_probability_pct"]
        cur_entry_hit = trade_info["entry_hit_probability_pct"] if trade_info else (cur_live_hit or 50.0)
        cur_delta = round(cur_live_hit - cur_entry_hit, 1) if (cur_live_hit is not None and cur_entry_hit is not None) else 0.0

        return {
            "period_id": self.period_id,
            "name": self.name,
            "hours": self.hours,
            "leverage": f"{int(self.leverage)}x",
            "fee_rate_pct": round(self.fee_rate * 100.0, 2),
            "is_liquidated": self.is_liquidated,
            "balance_usdt": round(self.balance_usdt, 2),
            "current_equity_usdt": round(current_equity_usdt, 2),
            "initial_usdt": self.initial_usdt,
            "total_return_pct": round(total_return_pct, 2),
            "total_realized_pnl_usdt": round(self.total_realized_pnl_usdt, 2),
            "total_fees_paid_usdt": round(self.total_fees_paid_usdt, 2),
            "win_rate_pct": round(win_rate, 1),
            "total_trades": total_closed,
            "winning_trades": winning_trades,
            "active_trade": trade_info,
            "live_hit_probability_pct": round(cur_live_hit, 1) if cur_live_hit is not None else 50.0,
            "entry_hit_probability_pct": round(cur_entry_hit, 1) if cur_entry_hit is not None else 50.0,
            "live_hit_prob_delta": cur_delta,
            "live_confidence": self.live_confidence or (trade_info.get("confidence") if trade_info else "Moderate Conviction"),
            "live_prob_up": self.live_prob_up,
            "live_prob_down": self.live_prob_down,
            "active_alert": self.active_alert,
            "reentry_blocked_until": self.reentry_blocked_until.isoformat() if self.reentry_blocked_until else None,
            "last_signal_direction": self.last_signal_direction,
            "last_signal_hit_prob": self.last_signal_hit_prob,
            "recent_trades": list(reversed(self.trade_history[-10:])),
            "equity_history": self.equity_history[-30:]
        }

    def open_new_trade(self, live_price: float, current_time: datetime, model_result: Optional[Dict[str, Any]] = None):
        """
        Open a 4x leveraged LONG or SHORT trade at the start of the period.
        Initial balance is 1,000.00 USDT.
        Deducts upfront fee (0.1% or 0.3% for 3d).
        Applies slippage (Buy +0.01%, Sell -0.01%).
        """
        if self.balance_usdt <= 1.0:
            self.is_liquidated = True
            return False

        # Do not re-enter immediately after an early probability-based exit.
        if self.reentry_blocked_until is not None and current_time < self.reentry_blocked_until:
            return False

        # If model result not provided, synchronize fresh data & run inference
        if model_result is None:
            try:
                sync_latest_market_and_news_data()
                model_result = period_model_signal(self.hours)
            except Exception as e:
                print(f"[!] {self.period_id.upper()} inference failed, no trade opened: {e}")
                return False

        action_label = str(model_result.get("action", "BUY")).upper()
        if "WAIT" in action_label:
            return False

        direction = "LONG" if "BUY" in action_label else "SHORT"
        _pu = float(model_result.get("prob_up", 50.0))
        _pd = float(model_result.get("prob_down", 50.0))
        _hit = _pu if direction == "LONG" else _pd
        self.live_prob_up, self.live_prob_down = round(_pu, 1), round(_pd, 1)
        self.live_hit_prob = round(_hit, 1)

        # Entry gate: only open when hit probability is above the threshold
        if _hit <= MIN_ENTRY_HIT_PROB:
            return False

        # Deduct upfront fee (0.1% or 0.3% for 3d)
        fee_usdt = self.balance_usdt * self.fee_rate
        self.balance_usdt -= fee_usdt
        self.total_fees_paid_usdt += fee_usdt

        margin_usdt = self.balance_usdt
        position_notional_usdt = margin_usdt * self.leverage

        # Apply slippage: Buy price is 0.01% higher, Sell price is 0.01% lower
        if direction == "LONG":
            entry_price = live_price * 1.0001  # +0.01% higher
            # Liquidation price: adverse drop of 23% wipes out 4x margin
            liq_price = entry_price * (1.0 - (1.0 / self.leverage - 0.02))  # ~0.77x
            target_price = entry_price + abs(model_result.get("pred_total_delta_dollars", live_price * 0.02))
        else:  # SHORT
            entry_price = live_price * 0.9999  # -0.01% lower
            # Liquidation price: adverse rally of 23% wipes out 4x margin
            liq_price = entry_price * (1.0 + (1.0 / self.leverage - 0.02))  # ~1.23x
            target_price = entry_price - abs(model_result.get("pred_total_delta_dollars", live_price * 0.02))

        position_qty_btc = position_notional_usdt / entry_price

        start_time = current_time
        end_time = start_time + timedelta(hours=self.hours)

        prob_up = float(model_result.get("prob_up", 50.0))
        prob_down = float(model_result.get("prob_down", 50.0))
        hit_prob = prob_up if direction == "LONG" else prob_down
        self.live_hit_prob = round(hit_prob, 1)
        self.live_prob_up = round(prob_up, 1)
        self.live_prob_down = round(prob_down, 1)
        self.live_confidence = model_result.get("confidence", "High Conviction")

        # Check for Direction Change / Probability Swing Alert
        alert = self.evaluate_alert(direction, hit_prob, live_price, current_time)
        if alert and self.manager:
            self.manager.record_alert(alert)

        trade_id = f"{self.period_id}-{int(start_time.timestamp())}"
        self.active_trade = {
            "trade_id": trade_id,
            "direction": direction,
            "market_price_at_entry": live_price,
            "entry_price": entry_price,
            "liquidation_price": liq_price,
            "target_price": target_price,
            "margin_usdt": margin_usdt,
            "position_notional_usdt": position_notional_usdt,
            "position_qty_btc": position_qty_btc,
            "fee_usdt": fee_usdt,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "opened_at_exact": start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "opened_at_local": start_time.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
            "end_at_exact": end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "confidence": model_result.get("confidence", "High Conviction"),
            "prob_up": round(prob_up, 1),
            "prob_down": round(prob_down, 1),
            "entry_hit_probability_pct": round(hit_prob, 1),
            "hit_probability_pct": round(hit_prob, 1),
            "live_hit_probability_pct": round(hit_prob, 1),
            "expected_delta_usd": model_result.get("pred_total_delta_dollars", 0.0)
        }
        return True

    def close_active_trade(self, live_price: float, current_time: datetime, reason: str = "PERIOD_EXPIRED", exit_reason: Optional[str] = None):
        """
        Close active position at end of period (or liquidation).
        Applies closing slippage and books realized PnL in USDT.
        """
        if not self.active_trade:
            return

        t = self.active_trade
        direction = t["direction"]
        entry_price = t["entry_price"]
        pos_qty_btc = t["position_qty_btc"]

        if reason == "LIQUIDATED":
            exit_price = t["liquidation_price"]
            net_pnl_usdt = -t["margin_usdt"]
            pnl_pct = -100.0
            self.balance_usdt = 0.0
            self.is_liquidated = True
        else:
            # Normal period completion:
            # Closing LONG sells at market * 0.9999 (-0.01% slippage)
            # Closing SHORT buys at market * 1.0001 (+0.01% slippage)
            if direction == "LONG":
                exit_price = live_price * 0.9999
                price_move_pct = (exit_price - entry_price) / entry_price
                net_pnl_usdt = pos_qty_btc * (exit_price - entry_price)
            else:
                exit_price = live_price * 1.0001
                price_move_pct = (entry_price - exit_price) / entry_price
                net_pnl_usdt = pos_qty_btc * (entry_price - exit_price)

            pnl_pct = price_move_pct * self.leverage * 100.0
            self.balance_usdt = max(0.0, self.balance_usdt + net_pnl_usdt)

        self.total_realized_pnl_usdt += net_pnl_usdt

        # Record trade in history
        history_record = {
            "trade_id": t["trade_id"],
            "period": self.period_id,
            "direction": direction,
            "direction_icon": "🟢 LONG" if direction == "LONG" else "🔴 SHORT",
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "start_time": t["start_time"],
            "exit_time": current_time.isoformat(),
            "net_pnl_usdt": round(net_pnl_usdt, 2),
            "pnl_pct": round(pnl_pct, 2),
            "fee_usdt": round(t["fee_usdt"], 2),
            "status": reason,
            "exit_reason": exit_reason or reason,
            "balance_after_usdt": round(self.balance_usdt, 2),
            "confidence": t["confidence"],
            "hit_probability_pct": t.get("hit_probability_pct", 50.0)
        }
        self.trade_history.append(history_record)

        # Snapshot equity history
        self.equity_history.append({
            "timestamp": current_time.strftime("%m-%d %H:%M"),
            "equity_usdt": round(self.balance_usdt, 2)
        })

        self.active_trade = None
        self.active_alert = None

    def update_tick(self, live_price: float, current_time: datetime):
        """
        Evaluate current state on every live ticker tick:
        1. Check liquidation threshold
        2. Check period completion
        3. If no active trade, open next trade
        """
        if self.is_liquidated:
            return

        if self.active_trade:
            t = self.active_trade
            direction = t["direction"]
            liq_price = t["liquidation_price"]
            end_time = datetime.fromisoformat(t["end_time"])

            # 1. Check Liquidation
            if direction == "LONG" and live_price <= liq_price:
                print(f"[!] {self.period_id.upper()} LONG position LIQUIDATED at ${live_price:,.2f} (Liq: ${liq_price:,.2f})")
                self.close_active_trade(live_price, current_time, reason="LIQUIDATED")
                return
            elif direction == "SHORT" and live_price >= liq_price:
                print(f"[!] {self.period_id.upper()} SHORT position LIQUIDATED at ${live_price:,.2f} (Liq: ${liq_price:,.2f})")
                self.close_active_trade(live_price, current_time, reason="LIQUIDATED")
                return

            # 2. Check Period Expiration
            if current_time >= end_time:
                print(f"[*] {self.period_id.upper()} period completed ({self.hours}h). Closing trade and opening next period...")
                self.close_active_trade(live_price, current_time, reason="CLOSED")
                self.open_new_trade(live_price, current_time)
        # else: flat -> wait for update_live_signals to open when hit prob > MIN_ENTRY_HIT_PROB


class MultiPeriodTradingManager:
    """
    Coordinates all 5 period trading strategies in parallel.
    Maintains continuous background loop, live updates, and persistence.
    """
    def __init__(self):
        self.trackers: Dict[str, PeriodStrategyTracker] = {}
        self.alerts: List[Dict[str, Any]] = []
        for pid, cfg in PERIOD_CONFIGS.items():
            self.trackers[pid] = PeriodStrategyTracker(pid, cfg, manager=self)

        self.latest_live_price = 63700.0
        self.last_update_utc = datetime.now(timezone.utc)
        self.is_running = False

        if STATE_FILE.exists():
            self._load_state_if_exists()
        else:
            self.reset_all(clean_start=True)

    def record_alert(self, alert: Dict[str, Any]):
        """Store newly triggered alert and ensure bounded buffer size."""
        if not any(a.get("alert_id") == alert.get("alert_id") for a in self.alerts):
            self.alerts.append(alert)
            if len(self.alerts) > 50:
                self.alerts.pop(0)

    def dismiss_alert(self, alert_id: str) -> bool:
        """Mark an alert as dismissed in buffer and clear it from active tracker."""
        updated = False
        for a in self.alerts:
            if a.get("alert_id") == alert_id:
                a["dismissed"] = True
                updated = True
        for tr in self.trackers.values():
            if tr.active_alert and tr.active_alert.get("alert_id") == alert_id:
                tr.active_alert = None
                updated = True
        if updated:
            self.save_state()
        return updated

    def clear_alerts(self):
        """Clear all stored alerts and tracker active alerts."""
        self.alerts = []
        for tr in self.trackers.values():
            tr.active_alert = None
        self.save_state()

    def update_live_signals(self, live_price: float, current_time: datetime):
        """
        Continuously evaluate fresh model inferences across all periods:
        1. Updates real-time live hit probabilities and confidence.
        2. Evaluates direction reversal alerts for high-conviction signals (>=60%).
        """
        for pid, tr in self.trackers.items():
            if tr.is_liquidated:
                continue
            try:
                model_result = period_model_signal(tr.hours)
                prob_up = float(model_result.get("prob_up", 50.0))
                prob_down = float(model_result.get("prob_down", 50.0))
                tr.live_prob_up = round(prob_up, 1)
                tr.live_prob_down = round(prob_down, 1)
                tr.live_confidence = model_result.get("confidence", "Moderate Conviction")
                tr.last_live_eval_time = current_time

                action_label = str(model_result.get("action", "")).upper()
                has_actionable_signal = "WAIT" not in action_label
                dir_now = "LONG" if prob_up >= 50.0 else "SHORT"
                hit_prob_now = prob_up if dir_now == "LONG" else prob_down

                if tr.active_trade:
                    at = tr.active_trade
                    dir_active = at["direction"]
                    hrs_left = (datetime.fromisoformat(at["end_time"]) - current_time).total_seconds() / 3600.0
                    anch = trade_end_hit_prob(model_result, dir_active, at["entry_price"], live_price, hrs_left, tr.hours)
                    tr.live_prob_up, tr.live_prob_down = anch["prob_up"], anch["prob_down"]
                    tr.live_hit_prob = anch["prob_up"] if dir_active == "LONG" else anch["prob_down"]
                    # Exit gate: probability of finishing the period on the right side collapsed
                    if tr.live_hit_prob < EXIT_HIT_PROB:
                        print(f"[*] {pid.upper()} closing {dir_active}: end-of-period hit prob {tr.live_hit_prob:.1f}% < {EXIT_HIT_PROB}%")
                        tr.close_active_trade(live_price, current_time, reason="CLOSED", exit_reason="PROB_EXIT")
                        # Block an immediate re-entry at the same (bad) price with a new timer.
                        tr.reentry_blocked_until = current_time + timedelta(minutes=max(15, tr.hours * 15))
                if not tr.active_trade:
                    tr.live_hit_prob = round(hit_prob_now, 1)
                    # Flat: open a trade only with an actionable, high-conviction signal
                    if has_actionable_signal and hit_prob_now > MIN_ENTRY_HIT_PROB:
                        if tr.open_new_trade(live_price, current_time, model_result):
                            print(f"[+] {pid.upper()} opened {dir_now} at hit prob {hit_prob_now:.1f}% (> {MIN_ENTRY_HIT_PROB}%)")
                            continue

                # Evaluate alerts only for actionable model signals. The live
                # (anchored) win probability is attached for context so alert
                # text cannot contradict the probability shown on the dashboard.
                if has_actionable_signal:
                    alert = tr.evaluate_alert(dir_now, hit_prob_now, live_price, current_time,
                                              live_hit_prob=tr.live_hit_prob)
                    if alert:
                        self.record_alert(alert)
            except Exception as e:
                print(f"[!] {pid.upper()} live signal update failed: {e}")

    evaluate_live_alerts = update_live_signals

    def reset_all(self, clean_start: bool = True):
        """
        Reset all 5 period trackers to exactly 1,000.00 USDT initial balance.
        - Sets balance to 1,000 USDT
        - Completely clears any previous history (clean start from this exact moment)
        - Synchronizes latest market candles & GDELT news into SQLite
        - Stays flat; a new trade opens only when hit prob > MIN_ENTRY_HIT_PROB
        - Records exact start timestamps
        - Persists fresh state to live_trading_state.json
        """
        now = datetime.now(timezone.utc)
        self.latest_live_price = fetch_live_btc_price_fast()
        self.last_update_utc = now
        self.alerts = []

        print(f"[*] Resetting all period engines to 1,000.00 USDT at {now.strftime('%Y-%m-%d %H:%M:%S UTC')} (BTC: ${self.latest_live_price:,.2f})...")
        try:
            sync_latest_market_and_news_data(force=True)
        except Exception as sync_e:
            print(f"[!] Pre-reset sync note: {sync_e}")

        for pid, tr in self.trackers.items():
            tr.balance_usdt = 1000.0
            tr.initial_usdt = 1000.0
            tr.total_realized_pnl_usdt = 0.0
            tr.total_fees_paid_usdt = 0.0
            tr.is_liquidated = False
            tr.active_trade = None
            tr.last_signal_direction = None
            tr.last_signal_hit_prob = None
            tr.active_alert = None
            tr.last_alert_signature = None
            tr.last_alert_time = None
            tr.recent_alert_times = {}
            tr.reentry_blocked_until = None
            tr.trade_history = []
            tr.equity_history = [{
                "timestamp": now.strftime("%m-%d %H:%M"),
                "equity_usdt": 1000.0
            }]
            tr.live_hit_prob = None
            # Fresh trade opens on its own once hit prob > MIN_ENTRY_HIT_PROB

        self.save_state()
        print("[+] All periods successfully reset to 1,000.00 USDT with exact start timestamps.")

    def _load_state_if_exists(self):
        """Restore active state from disk if available."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    saved = json.load(f)
                    self.alerts = saved.get("alerts", [])
                    for pid, data in saved.get("periods", {}).items():
                        if pid in self.trackers:
                            tr = self.trackers[pid]
                            if "balance_usdt" in data:
                                tr.balance_usdt = data["balance_usdt"]
                                tr.total_realized_pnl_usdt = data.get("total_realized_pnl_usdt", 0.0)
                                tr.total_fees_paid_usdt = data.get("total_fees_paid_usdt", 0.0)
                            elif "balance_btc" in data:
                                # Migrate legacy BTC state to USDT
                                tr.balance_usdt = 1000.0
                                tr.total_realized_pnl_usdt = 0.0
                                tr.total_fees_paid_usdt = 0.0
                            tr.is_liquidated = data.get("is_liquidated", False)
                            tr.last_signal_direction = data.get("last_signal_direction")
                            tr.last_signal_hit_prob = data.get("last_signal_hit_prob")
                            tr.live_hit_prob = data.get("live_hit_prob")
                            tr.live_prob_up = data.get("live_prob_up")
                            tr.live_prob_down = data.get("live_prob_down")
                            tr.live_confidence = data.get("live_confidence")
                            tr.active_alert = data.get("active_alert")
                            tr.last_alert_time = _parse_state_datetime(data.get("last_alert_time"))
                            tr.reentry_blocked_until = _parse_state_datetime(data.get("reentry_blocked_until"))
                            sig = data.get("last_alert_signature")
                            tr.last_alert_signature = tuple(sig) if isinstance(sig, (list, tuple)) and len(sig) == 4 else None
                            if data.get("active_trade"):
                                tr.active_trade = data["active_trade"]
                            if tr.active_trade:
                                if tr.last_signal_direction is None:
                                    tr.last_signal_direction = tr.active_trade.get("direction")
                                if tr.last_signal_hit_prob is None:
                                    tr.last_signal_hit_prob = tr.active_trade.get("hit_probability_pct", 50.0)
                                if tr.live_hit_prob is None:
                                    tr.live_hit_prob = tr.active_trade.get("entry_hit_probability_pct", tr.active_trade.get("hit_probability_pct", 50.0))
                                if tr.live_prob_up is None:
                                    tr.live_prob_up = tr.active_trade.get("prob_up", 50.0)
                                if tr.live_prob_down is None:
                                    tr.live_prob_down = tr.active_trade.get("prob_down", 50.0)
                                if tr.live_confidence is None:
                                    tr.live_confidence = tr.active_trade.get("confidence", "High Conviction")
                            if data.get("trade_history"):
                                tr.trade_history = data["trade_history"]
                            if data.get("equity_history"):
                                tr.equity_history = data["equity_history"]
                print("[+] Loaded persistent trading state from file.")
            except Exception as e:
                print(f"[!] Warning: Failed to load state: {e}")

    def save_state(self):
        """Persist state to JSON file."""
        try:
            payload = {
                "last_update_utc": self.last_update_utc.isoformat(),
                "latest_live_price": self.latest_live_price,
                "alerts": self.alerts[-50:],
                "periods": {
                    pid: {
                        "balance_usdt": tr.balance_usdt,
                        "total_realized_pnl_usdt": tr.total_realized_pnl_usdt,
                        "total_fees_paid_usdt": tr.total_fees_paid_usdt,
                        "is_liquidated": tr.is_liquidated,
                        "active_trade": tr.active_trade,
                        "last_signal_direction": tr.last_signal_direction,
                        "last_signal_hit_prob": tr.last_signal_hit_prob,
                        "live_hit_prob": tr.live_hit_prob,
                        "live_prob_up": tr.live_prob_up,
                        "live_prob_down": tr.live_prob_down,
                        "live_confidence": tr.live_confidence,
                        "active_alert": tr.active_alert,
                        "last_alert_signature": list(tr.last_alert_signature) if tr.last_alert_signature else None,
                        "last_alert_time": tr.last_alert_time.isoformat() if tr.last_alert_time else None,
                        "reentry_blocked_until": tr.reentry_blocked_until.isoformat() if tr.reentry_blocked_until else None,
                        "trade_history": tr.trade_history[-25:],
                        "equity_history": tr.equity_history[-50:]
                    }
                    for pid, tr in self.trackers.items()
                }
            }
            with open(STATE_FILE, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[!] Warning: Failed to save state: {e}")

    def process_tick(self, live_price: float):
        """Update live price and trigger tick checks across all 5 period trackers."""
        self.latest_live_price = live_price
        now = datetime.now(timezone.utc)
        self.last_update_utc = now

        for pid, tracker in self.trackers.items():
            tracker.update_tick(live_price, now)

    def get_full_dashboard_data(self) -> Dict[str, Any]:
        """Produce real-time aggregated metrics across all 5 strategies."""
        periods_data = {
            pid: tr.to_dict(self.latest_live_price)
            for pid, tr in self.trackers.items()
        }

        # Calculate Max Balance and Max Net Return across the 5 types
        best_balance_pid = max(periods_data.keys(), key=lambda pid: periods_data[pid]["current_equity_usdt"])
        max_balance_usdt = periods_data[best_balance_pid]["current_equity_usdt"]
        max_balance_period = periods_data[best_balance_pid]["name"]

        best_return_pid = max(periods_data.keys(), key=lambda pid: periods_data[pid]["total_return_pct"])
        max_return_pct = periods_data[best_return_pid]["total_return_pct"]
        max_return_period = periods_data[best_return_pid]["name"]

        all_closed_trades = []
        for p in periods_data.values():
            all_closed_trades.extend(p["recent_trades"])
        all_closed_trades.sort(key=lambda x: x.get("exit_time", ""), reverse=True)

        return {
            "live_btc_price": round(self.latest_live_price, 2),
            "timestamp_utc": self.last_update_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "max_balance_usdt": round(max_balance_usdt, 2),
            "max_balance_period": max_balance_period,
            "max_return_pct": round(max_return_pct, 2),
            "max_return_period": max_return_period,
            "periods": periods_data,
            "recent_global_trades": all_closed_trades[:15],
            "alerts": list(reversed(self.alerts[-20:])),
            "active_alerts_count": len([a for a in self.alerts if not a.get("dismissed", False)]),
        }


# Singleton Trading Manager Instance
manager = MultiPeriodTradingManager()


async def background_market_watcher(poll_interval: float = 2.5):
    """
    Continuous background loop that runs while the server is active:
    Polls live Binance price every ~2.5 seconds, updates 4x leverage PnL,
    evaluates liquidations, and closes/opens period trades.
    Synchronizes GDELT news and Binance candles every 15 minutes.
    Evaluates live directional signals & hit probabilities every 30 seconds.
    Refreshes every period's model from the growing historical database every
    MODEL_RETRAIN_INTERVAL_HOURS (and once shortly after startup).
    """
    print("[*] Started background market watcher loop...")
    manager.is_running = True
    tick_count = 0
    _evaluating_live_signals = False
    _retraining = False
    # Seed from the oldest period artifact's training time. If all artifacts
    # are already fresh (e.g. just retrained via --train-all-periods) the first
    # background refresh waits the full interval. A stale/missing artifact
    # yields 0.0, forcing a refresh shortly after startup on fresh installs.
    def _artifact_freshness_ts() -> float:
        try:
            import joblib
            from config import LIVE_PERIOD_X_DAYS
            from train import get_x_period_artifact_path
            stamps = []
            for x_days in LIVE_PERIOD_X_DAYS.values():
                path = get_x_period_artifact_path(x_days)
                if not os.path.exists(path):
                    return 0.0
                stamps.append(float(joblib.load(path).get("trained_at", 0.0)))
            return min(stamps) if stamps else 0.0
        except Exception:
            return 0.0

    _last_retrain_ts = _artifact_freshness_ts()

    async def _run_live_eval(price: float, dt: datetime):
        nonlocal _evaluating_live_signals
        try:
            await asyncio.to_thread(manager.update_live_signals, price, dt)
        except Exception as e:
            print(f"[!] Live signal eval error: {e}")
        finally:
            _evaluating_live_signals = False

    async def _run_period_retrain():
        nonlocal _retraining
        _retraining = True
        try:
            from train import train_all_periods
            await asyncio.to_thread(train_all_periods)
            print("[+] All period models refreshed from the historical database.")
        except Exception as e:
            print(f"[!] Periodic model retrain failed: {e}")
        finally:
            _retraining = False

    while manager.is_running:
        try:
            live_price = fetch_live_btc_price_fast()
            manager.process_tick(live_price)
            manager.save_state()

            tick_count += 1

            # On tick 1 (startup) and every 12 ticks (~30s), trigger non-blocking live hit prob evaluation
            if (tick_count == 1 or tick_count % 12 == 0) and not _evaluating_live_signals:
                _evaluating_live_signals = True
                asyncio.create_task(_run_live_eval(live_price, datetime.now(timezone.utc)))

            # Every 15 minutes (360 ticks * 2.5s = 900s), sync newest GDELT batch & candles
            if tick_count % 360 == 0:
                try:
                    await asyncio.to_thread(sync_latest_market_and_news_data)
                except Exception as sync_err:
                    print(f"[!] Background sync error: {sync_err}")

            # Retrain all period models from the full history. First run is
            # delayed a few ticks so the server is up and data is available.
            if (not _retraining and tick_count >= 5 and
                    time.time() - _last_retrain_ts >= MODEL_RETRAIN_INTERVAL_HOURS * 3600):
                _last_retrain_ts = time.time()
                asyncio.create_task(_run_period_retrain())

        except Exception as e:
            print(f"[!] Error in market watcher loop: {e}")
        await asyncio.sleep(poll_interval)
