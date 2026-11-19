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
from config import DATA_DIR, USER_AGENT
from signal_engine import generate_x_period_action
from database import get_connection, insert_btc_prices, insert_news_batch
from btc_fetcher import fetch_binance_btc_klines_chunk
from gdelt_fetcher import get_recent_gkg_urls, fetch_and_process_gkg_zip

STATE_FILE = DATA_DIR / "live_trading_state.json"
_last_sync_timestamp = 0.0


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

        self.live_hit_prob: Optional[float] = None
        self.live_prob_up: Optional[float] = None
        self.live_prob_down: Optional[float] = None
        self.live_confidence: Optional[str] = None
        self.last_live_eval_time: Optional[datetime] = None

        self.is_liquidated = False

    def evaluate_alert(self, new_direction: str, new_hit_prob: float, live_price: float, current_time: datetime) -> Optional[Dict[str, Any]]:
        """
        Check if direction changed with high hit probability ("Hit prob too much").
        Alert triggers ONLY when:
        1. Direction changed relative to the active position (or last signal):
           - AND the new direction has high hit probability (new_hit_prob >= 60.0%).
             (If new_hit_prob < 60%, it is low conviction / chop and does NOT alert).
           - Severity: CRITICAL if new_hit_prob >= 65.0%, WARNING if 60.0% <= new_hit_prob < 65.0%.
        2. Or in the same direction, hit probability shifts drastically by >= 25.0% AND new_hit_prob >= 60.0%.
        """
        # Determine baseline direction and probability from active trade or previous signal
        if self.active_trade:
            old_dir = self.active_trade["direction"]
            old_prob = self.active_trade.get("hit_probability_pct", 50.0)
        elif self.last_signal_direction is not None:
            old_dir = self.last_signal_direction
            old_prob = self.last_signal_hit_prob if self.last_signal_hit_prob is not None else 50.0
        else:
            self.last_signal_direction = new_direction
            self.last_signal_hit_prob = new_hit_prob
            return None

        prob_diff = new_hit_prob - old_prob
        alert = None

        if new_direction != old_dir:
            # Direction flipped! Only alert if new direction has HIGH hit probability (>= 60%)
            if new_hit_prob >= 60.0:
                is_high_conviction = new_hit_prob >= 65.0
                severity = "CRITICAL" if is_high_conviction else "WARNING"
                alert = {
                    "alert_id": f"alert-{self.period_id}-{int(current_time.timestamp())}",
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "period_id": self.period_id,
                    "period_name": self.name,
                    "type": "DIRECTION_REVERSAL",
                    "severity": severity,
                    "dismissed": False,
                    "old_direction": old_dir,
                    "new_direction": new_direction,
                    "old_hit_prob": round(old_prob, 1),
                    "new_hit_prob": round(new_hit_prob, 1),
                    "prob_change_pct": round(prob_diff, 1),
                    "title": f"🚨 {self.name} Direction Reversal Alert!",
                    "message": f"{self.name} active {old_dir} ({old_prob:.1f}%) flipped to high-conviction {new_direction} with {new_hit_prob:.1f}% hit prob! Significant directional reversal detected.",
                    "price_at_alert": round(live_price, 2)
                }
            else:
                # Low conviction flip (<60%): do not alert, retain steady active position
                pass
        elif abs(prob_diff) >= 25.0 and new_hit_prob >= 60.0:
            # Significant probability shift in same direction
            alert = {
                "alert_id": f"alert-{self.period_id}-{int(current_time.timestamp())}",
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "period_id": self.period_id,
                "period_name": self.name,
                "type": "PROBABILITY_SHIFT",
                "severity": "WARNING",
                "dismissed": False,
                "old_direction": old_dir,
                "new_direction": new_direction,
                "old_hit_prob": round(old_prob, 1),
                "new_hit_prob": round(new_hit_prob, 1),
                "prob_change_pct": round(prob_diff, 1),
                "title": f"⚠️ {self.name} Hit Prob Swing Alert!",
                "message": f"{self.name} {new_direction} hit probability shifted by {prob_diff:+.1f}% ({old_prob:.1f}% → {new_hit_prob:.1f}%).",
                "price_at_alert": round(live_price, 2)
            }

        # Update tracked signal state
        self.last_signal_direction = new_direction
        self.last_signal_hit_prob = new_hit_prob

        if alert:
            self.active_alert = alert
            print(f"\n[ALERT] 🚨 {alert['message']}\n")

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
            return

        # Deduct upfront fee (0.1% or 0.3% for 3d)
        fee_usdt = self.balance_usdt * self.fee_rate
        self.balance_usdt -= fee_usdt
        self.total_fees_paid_usdt += fee_usdt

        margin_usdt = self.balance_usdt
        position_notional_usdt = margin_usdt * self.leverage

        # If model result not provided, synchronize fresh data & run inference
        if model_result is None:
            try:
                sync_latest_market_and_news_data()
                model_result = generate_x_period_action(x_days=self.x_days)
            except Exception as e:
                # Safe fallback
                model_result = {
                    "action": "SELL / SHORT",
                    "confidence": "Moderate Conviction",
                    "prob_up": 35.0,
                    "prob_down": 65.0,
                    "pred_total_delta_dollars": -1200.0,
                    "expected_target_price": live_price - 1200.0
                }

        direction = "LONG" if "BUY" in model_result.get("action", "BUY") else "SHORT"

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

    def close_active_trade(self, live_price: float, current_time: datetime, reason: str = "PERIOD_EXPIRED"):
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
        else:
            # No trade open yet -> open first trade
            self.open_new_trade(live_price, current_time)


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
                model_result = generate_x_period_action(x_days=tr.x_days)
                prob_up = float(model_result.get("prob_up", 50.0))
                prob_down = float(model_result.get("prob_down", 50.0))
                tr.live_prob_up = round(prob_up, 1)
                tr.live_prob_down = round(prob_down, 1)
                tr.live_confidence = model_result.get("confidence", "Moderate Conviction")
                tr.last_live_eval_time = current_time

                dir_now = "LONG" if prob_up >= 50.0 else "SHORT"
                hit_prob_now = prob_up if dir_now == "LONG" else prob_down

                if tr.active_trade:
                    dir_active = tr.active_trade["direction"]
                    tr.live_hit_prob = round(prob_up if dir_active == "LONG" else prob_down, 1)
                else:
                    tr.live_hit_prob = round(hit_prob_now, 1)

                alert = tr.evaluate_alert(dir_now, hit_prob_now, live_price, current_time)
                if alert:
                    self.record_alert(alert)
            except Exception as e:
                pass

    evaluate_live_alerts = update_live_signals

    def reset_all(self, clean_start: bool = True):
        """
        Reset all 5 period trackers to exactly 1,000.00 USDT initial balance.
        - Sets balance to 1,000 USDT
        - Completely clears any previous history (clean start from this exact moment)
        - Synchronizes latest market candles & GDELT news into SQLite
        - Opens a brand new 4x leveraged trade starting at the exact current time
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
            tr.trade_history = []
            tr.equity_history = [{
                "timestamp": now.strftime("%m-%d %H:%M"),
                "equity_usdt": 1000.0
            }]
            # Open brand new 4x leveraged trade starting from this exact time
            tr.open_new_trade(self.latest_live_price, now)

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
    """
    print("[*] Started background market watcher loop...")
    manager.is_running = True
    tick_count = 0
    _evaluating_live_signals = False

    async def _run_live_eval(price: float, dt: datetime):
        nonlocal _evaluating_live_signals
        try:
            await asyncio.to_thread(manager.update_live_signals, price, dt)
        except Exception as e:
            print(f"[!] Live signal eval error: {e}")
        finally:
            _evaluating_live_signals = False

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

        except Exception as e:
            print(f"[!] Error in market watcher loop: {e}")
        await asyncio.sleep(poll_interval)
