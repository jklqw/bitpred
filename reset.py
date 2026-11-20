#!/usr/bin/env python3
"""CLI Command: Reset Bitcoin 4x Leveraged Server to 1,000 USDT with exact start timestamp."""

import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root and .venv site-packages are in sys.path
BASE_DIR = Path(__file__).resolve().parent
_site_packages = BASE_DIR / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _site_packages.is_dir() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def reset_trading_server():
    from config import MIN_ENTRY_HIT_PROB

    server_port = 8000
    server_url = f"http://127.0.0.1:{server_port}/api/reset"
    server_running = False

    # Prefer the live server: a running web_server keeps its own in-memory
    # manager and would otherwise overwrite a disk-only reset within 2.5s.
    try:
        req = urllib.request.Request(
            server_url,
            data=b"",
            method="POST",
            headers={"User-Agent": "BTC-Reset-CLI"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        server_running = True
        exact_time = data.get("exact_start_time_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
        btc_price = data.get("live_btc_price", 0.0)
        portfolio = data.get("portfolio", {})
    except Exception as e:
        print(f"[!] Live dashboard not reachable ({e}); resetting persisted engine state directly.")
        from trading_engine import manager
        manager.reset_all(clean_start=True)
        exact_time = manager.last_update_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        btc_price = manager.latest_live_price
        portfolio = manager.get_full_dashboard_data()

    print("\n" + "═" * 86)
    print("  ⚡ BITCOIN 4X LEVERAGED STRATEGY RESET COMPLETE — CLEAN START")
    print("═" * 86)
    print(f"  🕒 Exact Start Time (UTC)  : {exact_time}")
    print(f"  🕒 Exact Start Time (Local): {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  💰 Live Bitcoin Price      : ${btc_price:,.2f}")
    print(f"  📦 Capital per Period      : 1,000.00 USDT (5 Parallel Strategies)")
    print(f"  🌐 Live Web Dashboard      : http://localhost:{server_port}")
    if server_running:
        print("  🟢 Server Status           : Active & Synchronized with Web Dashboard")
    else:
        print("  🟡 Note                    : Saved to disk. (Re)start web server with: python3 web_server.py")
    print("─" * 86)
    print(f"  {'PERIOD':<8} {'DIR':<10} {'ENTRY (+/-0.01%)':<18} {'LIQUIDATION':<16} {'FEE DEDUCTED':<16} {'HIT PROB':<10} {'CONFIDENCE'}")
    print("─" * 86)

    periods = portfolio.get("periods", {})
    for pid in ["3h", "6h", "12h", "1d", "3d"]:
        p = periods.get(pid, {})
        t = p.get("active_trade")
        if t:
            dir_str = t.get("direction_icon", t.get("direction"))
            entry_str = f"${t.get('entry_price', 0):,.2f}"
            liq_str = f"${t.get('liquidation_price', 0):,.2f}"
            fee_val = t.get("fee_deducted_usdt", t.get("fee_usdt", 0))
            fee_str = f"${fee_val:,.2f} USDT"
            hit_prob = t.get("hit_probability_pct", t.get("prob_up" if "LONG" in dir_str else "prob_down", 50.0))
            hit_str = f"{hit_prob:.1f}%"
            conf_str = t.get("confidence", "Moderate")
            print(f"  {pid.upper():<8} {dir_str:<10} {entry_str:<18} {liq_str:<16} {fee_str:<16} {hit_str:<10} {conf_str}")
        else:
            print(f"  {pid.upper():<8} waiting for signal: trade opens when hit prob > {MIN_ENTRY_HIT_PROB:g}%")

    print("═" * 86)
    print("  All past trades have been wiped. Monitoring and live PnL begin from this exact second.\n")


if __name__ == "__main__":
    reset_trading_server()
