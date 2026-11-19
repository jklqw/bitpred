#!/usr/bin/env python3
"""Interactive & CLI Command: Direct Bitcoin Action Recommendation (BUY or SELL).

Uses an X-day period framework where:
  - Input: X (period length in days).
  - Window 1: First X/2 days of this X-day period.
  - Window 2: X/2 days of the previous week (7 days prior).
  - Target: Delta price = Price(end of X days) - Price(start of X day).
  - Output: Direct BUY or SELL recommendation right now with price targets & risk levels.
"""

import sys
from pathlib import Path

# Auto-bootstrap virtualenv site-packages if running with system python
_BASE_DIR = Path(__file__).resolve().parent
_site_packages = _BASE_DIR / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _site_packages.is_dir() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import argparse
from signal_engine import generate_x_period_action, print_x_period_action_dashboard



def get_x_days_input(cli_value: float = None) -> float:
    """Obtain period length X in days via argument or user input prompt."""
    if cli_value is not None:
        return float(cli_value)

    # Only prompt if running interactively with zero arguments
    if len(sys.argv) == 1 and sys.stdin.isatty():
        try:
            user_val = input("\nEnter period length X in days [Press Enter for 4.0]: ").strip()
            if not user_val:
                return 4.0
            return float(user_val)
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nDefaulting to 4.0 days.")
            return 4.0
    return 4.0


def run_history_test(x_days: float = 4.0, num_periods: int = 15):
    """
    Test and audit model predictions on completed historical periods.
    """
    import os
    import joblib
    import numpy as np
    import pandas as pd
    from features import build_x_period_dataset
    from train import get_x_period_artifact_path, train_x_period_pipeline

    artifact_path = get_x_period_artifact_path(x_days)
    if not os.path.exists(artifact_path):
        print(f"[*] Artifact for X={x_days}d not found. Training model now...")
        train_x_period_pipeline(x_days=x_days)

    artifact = joblib.load(artifact_path)
    scaler = artifact["scaler"]
    ensemble = artifact["ensemble"]
    regressor = artifact["regressor"]
    feature_cols = artifact["feature_cols"]
    threshold = artifact.get("optimal_threshold", 0.50)

    print(f"\n[*] Evaluating model on past completed {x_days:g}-day periods...")
    dataset, meta = build_x_period_dataset(x_days=x_days, step_hours=24)

    X = dataset[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X)

    probs = ensemble.predict_proba(X_scaled)[:, 1]
    pred_dirs = (probs >= threshold).astype(int)
    pred_ret = regressor.predict(X_scaled)

    dataset["prob_up"] = np.round(probs * 100.0, 1)
    dataset["pred_dir"] = pred_dirs
    dataset["pred_delta"] = dataset["p_start"] * pred_ret
    dataset["pred_end"] = dataset["p_start"] + dataset["pred_delta"]
    dataset["is_correct"] = (dataset["pred_dir"] == dataset["target_dir"])

    recent = dataset.tail(num_periods).copy()

    print("\n" + "=" * 105)
    print(f"       HISTORICAL PREDICTION AUDIT: LAST {len(recent)} COMPLETED {x_days:g}-DAY PERIODS")
    print("=" * 105)
    print(f"{'Decision Date (UTC)':<18} {'P_Start':<10} {'P_Mid':<10} {'P_End':<10} {'Actual Delta':<14} {'Model Pred':<12} {'Exp Delta':<13} {'Prob Up':<9} {'Result':<8}")
    print("-" * 105)

    hits = 0
    trade_returns = []

    for idx, row in recent.iterrows():
        is_hit = row["is_correct"]
        if is_hit:
            hits += 1
        hit_str = "✅ HIT " if is_hit else "❌ MISS"
        pred_str = "🟢 BUY " if row["pred_dir"] == 1 else "🔴 SELL"
        actual_delta_str = f"${row['delta_p']:+,.0f} ({row['delta_p_pct']*100:+.1f}%)"
        exp_delta_str = f"${row['pred_delta']:+,.0f}"
        
        # PnL: if BUY, gain delta_p_pct; if SELL, gain -delta_p_pct (short)
        trade_ret = row["delta_p_pct"] if row["pred_dir"] == 1 else -row["delta_p_pct"]
        trade_returns.append(trade_ret)

        print(f"{idx.strftime('%Y-%m-%d %H:%M'):<18} ${row['p_start']:<9,.0f} ${row['p_curr']:<9,.0f} ${row['p_end']:<9,.0f} {actual_delta_str:<14} {pred_str:<12} {exp_delta_str:<13} {row['prob_up']:>5.1f}%   {hit_str:<8}")

    print("=" * 105)
    hit_rate = (hits / len(recent)) * 100.0 if len(recent) > 0 else 0.0
    cum_strat = (np.prod(1.0 + np.array(trade_returns)) - 1.0) * 100.0
    cum_bh = ((recent['p_end'].iloc[-1] - recent['p_start'].iloc[0]) / recent['p_start'].iloc[0]) * 100.0

    print(f"📊 SUMMARY PERFORMANCE ({len(recent)} PERIODS):")
    print(f"  • Prediction Hit Rate         : {hits}/{len(recent)} ({hit_rate:.1f}%)")
    print(f"  • Strategy Return (Long/Short): {cum_strat:+.2f}%")
    print(f"  • Benchmark Buy & Hold Return : {cum_bh:+.2f}%")
    print("=" * 105 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bitcoin Real-Time Action Recommendation (BUY or SELL)"
    )
    parser.add_argument(
        "positional_x",
        type=float,
        nargs="?",
        default=None,
        help="Optional period length X in days (e.g. 4 or 7)"
    )
    parser.add_argument(
        "--x-days", "-x",
        type=float,
        default=None,
        help="Period length X in days (default: 4.0)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force re-train model for this X period before generating action"
    )
    parser.add_argument(
        "--history", "--test-history",
        type=int,
        nargs="?",
        const=15,
        default=None,
        help="Test and audit model predictions across the last N completed periods (default: 15)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset live 4x trading server to 1.0 BTC clean start from right now"
    )

    args = parser.parse_args()

    if args.reset:
        from reset import reset_trading_server
        reset_trading_server()
        return

    # Determine X
    raw_x = args.x_days if args.x_days is not None else args.positional_x
    x_days = get_x_days_input(raw_x)

    if x_days <= 0:
        print("[!] Error: X days must be positive (e.g. 2, 4, 7).")
        sys.exit(1)

    if args.train:
        from train import train_x_period_pipeline
        print(f"[*] Force training model for X = {x_days:g} days...")
        train_x_period_pipeline(x_days=x_days)

    if args.history is not None:
        run_history_test(x_days=x_days, num_periods=args.history)
        return

    print(f"\n[*] Analyzing live Bitcoin market & GDELT news for {x_days:g}-day period...")
    action_data = generate_x_period_action(x_days=x_days)
    print_x_period_action_dashboard(action_data)


if __name__ == "__main__":
    main()

