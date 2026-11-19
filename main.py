"""Unified CLI and Pipeline Runner for GDELT News & Bitcoin Price Prediction."""

import argparse
import sys
import pandas as pd
from config import (
    DB_PATH,
    DATA_DIR,
    DEFAULT_INTERVAL,
    DEFAULT_LOOKBACK_CANDLES,
    DEFAULT_PREDICTION_HORIZON,
    DEFAULT_TARGET_THRESHOLD,
    DEFAULT_TAKER_FEE,
    DEFAULT_SLIPPAGE
)
from database import init_db, get_database_stats
from gdelt_fetcher import ingest_recent_gdelt_news
from btc_fetcher import ingest_btc_market_data


def run_stats():
    """Print comprehensive database summary, news categories, and coverage."""
    stats = get_database_stats()
    print("\n" + "=" * 70)
    print("           GDELT NEWS & BITCOIN DATABASE SUMMARY")
    print("=" * 70)
    print(f"Database Path       : {DB_PATH}")
    print(f"Total News Articles : {stats['total_news']}")
    print(f"News Date Range     : {stats['news_date_range'][0]} to {stats['news_date_range'][1]}")
    print(f"Total BTC Candles   : {stats['btc_price_count']}")
    print(f"BTC Date Range      : {stats['btc_date_range'][0]} to {stats['btc_date_range'][1]}")
    print("\n--- CATEGORY BREAKDOWN ---")
    print(f"{'Category':<28} {'Articles':<10} {'Avg Tone':<10}")
    print("-" * 55)
    for cat, count, avg_tone in stats["categories"]:
        tone_str = f"{avg_tone:+.2f}" if avg_tone is not None else "N/A"
        print(f"{cat:<28} {count:<10} {tone_str:<10}")
    print("=" * 70 + "\n")


def run_export():
    """Export SQLite tables to CSV for external analysis or inspections."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))

    articles_csv = DATA_DIR / "gdelt_news_categorized.csv"
    btc_csv = DATA_DIR / "btc_prices_ohlcv.csv"

    print(f"[*] Exporting news articles to {articles_csv}...")
    df_news = pd.read_sql_query("SELECT * FROM news_articles;", conn)
    df_news.to_csv(articles_csv, index=False)

    print(f"[*] Exporting BTC prices to {btc_csv}...")
    df_btc = pd.read_sql_query("SELECT * FROM btc_prices ORDER BY timestamp ASC;", conn)
    df_btc.to_csv(btc_csv, index=False)

    conn.close()
    print(f"[+] Successfully exported {len(df_news)} news articles and {len(df_btc)} BTC candles.")


def run_train(args):
    """Run model training, validation, threshold tuning, and untouched test backtesting."""
    if args.x_days is not None:
        from train import train_x_period_pipeline
        results = train_x_period_pipeline(
            x_days=args.x_days,
            taker_fee=args.taker_fee,
            slippage=args.slippage
        )
        meta = results["metadata"]

        print("\n" + "=" * 85)
        print(f"               X-DAY PERIOD DATASET & FORMULA ({args.x_days:g} DAYS)")
        print("=" * 85)
        print(f"Period Length (X)       : {meta['x_days']} days ({meta['total_period_hours']} hours)")
        print(f"Observed Window (X/2)   : First {meta['x_days']/2.0:g} days ({meta['half_period_hours']} hours) of this period")
        print(f"Previous Week Window    : Corresponding {meta['x_days']/2.0:g} days from 7 days prior (168h offset)")
        print(f"Target Formula          : Price(end of {meta['x_days']:g} days) - Price(start of {meta['x_days']:g} days)")
        print(f"Total Period Samples    : {meta['total_samples']}")
        print(f"Date Range              : {meta['date_start']}  to  {meta['date_end']}")
        print(f"Upward Move Ratio       : {meta['up_ratio']*100:.2f}%")

        print("\n" + "=" * 85)
        print("           UNTOUCHED TEST SET RESULTS (Final Out-of-Sample Evaluation)")
        print("=" * 85)
        test_cols = ["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "Net_Return", "BH_Net_Return", "Sharpe_Ratio", "Max_Drawdown", "Win_Rate", "Total_Trades", "Confusion_Matrix"]
        test_disp = results["test_results"][[c for c in test_cols if c in results["test_results"].columns]]
        print(test_disp.to_string())

        print("\n" + "-" * 85)
        print(f"★ High-Conviction Filtered Accuracy: {results['high_conv_acc']*100:.2f}% ({results['high_conv_count']} high-conviction trades)")
        print(f"★ Tuned Optimal Classification Threshold: {results['optimal_threshold']:.2f}")
        print("-" * 85)

        print("\n" + "=" * 85)
        print("              TOP 15 MOST INFLUENTIAL FEATURES (RF Importance)")
        print("=" * 85)
        fi = results["feature_importance"]
        print(fi.head(15).to_string(index=False))
        return

    from train import train_and_evaluate_pipeline

    results = train_and_evaluate_pipeline(
        freq=DEFAULT_INTERVAL,
        prediction_horizon=args.horizon,
        target_threshold=args.target_threshold,
        overlap_only=args.overlap_only,
        run_walk_forward=not args.no_walk_forward,
        taker_fee=args.taker_fee,
        slippage=args.slippage
    )

    meta = results["metadata"]
    bal = results["balance_info"]

    print("\n" + "=" * 85)
    print("                         DATASET AUDIT & PARTITIONS")
    print("=" * 85)
    print(f"Total Samples (candles) : {meta['total_samples']}")
    print(f"Date Range              : {meta['date_start']}  to  {meta['date_end']}")
    print(f"Prediction Horizon      : {meta['prediction_horizon']} candle(s) forward")
    print(f"Target Return Threshold : {meta['target_threshold']}")
    print(f"News Articles in DB     : {meta['news_articles_in_db']}")
    print(f"News Date Range in DB   : {meta['news_min_date']} to {meta['news_max_date']}")
    print(f"Training Partition      : {bal['Train_Size']} samples ({bal['Train_Positive_Pct']}% Up)")
    print(f"Validation Partition    : {bal['Val_Size']} samples ({bal['Val_Positive_Pct']}% Up)")
    print(f"Untouched Test Partition: {bal['Test_Size']} samples ({bal['Test_Positive_Pct']}% Up)")

    if args.diagnose_features:
        print("\n" + "=" * 85)
        print("                  FEATURE COVERAGE & NON-ZERO RATES (Top 25)")
        print("=" * 85)
        cov = results["coverage_report"]
        print(cov.head(25).to_string(index=False))

    if results["walk_forward_results"] is not None:
        print("\n" + "=" * 85)
        print("           WALK-FORWARD CROSS VALIDATION (Expanding Dev Window)")
        print("=" * 85)
        print(results["walk_forward_results"].to_string(index=False))

    print("\n" + "=" * 85)
    print("             VALIDATION SET RESULTS (Model Selection & Tuning)")
    print("=" * 85)
    val_cols = ["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "Net_Return", "Sharpe_Ratio", "Max_Drawdown", "Total_Trades"]
    val_disp = results["validation_results"][[c for c in val_cols if c in results["validation_results"].columns]]
    print(val_disp.to_string())

    print("\n" + "=" * 85)
    print("           UNTOUCHED TEST SET RESULTS (Final Out-of-Sample Evaluation)")
    print("=" * 85)
    test_cols = ["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "Net_Return", "BH_Net_Return", "Sharpe_Ratio", "Max_Drawdown", "Win_Rate", "Total_Trades", "Confusion_Matrix"]
    test_disp = results["test_results"][[c for c in test_cols if c in results["test_results"].columns]]
    print(test_disp.to_string())

    print("\n" + "=" * 85)
    print("              TOP 15 MOST INFLUENTIAL FEATURES (RF Importance)")
    print("=" * 85)
    fi = results["feature_importance"]
    print(fi.head(15).to_string(index=False))

    # Diagnostic check for news feature importance
    news_fi = fi[fi["Type"] == "News"]
    max_news_imp = news_fi["RF_Importance"].max() if not news_fi.empty else 0.0
    sum_news_imp = news_fi["RF_Importance"].sum() if not news_fi.empty else 0.0
    print("-" * 85)
    print(f"Total News Features Importance Share: {sum_news_imp * 100:.2f}% (Max single news feature: {max_news_imp * 100:.2f}%)")
    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="GDELT News Database & Bitcoin Price Prediction Pipeline"
    )
    parser.add_argument("--init-db", action="store_true", help="Initialize database schema")
    parser.add_argument("--fetch-btc", type=int, nargs="?", const=DEFAULT_LOOKBACK_CANDLES,
                        help=f"Fetch BTC candles with chunked pagination (default: {DEFAULT_LOOKBACK_CANDLES})")
    parser.add_argument("--fetch-news", type=int, nargs="?", const=4,
                        help="Fetch recent GDELT batches (default: 4 batches = 1 hour)")
    parser.add_argument("--news-sample-hours", type=int,
                        help="Sample 1 GDELT batch per hour across the specified number of historical hours")
    parser.add_argument("--filter-finance", action="store_true",
                        help="Filter GDELT ingestion to finance/crypto/macro/regulation articles only")
    parser.add_argument("--stats", action="store_true",
                        help="Show database statistics and category breakdown")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export database tables to CSV")
    parser.add_argument("--train", action="store_true",
                        help="Train ML models and evaluate on untouched test set")
    parser.add_argument("--x-days", "-x", type=float, default=None,
                        help="Period length X in days for X-day analysis (e.g. 4.0 or 7.0)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_PREDICTION_HORIZON,
                        help=f"Prediction horizon in candles (default: {DEFAULT_PREDICTION_HORIZON})")
    parser.add_argument("--target-threshold", type=float, default=DEFAULT_TARGET_THRESHOLD,
                        help=f"Return threshold for positive class label (default: {DEFAULT_TARGET_THRESHOLD})")
    parser.add_argument("--overlap-only", action="store_true",
                        help="Train strictly on the date window where news data is available")
    parser.add_argument("--no-walk-forward", action="store_true",
                        help="Skip walk-forward cross validation")
    parser.add_argument("--diagnose-features", action="store_true",
                        help="Print feature coverage and non-zero rate diagnostics")
    parser.add_argument("--taker-fee", type=float, default=DEFAULT_TAKER_FEE,
                        help=f"Taker fee per trade side for backtest (default: {DEFAULT_TAKER_FEE})")
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE,
                        help=f"Slippage per trade side for backtest (default: {DEFAULT_SLIPPAGE})")
    parser.add_argument("--action", action="store_true",
                        help="Output direct BUY or SELL action recommendation right now for the configured X-day period")
    parser.add_argument("--signal", "--predict-now", action="store_true", dest="signal",
                        help="Analyze current live market data & news and output BUY, SELL, or WAIT recommendation")
    parser.add_argument("--all", action="store_true",
                        help="Run full end-to-end pipeline: init, fetch BTC (5000), fetch news, report stats, and train")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.init_db or args.all:
        print("[*] Initializing SQLite database...")
        init_db()
        print("[+] Database initialized successfully.")

    if args.fetch_btc is not None or args.all:
        lookback = args.fetch_btc if args.fetch_btc is not None else DEFAULT_LOOKBACK_CANDLES
        ingest_btc_market_data(interval=DEFAULT_INTERVAL, lookback_candles=lookback)

    if args.news_sample_hours is not None:
        ingest_recent_gdelt_news(
            sample_hours=args.news_sample_hours,
            filter_crypto_finance=args.filter_finance
        )
    elif args.fetch_news is not None or args.all:
        batches = args.fetch_news if args.fetch_news is not None else 4
        ingest_recent_gdelt_news(
            num_batches=batches,
            filter_crypto_finance=args.filter_finance
        )

    if args.stats or args.all:
        run_stats()

    if args.export_csv:
        run_export()

    if args.train or args.all:
        run_train(args)

    if args.action or (args.signal and args.x_days is not None):
        from signal_engine import generate_x_period_action, print_x_period_action_dashboard
        x_val = args.x_days if args.x_days is not None else 1.0
        action_data = generate_x_period_action(x_days=x_val)
        print_x_period_action_dashboard(action_data)
    elif args.signal:
        from trading_engine import manager, fetch_live_btc_price_fast
        live_p = fetch_live_btc_price_fast()
        manager.process_tick(live_p)
        print("\n" + "═" * 86)
        print("     ⚡ BITCOIN MULTI-PERIOD ACTION RECOMMENDATION (LIVE SERVER ENGINE)")
        print("═" * 86)
        print(f"  Live Bitcoin Price : ${manager.latest_live_price:,.2f}")
        print(f"  Current Time (UTC) : {manager.last_update_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("─" * 94)
        print(f"  {'PERIOD':<8} {'ACTION':<14} {'LIVE HIT PROB':<18} {'ENTRY HIT PROB':<16} {'CONFIDENCE':<20} {'TARGET PRICE'}")
        print("─" * 94)
        for pid in ["3h", "6h", "12h", "1d", "3d"]:
            p_data = manager.trackers[pid].to_dict(manager.latest_live_price)
            t = p_data.get("active_trade")
            if t:
                act = t.get("direction_icon", "---")
                live_hit = p_data.get("live_hit_probability_pct", 50.0)
                entry_hit = p_data.get("entry_hit_probability_pct", 50.0)
                delta_hit = p_data.get("live_hit_prob_delta", 0.0)
                delta_icon = "▲" if delta_hit >= 0 else "▼"
                delta_str = f"({delta_icon} {delta_hit:+.1f}%)"
                live_str = f"{live_hit:.1f}% {delta_str}"
                entry_str = f"{entry_hit:.1f}%"
                conf = p_data.get("live_confidence", t.get("confidence", "---"))
                target = f"${t.get('target_price', 0):,.2f}"
                print(f"  {pid.upper():<8} {act:<14} {live_str:<18} {entry_str:<16} {conf:<20} {target}")
        print("═" * 94)

        active_alerts = [a for a in manager.alerts if not a.get("dismissed", False)]
        if active_alerts:
            print("\n" + "╔" + "═" * 84 + "╗")
            print("║  🚨 ACTIVE STRATEGY ALERTS (DIRECTION REVERSALS / HIT PROB SWINGS)            ║")
            print("╠" + "═" * 84 + "╣")
            for a in active_alerts[-5:]:
                sev = a.get('severity', 'WARNING')
                pname = a.get('period_name', a.get('period_id'))
                old_d = a.get('old_direction', '---')
                old_p = a.get('old_hit_prob', 0)
                new_d = a.get('new_direction', '---')
                new_p = a.get('new_hit_prob', 0)
                dp = a.get('prob_change_pct', 0)
                px = a.get('price_at_alert', 0)
                msg_str = f"[{sev}] {pname}: {old_d} ({old_p:.1f}%) ➔ {new_d} ({new_p:.1f}%) [Δ {dp:+.1f}%] @ ${px:,.2f}"
                print(f"║  • {msg_str:<80}║")
            print("╚" + "═" * 84 + "╝\n")

        print("  Tip: For detailed single-period dashboard, run: python3 action.py -x <days>")
        print("       (e.g., python3 action.py -x 0.125 for 3h, -x 1 for 1d, -x 3 for 3d)\n")


if __name__ == "__main__":
    main()

