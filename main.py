"""Unified CLI and Pipeline Runner for GDELT News Database and Bitcoin Prediction."""

import argparse
import sys
import pandas as pd
from config import DB_PATH, DATA_DIR
from database import init_db, get_database_stats
from gdelt_fetcher import ingest_recent_gdelt_news
from btc_fetcher import ingest_btc_market_data
from features import build_ml_dataset


def run_stats():
    """Print database statistics."""
    stats = get_database_stats()
    print("\n" + "=" * 65)
    print("           GDELT NEWS & BITCOIN DATABASE SUMMARY")
    print("=" * 65)
    print(f"Database File: {DB_PATH}")
    print(f"Total News Articles: {stats['total_news']}")
    print(f"News Date Range    : {stats['news_date_range'][0]} to {stats['news_date_range'][1]}")
    print(f"Total BTC Candles  : {stats['btc_price_count']}")
    print(f"BTC Date Range     : {stats['btc_date_range'][0]} to {stats['btc_date_range'][1]}")
    print("\n--- CATEGORY BREAKDOWN ---")
    print(f"{'Category':<28} {'Articles':<10} {'Avg Tone':<10}")
    print("-" * 50)
    for cat, count, avg_tone in stats["categories"]:
        tone_str = f"{avg_tone:+.2f}" if avg_tone is not None else "N/A"
        print(f"{cat:<28} {count:<10} {tone_str:<10}")
    print("=" * 65 + "\n")


def run_export():
    """Export SQLite tables to CSV for easy analysis."""
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
    print(f"[+] Successfully exported {len(df_news)} articles and {len(df_btc)} BTC candles.")


def run_train():
    """Train machine learning models and evaluate performance."""
    from train import train_and_evaluate_models
    results = train_and_evaluate_models(freq="1h")

    print("\n" + "=" * 75)
    print("                 MODEL COMPARISON & BACKTEST REPORT")
    print("=" * 75)
    print(results["model_results"].to_string())

    print("\n" + "=" * 75)
    print("             TOP 15 MOST PREDICTIVE FEATURES (by RF Importance)")
    print("=" * 75)
    print(results["feature_importance"].head(15).to_string(index=False))
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="GDELT News Database & Bitcoin Price Prediction Pipeline"
    )
    parser.add_argument("--init-db", action="store_true", help="Initialize database schema")
    parser.add_argument("--fetch-news", type=int, nargs="?", const=4, help="Fetch recent GDELT batches (default: 4 batches = 1 hour)")
    parser.add_argument("--fetch-btc", type=int, nargs="?", const=500, help="Fetch recent BTC candles (default: 500 hourly candles)")
    parser.add_argument("--filter-finance", action="store_true", help="Filter GDELT ingestion to finance/crypto/macro only")
    parser.add_argument("--stats", action="store_true", help="Show database statistics and category counts")
    parser.add_argument("--export-csv", action="store_true", help="Export database to CSV files")
    parser.add_argument("--train", action="store_true", help="Train ML models on news features and BTC prices")
    parser.add_argument("--all", action="store_true", help="Run full end-to-end pipeline: init, fetch news, fetch BTC, and report stats")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.init_db or args.all:
        print("[*] Initializing SQLite database...")
        init_db()
        print("[+] Database initialized successfully.")

    if args.fetch_btc is not None or args.all:
        limit = args.fetch_btc if args.fetch_btc is not None else 500
        ingest_btc_market_data(interval="1h", limit=limit)

    if args.fetch_news is not None or args.all:
        batches = args.fetch_news if args.fetch_news is not None else 4
        ingest_recent_gdelt_news(num_batches=batches, filter_crypto_finance=args.filter_finance)

    if args.stats or args.all:
        run_stats()

    if args.export_csv:
        run_export()

    if args.train:
        run_train()


if __name__ == "__main__":
    main()
