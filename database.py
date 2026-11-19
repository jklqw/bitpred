"""Database layer using SQLite for GDELT news and Bitcoin market data."""

import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
from config import DB_PATH


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return an active connection to the SQLite database."""
    target_path = db_path or str(DB_PATH)
    conn = sqlite3.connect(target_path)
    conn.execute("PRAGMA journal_mode=WAL;")  # Better concurrent read/write performance
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: Optional[str] = None):
    """Initialize database tables and indexes."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # News Articles Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gkg_record_id TEXT UNIQUE,
                timestamp DATETIME NOT NULL,
                source_domain TEXT,
                url TEXT NOT NULL,
                title TEXT,
                category TEXT NOT NULL,
                avg_tone REAL,
                positive_score REAL,
                negative_score REAL,
                polarity REAL,
                activity_density REAL,
                word_count INTEGER,
                raw_themes TEXT,
                organizations TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Indexes for fast querying & time-series slicing
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news_articles(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_category ON news_articles(category);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_source ON news_articles(source_domain);")

        # Bitcoin Prices Table (OHLCV)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS btc_prices (
                timestamp DATETIME PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL,
                trades_count INTEGER
            );
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_btc_timestamp ON btc_prices(timestamp);")

        conn.commit()


def insert_news_batch(articles: List[Dict[str, Any]], db_path: Optional[str] = None) -> int:
    """Insert a batch of news articles, ignoring duplicates by gkg_record_id."""
    if not articles:
        return 0

    query = """
        INSERT OR IGNORE INTO news_articles (
            gkg_record_id, timestamp, source_domain, url, title,
            category, avg_tone, positive_score, negative_score,
            polarity, activity_density, word_count, raw_themes,
            organizations
        ) VALUES (
            :gkg_record_id, :timestamp, :source_domain, :url, :title,
            :category, :avg_tone, :positive_score, :negative_score,
            :polarity, :activity_density, :word_count, :raw_themes,
            :organizations
        );
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(query, articles)
        conn.commit()
        return cursor.rowcount


def insert_btc_prices(prices: List[Dict[str, Any]], db_path: Optional[str] = None) -> int:
    """Insert or replace Bitcoin OHLCV price records."""
    if not prices:
        return 0

    query = """
        INSERT OR REPLACE INTO btc_prices (
            timestamp, open, high, low, close, volume, quote_volume, trades_count
        ) VALUES (
            :timestamp, :open, :high, :low, :close, :volume, :quote_volume, :trades_count
        );
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(query, prices)
        conn.commit()
        return cursor.rowcount


def get_database_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve statistical summary of stored data."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Count news
        cursor.execute("SELECT COUNT(*) FROM news_articles;")
        total_news = cursor.fetchone()[0]

        # Category breakdown
        cursor.execute("""
            SELECT category, COUNT(*), ROUND(AVG(avg_tone), 2)
            FROM news_articles
            GROUP BY category
            ORDER BY COUNT(*) DESC;
        """)
        category_stats = cursor.fetchall()

        # Date range for news
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM news_articles;")
        news_min, news_max = cursor.fetchone()

        # Count BTC prices
        cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM btc_prices;")
        btc_count, btc_min, btc_max = cursor.fetchone()

    return {
        "total_news": total_news,
        "news_date_range": (news_min, news_max),
        "categories": category_stats,
        "btc_price_count": btc_count,
        "btc_date_range": (btc_min, btc_max)
    }
