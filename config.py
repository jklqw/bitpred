"""Configuration settings for GDELT News Database and Bitcoin Price Prediction Pipeline."""

from pathlib import Path
import sys

# Paths
BASE_DIR = Path(__file__).resolve().parent

# Auto-bootstrap virtualenv site-packages if running with system python
_site_packages = BASE_DIR / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _site_packages.is_dir() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


DB_PATH = DATA_DIR / "crypto_news_alpha.db"

# GDELT URLs
GDELT_LASTUPDATE_URL = "https://data.gdeltproject.org/gdeltv2/lastupdate.txt"
GDELT_MASTERFILE_URL = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
GDELT_GKG_BASE_URL = "https://data.gdeltproject.org/gdeltv2/"

# Default Categories for News Classification
CATEGORIES = [
    "Crypto_Blockchain",
    "Monetary_Policy",
    "Macroeconomics",
    "Regulation_Legal",
    "Geopolitics_Conflict",
    "Tech_Cybersecurity",
    "Energy_Commodities",
    "Equities_FinancialMarkets",
    "General_News"
]

# Market Data Configuration
DEFAULT_INTERVAL = "1h"

# --- Historical training-data window -------------------------------------
# Training uses every available candle/article inside this window.  Set
# TRAINING_HISTORY_YEARS to 1 or 2 depending on how much backfill you want.
TRAINING_HISTORY_YEARS = 2
BTC_HISTORY_CANDLES = int(TRAINING_HISTORY_YEARS * 365.25 * 24)  # ~17,532 @ 1h
NEWS_HISTORY_DAYS = int(TRAINING_HISTORY_YEARS * 365.25)          # ~730 days

# GDELT publishes a GKG zip every 15 minutes. Downloading all of them for two
# years is ~200GB, so historical backfill samples one file per interval.
# Recent data is still ingested at full 15-minute cadence by the live sync.
NEWS_SAMPLE_INTERVAL_HOURS = 6
GDELT_MASTERFILE_CACHE_HOURS = 6

DEFAULT_LOOKBACK_CANDLES = BTC_HISTORY_CANDLES
BINANCE_CHUNK_SIZE = 1000        # Binance API max limit per request
BINANCE_RATE_LIMIT_SLEEP = 0.05  # Polite delay between chunk requests (seconds)

# Live periods -> X-day training configurations.  Each period trains and
# updates its OWN model artifact from the shared historical database.
LIVE_PERIOD_X_DAYS = {
    "3h": 0.25,
    "6h": 0.5,
    "12h": 1.0,
    "1d": 2.0,
    "3d": 6.0,
}

# Refresh every period's model from the accumulated history at this interval.
MODEL_RETRAIN_INTERVAL_HOURS = 24

# Trading & Backtest Configuration
DEFAULT_PREDICTION_HORIZON = 1   # Forecast 1 candle ahead (configurable: 1, 4, 12, 24)
DEFAULT_TARGET_THRESHOLD = 0.0   # Return threshold for positive classification (e.g., 0.0 or fee-adjusted 0.001)
DEFAULT_TAKER_FEE = 0.00075      # 0.075% taker fee (Binance standard with BNB discount)
DEFAULT_SLIPPAGE = 0.0002        # 0.02% estimated market impact slippage

# Live trading entry gate: open a trade only when hit probability is above this (%)
MIN_ENTRY_HIT_PROB = 75.0
# Close an open trade when its hit probability falls below this (%)
EXIT_HIT_PROB = 25.0

# Reproducibility
RANDOM_SEED = 42

# User-Agent header for courteous web requests
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
