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
GDELT_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
GDELT_MASTERFILE_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

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
DEFAULT_LOOKBACK_CANDLES = 5000  # Default ~208 days of hourly data (substantially increased from 500)
BINANCE_CHUNK_SIZE = 1000        # Binance API max limit per request
BINANCE_RATE_LIMIT_SLEEP = 0.05  # Polite delay between chunk requests (seconds)

# Trading & Backtest Configuration
DEFAULT_PREDICTION_HORIZON = 1   # Forecast 1 candle ahead (configurable: 1, 4, 12, 24)
DEFAULT_TARGET_THRESHOLD = 0.0   # Return threshold for positive classification (e.g., 0.0 or fee-adjusted 0.001)
DEFAULT_TAKER_FEE = 0.00075      # 0.075% taker fee (Binance standard with BNB discount)
DEFAULT_SLIPPAGE = 0.0002        # 0.02% estimated market impact slippage

# Reproducibility
RANDOM_SEED = 42

# User-Agent header for courteous web requests
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
