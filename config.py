"""Configuration settings for GDELT News Database and Bitcoin Price Prediction Pipeline."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
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

# User-Agent header for courteous web requests
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
