"""GDELT Global Knowledge Graph (GKG) Data Fetcher and Ingester."""

import io
import re
import html
import zipfile
import urllib.request
from datetime import datetime
from typing import List, Dict, Any, Optional

from config import GDELT_LASTUPDATE_URL, GDELT_MASTERFILE_URL, USER_AGENT
from classifier import classify_article
from database import insert_news_batch


def _extract_title(extra_xml: str) -> Optional[str]:
    """Extract and unescape <PAGE_TITLE> from GDELT V2ExtraXML."""
    if not extra_xml:
        return None
    match = re.search(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", extra_xml, re.IGNORECASE)
    if match:
        raw_title = match.group(1).strip()
        # Decode HTML entities like &#xF1; or &amp;
        return html.unescape(raw_title)
    return None


def _parse_gkg_date(date_str: str) -> str:
    """Convert GDELT YYYYMMDDHHMMSS format to ISO YYYY-MM-DD HH:MM:SS."""
    try:
        dt = datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return date_str


def _parse_tone(tone_str: str) -> Dict[str, float]:
    """Parse GDELT V1.5Tone comma-separated floats."""
    defaults = {
        "avg_tone": 0.0,
        "positive_score": 0.0,
        "negative_score": 0.0,
        "polarity": 0.0,
        "activity_density": 0.0,
        "word_count": 0
    }
    if not tone_str:
        return defaults

    parts = tone_str.split(",")
    try:
        if len(parts) >= 1:
            defaults["avg_tone"] = float(parts[0])
        if len(parts) >= 2:
            defaults["positive_score"] = float(parts[1])
        if len(parts) >= 3:
            defaults["negative_score"] = float(parts[2])
        if len(parts) >= 4:
            defaults["polarity"] = float(parts[3])
        if len(parts) >= 5:
            defaults["activity_density"] = float(parts[4])
        if len(parts) >= 7:
            defaults["word_count"] = int(float(parts[6]))
    except (ValueError, TypeError):
        pass

    return defaults


def fetch_and_process_gkg_zip(zip_url: str, filter_crypto_finance_only: bool = False) -> List[Dict[str, Any]]:
    """
    Download a GDELT GKG zip file in-memory, parse records, classify each article,
    and return structured article dictionaries.
    """
    req = urllib.request.Request(zip_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        zip_data = io.BytesIO(resp.read())

    articles = []
    with zipfile.ZipFile(zip_data) as zf:
        csv_filename = zf.namelist()[0]
        with zf.open(csv_filename) as f:
            for line in f:
                decoded = line.decode("utf-8", errors="ignore")
                fields = decoded.split("\t")
                if len(fields) < 27:
                    continue

                gkg_record_id = fields[0].strip()
                date_raw = fields[1].strip()
                source_domain = fields[3].strip()
                doc_url = fields[4].strip()
                raw_themes = fields[7].strip()
                raw_orgs = fields[13].strip()
                tone_raw = fields[15].strip()
                extra_xml = fields[26].strip()

                title = _extract_title(extra_xml)
                timestamp_iso = _parse_gkg_date(date_raw)
                tone_dict = _parse_tone(tone_raw)

                # Classify into primary and secondary categories
                primary_cat, _ = classify_article(
                    url=doc_url,
                    title=title or "",
                    themes=raw_themes,
                    orgs=raw_orgs
                )

                if filter_crypto_finance_only and primary_cat == "General_News":
                    continue

                article = {
                    "gkg_record_id": gkg_record_id,
                    "timestamp": timestamp_iso,
                    "source_domain": source_domain,
                    "url": doc_url,
                    "title": title,
                    "category": primary_cat,
                    "avg_tone": tone_dict["avg_tone"],
                    "positive_score": tone_dict["positive_score"],
                    "negative_score": tone_dict["negative_score"],
                    "polarity": tone_dict["polarity"],
                    "activity_density": tone_dict["activity_density"],
                    "word_count": tone_dict["word_count"],
                    "raw_themes": raw_themes[:500] if raw_themes else None,
                    "organizations": raw_orgs[:500] if raw_orgs else None,
                }
                articles.append(article)

    return articles


def get_recent_gkg_urls(num_batches: int = 4) -> List[str]:
    """
    Retrieve URLs for the most recent N 15-minute GKG batches from GDELT.
    Uses HTTP range requests against masterfilelist.txt to avoid downloading the multi-GB full list.
    """
    # Range request to fetch recent 100KB of masterfilelist
    req = urllib.request.Request(
        GDELT_MASTERFILE_URL,
        headers={"User-Agent": USER_AGENT, "Range": "bytes=-100000"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode("utf-8", errors="ignore")

    gkg_urls = []
    lines = text.strip().split("\n")
    for line in reversed(lines):
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2].endswith(".gkg.csv.zip"):
            gkg_urls.append(parts[2])
            if len(gkg_urls) >= num_batches:
                break

    return list(reversed(gkg_urls))


def ingest_recent_gdelt_news(num_batches: int = 4, filter_crypto_finance: bool = False) -> int:
    """
    Ingest the most recent N GDELT 15-minute batches into the local SQLite database.
    Returns total number of inserted articles.
    """
    urls = get_recent_gkg_urls(num_batches=num_batches)
    print(f"[*] Found {len(urls)} recent GDELT GKG batches to process.")

    total_inserted = 0
    for idx, url in enumerate(urls, start=1):
        filename = url.split("/")[-1]
        print(f"[{idx}/{len(urls)}] Downloading & processing {filename}...")
        try:
            articles = fetch_and_process_gkg_zip(url, filter_crypto_finance_only=filter_crypto_finance)
            inserted = insert_news_batch(articles)
            total_inserted += inserted
            print(f"  -> Extracted {len(articles)} articles, inserted {inserted} new records.")
        except Exception as e:
            print(f"  [!] Failed to process batch {filename}: {e}")

    return total_inserted
