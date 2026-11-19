"""GDELT Global Knowledge Graph (GKG) Data Fetcher and Ingester with sampling & historical support."""

import io
import re
import html
import time
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


def fetch_and_process_gkg_zip(
    zip_url: str,
    filter_crypto_finance_only: bool = False,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    Download a GDELT GKG zip file in-memory, parse records, classify each article,
    and return structured article dictionaries.
    """
    req = urllib.request.Request(zip_url, headers={"User-Agent": USER_AGENT})
    zip_data = None

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                zip_data = io.BytesIO(resp.read())
                break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  [!] GDELT file not found (404): {zip_url}")
                return []
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [!] Failed to download {zip_url}: {e}")
                return []
            time.sleep(1.5 * (attempt + 1))

    if not zip_data:
        return []

    articles = []
    try:
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
    except Exception as e:
        print(f"  [!] Error parsing zip content: {e}")

    return articles


def get_historical_gkg_urls(
    sample_hours: int = 24,
    step_batches: int = 4
) -> List[str]:
    """
    Retrieve URLs sampled across past hours from masterfilelist.txt.
    step_batches=4 samples 1 file per hour (since files are 15-min intervals).
    """
    bytes_needed = max(500000, sample_hours * 4 * 120)
    req = urllib.request.Request(
        GDELT_MASTERFILE_URL,
        headers={"User-Agent": USER_AGENT, "Range": f"bytes=-{bytes_needed}"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode("utf-8", errors="ignore")

    lines = text.strip().split("\n")
    gkg_urls = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2].endswith(".gkg.csv.zip"):
            gkg_urls.append(parts[2])

    total_files_needed = sample_hours * 4
    recent_slice = gkg_urls[-total_files_needed:] if len(gkg_urls) >= total_files_needed else gkg_urls
    sampled = recent_slice[::step_batches]
    return sampled


def get_recent_gkg_urls(num_batches: int = 4) -> List[str]:
    """
    Retrieve URLs for the most recent N 15-minute GKG batches from GDELT.
    """
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


def ingest_recent_gdelt_news(
    num_batches: Optional[int] = 4,
    sample_hours: Optional[int] = None,
    filter_crypto_finance: bool = False,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ingest GDELT news batches into SQLite:
    - If sample_hours is specified: samples across sample_hours (1 file per hour).
    - Otherwise ingests the last num_batches consecutive 15-minute files.
    """
    if sample_hours:
        urls = get_historical_gkg_urls(sample_hours=sample_hours, step_batches=4)
        print(f"[*] Ingesting {len(urls)} hourly sampled GDELT batches across {sample_hours} hours...")
    else:
        urls = get_recent_gkg_urls(num_batches=num_batches or 4)
        print(f"[*] Ingesting {len(urls)} consecutive GDELT 15-minute batches...")

    total_inserted = 0
    total_extracted = 0

    for idx, url in enumerate(urls, start=1):
        filename = url.split("/")[-1]
        print(f"  [{idx}/{len(urls)}] Processing {filename}...")
        articles = fetch_and_process_gkg_zip(url, filter_crypto_finance_only=filter_crypto_finance)
        total_extracted += len(articles)
        inserted = insert_news_batch(articles, db_path=db_path)
        total_inserted += inserted

    summary = {
        "batches_processed": len(urls),
        "total_extracted": total_extracted,
        "total_inserted": total_inserted
    }
    print(f"[+] GDELT Ingestion complete: {total_extracted} extracted, {total_inserted} new records inserted.")
    return summary
