"""GDELT Global Knowledge Graph (GKG) Data Fetcher and Ingester with sampling & historical support."""

import io
import os
import re
import html
import time
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import math
from typing import List, Dict, Any, Optional, Callable

from config import (
    GDELT_LASTUPDATE_URL,
    GDELT_MASTERFILE_URL,
    USER_AGENT,
    DATA_DIR,
    GDELT_MASTERFILE_CACHE_HOURS
)
from classifier import classify_article
from database import insert_news_batch, is_source_processed, mark_source_processed, init_db


def _extract_title(extra_xml: str) -> Optional[str]:
    """Extract and unescape <PAGE_TITLE> from GDELT V2ExtraXML."""
    if not extra_xml:
        return None
    match = re.search(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", extra_xml, re.IGNORECASE)
    if match:
        raw_title = match.group(1).strip()
        return html.unescape(raw_title)
    return None


def _parse_gkg_date(date_str: str) -> Optional[str]:
    """Convert GDELT YYYYMMDDHHMMSS format to ISO; return None for invalid dates."""
    try:
        dt = datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


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
            if attempt == max_retries - 1:
                raise
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
                    if timestamp_iso is None:
                        continue
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
    bytes_needed = max(500000, int(sample_hours * 4 * 3 * 160))
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
    if len(gkg_urls) < total_files_needed:
        raise RuntimeError(
            f"GDELT master file returned only {len(gkg_urls)} GKG URLs; "
            f"expected at least {total_files_needed} for {sample_hours} hours."
        )
    recent_slice = gkg_urls[-total_files_needed:]
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


# ---------------------------------------------------------------------------
# Historical backfill (1-2 years) for model training
# ---------------------------------------------------------------------------

def _masterfile_cache_path(days: int):
    """Cache key includes the covered history window so a longer range refetches."""
    return DATA_DIR / f"gdelt_masterfilelist_{int(days)}d.txt"


def _load_masterfilelist(days: int = 730, force_refresh: bool = False) -> str:
    """Download (and locally cache) the GDELT master file list tail.

    The complete list is ~130MB (2015-present). Only the tail covering the
    requested history is needed: ~140 bytes/line × 3 lines per 15-min block.
    A Range request keeps a 2-year backfill to roughly 30MB.
    """
    days = max(1, int(days))
    cache_path = _masterfile_cache_path(days)

    if cache_path.exists() and not force_refresh:
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_hours < GDELT_MASTERFILE_CACHE_HOURS:
            print(f"[*] Using cached GDELT master file list ({cache_path.name}).")
            return cache_path.read_text(encoding="utf-8", errors="ignore")

    tail_bytes = max(2_000_000, int(days * 96 * 3 * 140))
    print(f"[*] Downloading last ~{tail_bytes / 1e6:.0f}MB of the GDELT master file list "
          f"(cached for {GDELT_MASTERFILE_CACHE_HOURS}h)...")
    req = urllib.request.Request(
        GDELT_MASTERFILE_URL,
        headers={"User-Agent": USER_AGENT, "Range": f"bytes=-{tail_bytes}"}
    )
    tmp_path = cache_path.with_suffix(".tmp")
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp_path, cache_path)
    return cache_path.read_text(encoding="utf-8", errors="ignore")


def get_gkg_urls_for_range(
    start: datetime,
    end: datetime,
    interval_hours: int = 6,
    force_refresh_masterfile: bool = False
) -> List[str]:
    """Return one GKG URL per sampling interval within [start, end].

    GDELT publishes a file every 15 minutes. Downloading all of them for 1-2
    years would be hundreds of GB, so this samples a single file for each
    `interval_hours` block while keeping full cadence possible via
    `interval_hours=0` (return every file).
    """
    if end <= start:
        raise ValueError("end must be after start")

    days_needed = max(1, int((end - start).total_seconds() / 86400) + 1)
    text = _load_masterfilelist(days=days_needed, force_refresh=force_refresh_masterfile)
    interval_hours = max(0, int(interval_hours))

    selected: Dict[int, tuple] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[2].endswith(".gkg.csv.zip"):
            continue
        filename = parts[2].rsplit("/", 1)[-1]
        stamp = filename[:14]
        try:
            dt = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if not (start <= dt <= end):
            continue

        if interval_hours > 0:
            bucket = int(dt.timestamp()) // (interval_hours * 3600)
            # Keep the most recent file in each bucket.
            previous = selected.get(bucket)
            if previous is None or dt > previous[0]:
                selected[bucket] = (dt, parts[2])
        else:
            selected[int(dt.timestamp())] = (dt, parts[2])

    urls = [url for _, url in sorted(selected.values())]
    print(f"[*] Selected {len(urls)} GKG files between "
          f"{start.strftime('%Y-%m-%d')} and {end.strftime('%Y-%m-%d')} "
          f"(interval={interval_hours or 0.25}h).")
    return urls


def ingest_gdelt_history(
    days: int = 730,
    interval_hours: int = 6,
    filter_crypto_finance: bool = False,
    resume: bool = True,
    db_path: Optional[str] = None,
    force_refresh_masterfile: bool = False,
    workers: int = 8,
    stop_check: Optional[Callable[[], bool]] = None
) -> Dict[str, Any]:
    """Backfill GDELT news across `days` of history with resumable progress.

    Downloads/parses GKG files in parallel, then writes to SQLite sequentially
    from the main thread. Already-processed files are skipped when `resume=True`,
    so an interrupted backfill can simply be re-run to continue.
    """
    init_db(db_path=db_path)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(days))
    urls = get_gkg_urls_for_range(
        start, end,
        interval_hours=interval_hours,
        force_refresh_masterfile=force_refresh_masterfile
    )

    pending = []
    skipped = 0
    for url in urls:
        key = url.rsplit("/", 1)[-1]
        if resume and is_source_processed(key, db_path=db_path):
            skipped += 1
        else:
            pending.append((key, url))

    processed = inserted_total = extracted_total = failed = 0
    workers = max(1, min(int(workers), 16))
    print(f"[*] Processing {len(pending)} files with {workers} parallel workers "
          f"({skipped} already done).")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_and_process_gkg_zip, url, filter_crypto_finance): (key, url)
            for key, url in pending
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            if stop_check is not None and stop_check():
                # Cancel queued work; in-flight downloads finish but are not
                # written, so they are simply re-fetched on the next resume.
                for pending_future in futures:
                    pending_future.cancel()
                print("[*] Stop requested - checkpointing and exiting backfill.")
                break
            key, _ = futures[future]
            try:
                articles = future.result()
            except Exception as exc:
                failed += 1
                print(f"  [{idx}/{len(pending)}] [!] {key} failed: {exc}")
                continue

            inserted = insert_news_batch(articles, db_path=db_path)
            mark_source_processed(key, inserted, db_path=db_path)
            extracted_total += len(articles)
            inserted_total += inserted
            processed += 1
            if processed % 25 == 0 or processed == len(pending):
                print(f"  -> {processed}/{len(pending)} files done, "
                      f"{inserted_total} new articles inserted.")

    summary = {
        "days_requested": int(days),
        "interval_hours": int(interval_hours),
        "files_selected": len(urls),
        "files_processed": processed,
        "files_skipped_resume": skipped,
        "files_failed": failed,
        "total_extracted": extracted_total,
        "total_inserted": inserted_total
    }
    print(f"[+] Historical GDELT backfill complete: {processed} files processed, "
          f"{skipped} skipped (resume), {failed} failed, "
          f"{inserted_total} new articles inserted.")
    return summary


# ---------------------------------------------------------------------------
# Fast historical backfill via GDELT export CSV (~10x smaller than GKG zips)
# ---------------------------------------------------------------------------

def _export_url_to_article(fields: List[str]) -> Optional[Dict[str, Any]]:
    """Convert one GDELT 2.0 export CSV row into the news_articles schema."""
    if len(fields) < 61:
        return None
    timestamp = _parse_gkg_date(fields[59].strip())
    if timestamp is None:
        return None

    try:
        tone = float(fields[34])
    except (ValueError, TypeError):
        tone = 0.0
    try:
        mentions = float(fields[31])
    except (ValueError, TypeError):
        mentions = 0.0

    url = fields[60].strip()
    domain = ""
    if url:
        domain = url.split("//")[-1].split("/", 1)[0].replace("www.", "")

    event_code = fields[26].strip()
    themes = f"EVENT_{event_code}" if event_code else ""
    if fields[7].strip():
        themes += f",{fields[7].strip()}"
    if fields[17].strip():
        themes += f",{fields[17].strip()}"

    orgs = " | ".join(x.strip() for x in (fields[6], fields[16]) if x.strip())
    category, _ = classify_article(
        url=url,
        title="",
        themes=themes,
        orgs=orgs
    )

    return {
        "gkg_record_id": f"exp-{fields[0].strip()}",
        "timestamp": timestamp,
        "source_domain": domain[:255],
        "url": url,
        "title": None,
        "category": category,
        "avg_tone": tone,
        "positive_score": max(tone, 0.0),
        "negative_score": max(-tone, 0.0),
        "polarity": math.tanh(tone / 5.0),
        "activity_density": mentions,
        "word_count": 0,
        "raw_themes": themes[:500] or None,
        "organizations": orgs[:500] or None,
    }


def fetch_and_process_export_zip(
    zip_url: str,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """Download and parse a GDELT 2.0 export CSV zip into article rows."""
    req = urllib.request.Request(zip_url, headers={"User-Agent": USER_AGENT})
    zip_data = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                zip_data = io.BytesIO(resp.read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            if attempt == max_retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))

    articles: List[Dict[str, Any]] = []
    if zip_data is None:
        return articles
    try:
        with zipfile.ZipFile(zip_data) as zf:
            with zf.open(zf.namelist()[0]) as f:
                for line in f:
                    row = _export_url_to_article(line.decode("utf-8", errors="ignore").split("\t"))
                    if row is not None:
                        articles.append(row)
    except Exception as e:
        print(f"  [!] Error parsing export zip content: {e}")
    return articles


def get_export_urls_for_range(
    start: datetime,
    end: datetime,
    interval_hours: int = 6,
    force_refresh_masterfile: bool = False
) -> List[str]:
    """Select one GDELT export CSV file per sampling interval in a date range."""
    if end <= start:
        raise ValueError("end must be after start")

    days_needed = max(1, int((end - start).total_seconds() / 86400) + 1)
    text = _load_masterfilelist(days=days_needed, force_refresh=force_refresh_masterfile)
    interval_hours = max(0, int(interval_hours))

    selected: Dict[int, tuple] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[2].endswith(".export.CSV.zip"):
            continue
        stamp = parts[2].rsplit("/", 1)[-1][:14]
        try:
            dt = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if not (start <= dt <= end):
            continue
        if interval_hours > 0:
            bucket = int(dt.timestamp()) // (interval_hours * 3600)
            prev = selected.get(bucket)
            if prev is None or dt > prev[0]:
                selected[bucket] = (dt, parts[2])
        else:
            selected[int(dt.timestamp())] = (dt, parts[2])

    urls = [url for _, url in sorted(selected.values())]
    print(f"[*] Selected {len(urls)} export files between "
          f"{start.strftime('%Y-%m-%d')} and {end.strftime('%Y-%m-%d')} "
          f"(interval={interval_hours or 0.25}h).")
    return urls


def ingest_export_history(
    days: int = 730,
    interval_hours: int = 6,
    resume: bool = True,
    db_path: Optional[str] = None,
    force_refresh_masterfile: bool = False,
    workers: int = 8,
    stop_check: Optional[Callable[[], bool]] = None
) -> Dict[str, Any]:
    """Fast resumable multi-year news backfill using GDELT export CSV files.

    Export files are ~80KB versus ~3.6MB for GKG files and still carry AvgTone,
    event codes, actor names and source URLs, so they are ideal for building a
    large historical sentiment database for model training.
    """
    init_db(db_path=db_path)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(days))
    urls = get_export_urls_for_range(
        start, end,
        interval_hours=interval_hours,
        force_refresh_masterfile=force_refresh_masterfile
    )

    pending = []
    skipped = 0
    for url in urls:
        key = "export-" + url.rsplit("/", 1)[-1]
        if resume and is_source_processed(key, db_path=db_path):
            skipped += 1
        else:
            pending.append((key, url))

    processed = inserted_total = extracted_total = failed = 0
    workers = max(1, min(int(workers), 16))
    print(f"[*] Processing {len(pending)} export files with {workers} workers "
          f"({skipped} already done).")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_and_process_export_zip, url): (key, url)
            for key, url in pending
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            if stop_check is not None and stop_check():
                for pending_future in futures:
                    pending_future.cancel()
                print("[*] Stop requested - checkpointing and exiting backfill.")
                break
            key, _ = futures[future]
            try:
                articles = future.result()
            except Exception as exc:
                failed += 1
                print(f"  [{idx}/{len(pending)}] [!] {key} failed: {exc}")
                continue

            inserted = insert_news_batch(articles, db_path=db_path)
            mark_source_processed(key, inserted, source_type="export", db_path=db_path)
            extracted_total += len(articles)
            inserted_total += inserted
            processed += 1
            if processed % 50 == 0 or processed == len(pending):
                print(f"  -> {processed}/{len(pending)} files done, "
                      f"{inserted_total} new events inserted.")

    summary = {
        "days_requested": int(days),
        "interval_hours": int(interval_hours),
        "files_selected": len(urls),
        "files_processed": processed,
        "files_skipped_resume": skipped,
        "files_failed": failed,
        "total_extracted": extracted_total,
        "total_inserted": inserted_total,
    }
    print(f"[+] Export backfill complete: {processed} files processed, "
          f"{skipped} skipped, {failed} failed, "
          f"{inserted_total} new events inserted.")
    return summary
