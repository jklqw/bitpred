#!/usr/bin/env python3
"""Server-ready bootstrap: initialize DB -> backfill BTC -> backfill news -> initial train.

Run this on the production server (directly, via nohup/tmux, or via the bundled
systemd unit). Every phase is resumable:

  * BTC candles are upserted, so re-running simply refreshes/continues them.
  * GDELT files are checkpointed in `ingestion_progress`, so completed files
    are skipped on the next run.
  * Model artifacts are written atomically and only retrained when missing or
    older than MODEL_RETRAIN_INTERVAL_HOURS (unless --force-train is given).

A status snapshot is written to data/bootstrap_status.json and output is also
appended to data/bootstrap.log so progress can be monitored remotely.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent

# Auto-bootstrap the virtualenv site-packages when invoked with system python.
_site_packages = BASE_DIR / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _site_packages.is_dir() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (  # noqa: E402
    DB_PATH,
    DATA_DIR,
    TRAINING_HISTORY_YEARS,
    NEWS_HISTORY_DAYS,
    NEWS_SAMPLE_INTERVAL_HOURS,
    MODEL_RETRAIN_INTERVAL_HOURS,
    LIVE_PERIOD_X_DAYS,
)
from database import init_db, get_database_stats  # noqa: E402
from btc_fetcher import ingest_btc_history  # noqa: E402
from gdelt_fetcher import ingest_export_history, ingest_gdelt_history  # noqa: E402
from train import train_all_periods, get_x_period_artifact_path  # noqa: E402

STATUS_FILE = DATA_DIR / "bootstrap_status.json"
LOCK_FILE = DATA_DIR / "bootstrap.lock"
LOG_FILE = DATA_DIR / "bootstrap.log"

_STOP_REQUESTED = False


def _handle_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"\n[*] Received signal {signum}; finishing current file then stopping.", flush=True)


signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)


def stop_check() -> bool:
    return _STOP_REQUESTED


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Bootstrap:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.status: Dict[str, Any] = {
            "phase": "starting",
            "started_at": _utcnow(),
            "updated_at": _utcnow(),
            "config": {
                "btc_years": args.btc_years,
                "news_days": args.news_days,
                "news_source": args.news_source,
                "news_interval_hours": args.news_interval_hours,
                "news_workers": args.news_workers,
                "force_train": args.force_train,
            },
            "database": str(DB_PATH),
            "btc": {},
            "news": {},
            "training": {},
            "error": None,
        }

    # ------------------------------------------------------------------ utils
    def log(self, message: str) -> None:
        line = f"[{_utcnow()}] {message}"
        print(line, flush=True)
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass

    def save_status(self) -> None:
        self.status["updated_at"] = _utcnow()
        try:
            tmp = STATUS_FILE.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self.status, handle, indent=2, default=str)
            os.replace(tmp, STATUS_FILE)
        except Exception as exc:
            self.log(f"[!] Could not write status file: {exc}")

    def set_phase(self, phase: str) -> None:
        self.status["phase"] = phase
        self.save_status()
        self.log(f"=== Phase: {phase} ===")

    def _record_db_stats(self) -> None:
        try:
            stats = get_database_stats()
            self.status["database_stats"] = {
                "total_news": stats.get("total_news"),
                "news_date_range": list(stats.get("news_date_range", (None, None))),
                "btc_price_count": stats.get("btc_price_count"),
                "btc_date_range": list(stats.get("btc_date_range", (None, None))),
            }
        except Exception as exc:
            self.log(f"[!] Could not read DB stats: {exc}")
        self.save_status()

    # ----------------------------------------------------------------- phases
    def phase_init(self) -> None:
        if _STOP_REQUESTED:
            return
        self.set_phase("init_db")
        init_db()
        self._record_db_stats()
        self.log("[+] Database schema ready.")

    def phase_btc(self) -> None:
        if _STOP_REQUESTED or self.args.skip_btc:
            return
        self.set_phase("btc_backfill")
        summary = ingest_btc_history(years=self.args.btc_years)
        self.status["btc"] = {
            "date_start": summary.get("date_start"),
            "date_end": summary.get("date_end"),
            "inserted_or_updated": summary.get("inserted_or_updated"),
            "requested_candles": summary.get("requested_candles"),
        }
        self._record_db_stats()
        self.log("[+] BTC backfill phase complete.")

    def phase_news(self) -> None:
        if _STOP_REQUESTED or self.args.skip_news:
            return
        self.set_phase("news_backfill")
        common = dict(
            days=self.args.news_days,
            interval_hours=self.args.news_interval_hours,
            resume=not self.args.no_resume_news,
            workers=self.args.news_workers,
            stop_check=stop_check,
        )
        if self.args.news_source == "export":
            summary = ingest_export_history(**common)
        else:
            summary = ingest_gdelt_history(
                **common, filter_crypto_finance=self.args.filter_finance
            )
        self.status["news"] = {
            "files_selected": summary.get("files_selected"),
            "files_processed": summary.get("files_processed"),
            "files_skipped_resume": summary.get("files_skipped_resume"),
            "files_failed": summary.get("files_failed"),
            "total_inserted": summary.get("total_inserted"),
        }
        self._record_db_stats()
        self.log("[+] News backfill phase complete.")

    def _needs_initial_train(self) -> bool:
        """True when any live-period artifact is missing, stale, or unreadable."""
        if self.args.force_train:
            return True
        import joblib

        for x_days in LIVE_PERIOD_X_DAYS.values():
            path = get_x_period_artifact_path(x_days)
            if not path.exists():
                return True
            try:
                trained_at = float(joblib.load(path).get("trained_at", 0.0))
                age_hours = (time.time() - trained_at) / 3600.0
                if age_hours > MODEL_RETRAIN_INTERVAL_HOURS:
                    return True
            except Exception:
                return True
        return False

    def phase_train(self) -> None:
        if _STOP_REQUESTED or self.args.skip_train:
            return
        if not self._needs_initial_train():
            self.set_phase("train_skipped_fresh")
            self.log("[+] All period artifacts are fresh; skipping initial training.")
            return

        self.set_phase("initial_train")
        results = train_all_periods(
            taker_fee=self.args.taker_fee, slippage=self.args.slippage
        )
        self.status["training"] = {
            period: {
                "x_days": res.get("x_days"),
                "samples": res.get("samples"),
                "threshold": res.get("threshold"),
                "artifact_path": res.get("artifact_path"),
                "error": res.get("error"),
            }
            for period, res in results.items()
        }
        self.save_status()
        self.log("[+] Initial training phase complete.")

    # -------------------------------------------------------------------- run
    def run(self) -> None:
        try:
            self.phase_init()
            self.phase_btc()
            self.phase_news()
            self.phase_train()
            if _STOP_REQUESTED:
                self.status["phase"] = "stopped"
                self.log("[*] Bootstrap stopped by signal; resume by re-running.")
            else:
                self.status["phase"] = "done"
                self.log("[+] Bootstrap finished successfully.")
        except Exception as exc:
            self.status["phase"] = "error"
            self.status["error"] = str(exc)
            self.status["traceback"] = traceback.format_exc()
            self.log(f"[!] Bootstrap FAILED: {exc}")
            raise
        finally:
            self.save_status()


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Server bootstrap: download 1-2 years of BTC + news, then train all live models."
    )
    parser.add_argument("--btc-years", type=float, default=TRAINING_HISTORY_YEARS,
                        help=f"Years of hourly BTC candles to backfill (default: {TRAINING_HISTORY_YEARS})")
    parser.add_argument("--news-days", type=int, default=NEWS_HISTORY_DAYS,
                        help=f"Days of GDELT news to backfill (default: {NEWS_HISTORY_DAYS})")
    parser.add_argument("--news-source", choices=("export", "gkg"), default="export",
                        help="export (fast) or gkg (rich) historical source (default: export)")
    parser.add_argument("--news-interval-hours", type=int, default=NEWS_SAMPLE_INTERVAL_HOURS,
                        help=f"One file per N hours (default: {NEWS_SAMPLE_INTERVAL_HOURS})")
    parser.add_argument("--news-workers", type=int, default=8,
                        help="Parallel download workers (default: 8)")
    parser.add_argument("--filter-finance", action="store_true",
                        help="Filter rich GKG backfill to finance/crypto articles")
    parser.add_argument("--no-resume-news", action="store_true",
                        help="Re-download already-processed GDELT files")
    parser.add_argument("--skip-btc", action="store_true", help="Skip BTC backfill phase")
    parser.add_argument("--skip-news", action="store_true", help="Skip news backfill phase")
    parser.add_argument("--skip-train", action="store_true", help="Skip initial training phase")
    parser.add_argument("--force-train", action="store_true",
                        help="Retrain all period models even if artifacts are fresh")
    parser.add_argument("--taker-fee", type=float, default=0.00075,
                        help="Taker fee per trade side for backtest")
    parser.add_argument("--slippage", type=float, default=0.0002,
                        help="Slippage per trade side for backtest")
    parser.add_argument("--start-server", action="store_true",
                        help="After bootstrap, replace this process with the live web server")
    parser.add_argument("--status", action="store_true",
                        help="Print the last bootstrap status snapshot and exit")
    return parser.parse_args(argv)


def print_status() -> None:
    if not STATUS_FILE.exists():
        print("No bootstrap status found yet.")
        return
    print(STATUS_FILE.read_text(encoding="utf-8"))


def main(argv: Any = None) -> int:
    args = parse_args(argv)
    if args.status:
        print_status()
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = open(LOCK_FILE, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[!] Another bootstrap process holds {LOCK_FILE}; exiting.")
        return 0

    try:
        bootstrap = Bootstrap(args)
        bootstrap.log(f"[*] Bootstrap starting (PID {os.getpid()}), DB={DB_PATH}")
        bootstrap.run()
    except Exception:
        return 1
    finally:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
        finally:
            lock_handle.close()

    if _STOP_REQUESTED:
        return 0

    if args.start_server:
        bootstrap.log("[*] Handing over to the live web server...")
        os.execv(
            sys.executable,
            [sys.executable, "-u", str(BASE_DIR / "web_server.py")],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
