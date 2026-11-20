#!/usr/bin/env python3
"""Generate a static dashboard (index.html + status.json) — no web server / 127.0.0.1 needed.

Usage:
  python3 generate_html.py            # write once
  python3 generate_html.py --watch    # keep refreshing status.json every few seconds
  python3 generate_html.py --out DIR  # output directory (default: ./site)

Serve the output dir with any static file server (e.g. caddy, nginx, `python3 -m http.server`).
The page polls the relative file ./status.json, so it works at any host/path.
"""
import sys
import json
import time
import argparse
import tempfile
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
_sp = BASE_DIR / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _sp.is_dir() and str(_sp) not in sys.path:
    sys.path.insert(0, str(_sp))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

STATIC_PATCHES = [
    # Resolve status.json next to the page even when opened as /path/site (no trailing slash)
    ("fetch('/api/status')", "fetch(STATUS_URL + '?t=' + Date.now(), {cache: 'no-store'})"),
    # Static mode has no /api/alerts endpoint; use the same status.json (falls back to globalData.alerts)
    ("fetch('/api/alerts')", "fetch(STATUS_URL + '?t=' + Date.now(), {cache: 'no-store'})"),
    ("let globalData = null;",
     "const _p = location.pathname;\n    const _dir = (_p.endsWith('/') ? _p : _p.endsWith('.html') ? _p.replace(/[^/]*$/, '') : _p + '/');\n    const STATUS_URL = _dir + 'status.json';\n    let globalData = null;"),
    ("setInterval(fetchStatus, 2500);", "setInterval(fetchStatus, 5000);"),
    # Read-only static mode: actions need a live engine
    ("async function dismissSingleAlert(alertId) {",
     "async function dismissSingleAlert(alertId) {\n      alert('Static dashboard is read-only.'); return;"),
    ("async function clearAllAlerts() {",
     "async function clearAllAlerts() {\n      alert('Static dashboard is read-only.'); return;"),
    ("async function triggerReset() {",
     "async function triggerReset() {\n      alert('Static dashboard is read-only. Run: python3 reset.py'); return;"),
]


def build_html() -> str:
    from web_server import DASHBOARD_HTML
    html = DASHBOARD_HTML
    for old, new in STATIC_PATCHES:
        if old not in html:
            raise RuntimeError(f"Dashboard template changed; patch target missing: {old!r}")
        html = html.replace(old, new)
    return html


def atomic_write(path: Path, text: str):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def write_status(out: Path):
    from trading_engine import manager
    atomic_write(out / "status.json", json.dumps(manager.get_full_dashboard_data(), default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE_DIR / "site"))
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write(out / "index.html", build_html())
    write_status(out)
    print(f"Wrote {out}/index.html and status.json")
    if a.watch:
        import asyncio
        from trading_engine import background_market_watcher, manager

        async def run():
            task = asyncio.create_task(background_market_watcher(poll_interval=2.5))
            try:
                while True:
                    await asyncio.sleep(a.interval)
                    write_status(out)
            finally:
                manager.is_running = False
                task.cancel()
        asyncio.run(run())


if __name__ == "__main__":
    main()
