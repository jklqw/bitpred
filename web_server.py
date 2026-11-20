"""FastAPI Web Server for Real-Time Bitcoin 4x Leveraged PnL Monitoring Dashboard."""

import os
import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

# Auto-bootstrap virtualenv site-packages if running with system python
BASE_DIR = Path(__file__).resolve().parent
_site_packages = BASE_DIR / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _site_packages.is_dir() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from trading_engine import manager, background_market_watcher, PERIOD_CONFIGS


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Launch always-active background market monitoring task
    task = asyncio.create_task(background_market_watcher(poll_interval=2.5))
    yield
    manager.is_running = False
    task.cancel()


app = FastAPI(
    title="Bitcoin Alpha 4x Leveraged Trading Monitor",
    description="Live real-time monitoring dashboard for 4x leveraged Bitcoin multi-period trading strategies.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
async def get_status():
    """Return live status across all 5 period trading strategies."""
    return JSONResponse(content=manager.get_full_dashboard_data())


@app.get("/api/period/{period_id}")
async def get_period_detail(period_id: str):
    """Return detailed stats, equity curve, and history for a specific period."""
    if period_id not in manager.trackers:
        raise HTTPException(status_code=404, detail=f"Period '{period_id}' not found.")
    tr = manager.trackers[period_id]
    return JSONResponse(content=tr.to_dict(manager.latest_live_price))


@app.post("/api/reset")
async def reset_simulation():
    """Reset all period balances back to 1,000 USDT initial capital starting from right now."""
    await asyncio.to_thread(manager.reset_all, clean_start=True)
    return JSONResponse(content={
        "status": "success",
        "message": "All 5 periods successfully reset to 1,000 USDT with exact start timestamps.",
        "exact_start_time_utc": manager.last_update_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "live_btc_price": manager.latest_live_price,
        "portfolio": manager.get_full_dashboard_data()
    })


@app.get("/api/alerts")
async def get_alerts():
    """Return live alert history and active alerts across all period strategies."""
    active_alerts = list(reversed([a for a in manager.alerts if not a.get("dismissed", False)]))
    return JSONResponse(content={
        "status": "success",
        "active_alerts_count": len(active_alerts),
        "total_alerts": len(manager.alerts),
        "alerts": list(reversed(manager.alerts[-50:])),
        "active_alerts": active_alerts
    })


@app.post("/api/alerts/clear")
async def clear_all_alerts():
    """Clear all triggered alerts across all strategies."""
    manager.clear_alerts()
    return JSONResponse(content={
        "status": "success",
        "message": "All strategy alerts have been cleared."
    })


@app.post("/api/alerts/dismiss/{alert_id}")
async def dismiss_single_alert(alert_id: str):
    """Dismiss a specific alert by its ID."""
    dismissed = manager.dismiss_alert(alert_id)
    if not dismissed:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return JSONResponse(content={
        "status": "success",
        "alert_id": alert_id,
        "message": f"Alert {alert_id} dismissed."
    })


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BTC 4x Leveraged Live Trading Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #0a0d14;
      --bg-surface: #101522;
      --bg-card: #151c2e;
      --bg-card-hover: #1b243b;
      --border-subtle: #1f2a44;
      --border-bright: #2d3c60;
      --text-main: #f0f4fc;
      --text-muted: #8c9bb5;
      --text-dim: #546482;
      --accent-green: #00f090;
      --accent-green-bg: rgba(0, 240, 144, 0.12);
      --accent-red: #ff3b69;
      --accent-red-bg: rgba(255, 59, 105, 0.12);
      --accent-blue: #3b82f6;
      --accent-amber: #f59e0b;
      --accent-purple: #a855f7;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      overflow-x: hidden;
    }

    .mono {
      font-family: 'JetBrains Mono', monospace;
    }

    /* Top Navigation Bar */
    header {
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-subtle);
      padding: 14px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(12px);
    }

    .brand-section {
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .logo-badge {
      background: linear-gradient(135deg, #f7931a, #e67e00);
      color: #000;
      font-weight: 800;
      font-size: 16px;
      padding: 6px 12px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
      box-shadow: 0 0 20px rgba(247, 147, 26, 0.35);
    }

    .brand-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.3px;
    }

    .server-status {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      background: rgba(0, 240, 144, 0.1);
      color: var(--accent-green);
      padding: 4px 10px;
      border-radius: 20px;
      border: 1px solid rgba(0, 240, 144, 0.25);
    }

    .pulsing-dot {
      width: 8px;
      height: 8px;
      background-color: var(--accent-green);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--accent-green);
      animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 240, 144, 0.7); }
      70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(0, 240, 144, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 240, 144, 0); }
    }

    .header-metrics {
      display: flex;
      align-items: center;
      gap: 24px;
    }

    .header-metric-box {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
    }

    .metric-label {
      font-size: 11px;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .metric-value-lg {
      font-size: 20px;
      font-weight: 700;
    }

    /* Container */
    main {
      max-width: 1560px;
      margin: 0 auto;
      padding: 24px 28px 60px;
    }

    /* Tab Switcher */
    .tabs-bar {
      display: flex;
      gap: 10px;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 16px;
      margin-bottom: 24px;
      overflow-x: auto;
    }

    .tab-btn {
      background: var(--bg-surface);
      color: var(--text-muted);
      border: 1px solid var(--border-subtle);
      padding: 10px 18px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
      white-space: nowrap;
    }

    .tab-btn:hover {
      background: var(--bg-card);
      color: var(--text-main);
      border-color: var(--border-bright);
    }

    .tab-btn.active {
      background: var(--accent-blue);
      color: #fff;
      border-color: var(--accent-blue);
      box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
    }

    .tab-badge {
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 6px;
      background: rgba(0,0,0,0.3);
    }

    /* Overview Grid */
    .overview-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 20px;
      margin-bottom: 28px;
    }

    .period-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      cursor: pointer;
      transition: all 0.25s ease;
      position: relative;
      overflow: hidden;
    }

    .period-card:hover {
      transform: translateY(-3px);
      border-color: var(--border-bright);
      box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    }

    .period-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .period-title {
      font-size: 16px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .leverage-pill {
      font-size: 11px;
      font-weight: 700;
      background: var(--bg-card);
      color: var(--accent-amber);
      border: 1px solid rgba(245, 158, 11, 0.3);
      padding: 2px 7px;
      border-radius: 6px;
    }

    .action-badge {
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .badge-long {
      background: var(--accent-green-bg);
      color: var(--accent-green);
      border: 1px solid rgba(0, 240, 144, 0.3);
    }

    .badge-short {
      background: var(--accent-red-bg);
      color: var(--accent-red);
      border: 1px solid rgba(255, 59, 105, 0.3);
    }

    .pnl-hero-box {
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding: 12px 14px;
      border-radius: 12px;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
    }

    .pnl-num {
      font-size: 24px;
      font-weight: 800;
      display: flex;
      align-items: baseline;
      gap: 8px;
    }

    .pnl-positive { color: var(--accent-green); }
    .pnl-negative { color: var(--accent-red); }

    .card-meta-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      font-size: 12px;
    }

    .meta-item {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .meta-k { color: var(--text-dim); }
    .meta-v { font-weight: 600; }

    .progress-bar-container {
      width: 100%;
      height: 6px;
      background: var(--bg-card);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 4px;
    }

    .progress-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
      border-radius: 4px;
      transition: width 0.5s ease;
    }

    /* Single Period Detail View */
    .detail-view {
      display: none;
      flex-direction: column;
      gap: 24px;
    }

    .detail-hero-grid {
      display: grid;
      grid-template-columns: 1.2fr 1.8fr;
      gap: 24px;
    }

    @media (max-width: 1024px) {
      .detail-hero-grid { grid-template-columns: 1fr; }
    }

    .hero-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 20px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .confidence-pill {
      font-size: 12px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 8px;
      background: rgba(168, 85, 247, 0.15);
      color: var(--accent-purple);
      border: 1px solid rgba(168, 85, 247, 0.3);
    }

    /* Price levels display */
    .levels-strip {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 14px;
    }

    .level-box {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .level-val {
      font-size: 15px;
      font-weight: 700;
    }

    .liq-danger {
      color: var(--accent-red);
      font-weight: 700;
    }

    /* Table */
    .table-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 18px;
      padding: 22px;
      overflow-x: auto;
    }

    .table-title {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    th {
      text-align: left;
      padding: 12px 14px;
      color: var(--text-dim);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.5px;
      border-bottom: 1px solid var(--border-subtle);
    }

    td {
      padding: 14px;
      border-bottom: 1px solid rgba(31, 42, 68, 0.6);
    }

    tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    .rules-banner {
      background: rgba(247, 147, 26, 0.08);
      border: 1px solid rgba(247, 147, 26, 0.25);
      border-radius: 12px;
      padding: 14px 18px;
      font-size: 12px;
      color: #f3ba2f;
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 24px;
    }

    /* Chart SVG */
    .chart-container {
      width: 100%;
      height: 180px;
      position: relative;
    }

    /* Alerts Styling */
    .alert-banner-box {
      background: linear-gradient(90deg, rgba(255, 59, 105, 0.16), rgba(245, 158, 11, 0.12));
      border: 1px solid rgba(255, 59, 105, 0.5);
      border-radius: 14px;
      padding: 14px 20px;
      margin-bottom: 22px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      box-shadow: 0 0 25px rgba(255, 59, 105, 0.2);
      animation: alertPulse 2.5s infinite ease-in-out;
    }

    @keyframes alertPulse {
      0%, 100% { border-color: rgba(255, 59, 105, 0.4); box-shadow: 0 0 15px rgba(255, 59, 105, 0.15); }
      50% { border-color: rgba(255, 59, 105, 0.9); box-shadow: 0 0 30px rgba(255, 59, 105, 0.35); }
    }

    .alert-banner-left {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .alert-pulse-icon {
      font-size: 24px;
      animation: alertBounce 1.5s infinite ease-in-out;
    }

    @keyframes alertBounce {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.15); }
    }

    .alert-banner-text {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .alert-banner-title {
      font-size: 14px;
      font-weight: 800;
      color: #ff3b69;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .alert-banner-desc {
      font-size: 13px;
      color: var(--text-main);
    }

    .alert-banner-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .alert-btn-action {
      background: var(--accent-blue);
      color: #fff;
      border: none;
      padding: 8px 16px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 12px;
      cursor: pointer;
      white-space: nowrap;
    }

    .alert-btn-dismiss {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-muted);
      border: 1px solid var(--border-subtle);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
    }
    .alert-btn-dismiss:hover {
      background: rgba(255, 59, 105, 0.2);
      color: var(--accent-red);
    }

    .period-card.has-alert {
      border: 1px solid #ff3b69 !important;
      box-shadow: 0 0 22px rgba(255, 59, 105, 0.35) !important;
      animation: alertPulse 2.5s infinite ease-in-out;
    }

    .card-alert-badge {
      background: rgba(255, 59, 105, 0.16);
      border: 1px solid rgba(255, 59, 105, 0.4);
      color: #ff4d79;
      font-size: 11px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 8px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .badge-critical {
      background: rgba(255, 59, 105, 0.2);
      color: #ff3b69;
      border: 1px solid rgba(255, 59, 105, 0.5);
      font-size: 10px;
      font-weight: 800;
      padding: 3px 8px;
      border-radius: 6px;
      text-transform: uppercase;
    }

    .badge-warning {
      background: rgba(245, 158, 11, 0.2);
      color: #f59e0b;
      border: 1px solid rgba(245, 158, 11, 0.5);
      font-size: 10px;
      font-weight: 800;
      padding: 3px 8px;
      border-radius: 6px;
      text-transform: uppercase;
    }
  </style>
</head>
<body>

  <!-- Top Header -->
  <header>
    <div class="brand-section">
      <div class="logo-badge">
        <span>⚡</span> BTC 4X
      </div>
      <div>
        <div class="brand-title">Bitcoin Live Multi-Period Leveraged Monitor</div>
        <div style="font-size: 12px; color: var(--text-dim);">GDELT News & Price Predictive Automated Engine</div>
      </div>
      <div class="server-status">
        <div class="pulsing-dot"></div>
        <span>SERVER ALWAYS ACTIVE & MONITORING</span>
      </div>
    </div>

    <div class="header-metrics">
      <div class="header-metric-box">
        <span class="metric-label">Live BTC/USDT</span>
        <span id="header-btc-price" class="metric-value-lg mono" style="color: var(--accent-amber);">$---,--</span>
      </div>
      <div class="header-metric-box">
        <span class="metric-label">Top Strategy Balance (Max of 5)</span>
        <div style="display: flex; align-items: baseline; gap: 6px;">
          <span id="header-max-balance" class="metric-value-lg mono" style="color: var(--accent-green);">$---.-- USDT</span>
          <span id="header-max-balance-period" style="font-size: 11px; color: var(--text-dim); font-weight: 600;">(---)</span>
        </div>
      </div>
      <div class="header-metric-box">
        <span class="metric-label">Max Net Return</span>
        <div style="display: flex; align-items: baseline; gap: 6px;">
          <span id="header-max-return" class="metric-value-lg mono">0.00%</span>
          <span id="header-max-return-period" style="font-size: 11px; color: var(--text-dim); font-weight: 600;">(---)</span>
        </div>
      </div>
      <div style="display: flex; align-items: center; margin-left: 8px;">
        <button id="btn-reset" onclick="triggerReset()" style="background: rgba(239, 68, 68, 0.15); border: 1px solid var(--accent-red); color: var(--accent-red); padding: 8px 16px; border-radius: 8px; font-weight: 700; cursor: pointer; display: flex; align-items: center; gap: 6px; font-size: 13px;">
          <span>🔄</span> Reset to 1,000 USDT
        </button>
      </div>
    </div>
  </header>

  <main>
    <!-- Dynamic Direction Reversal & Hit Prob Alert Banner -->
    <div id="top-alert-banner" class="alert-banner-box" style="display: none;">
      <div class="alert-banner-left">
        <span class="alert-pulse-icon">🚨</span>
        <div class="alert-banner-text">
          <span id="alert-banner-title" class="alert-banner-title">STRATEGY DIRECTION REVERSAL ALERT</span>
          <span id="alert-banner-desc" class="alert-banner-desc"></span>
        </div>
      </div>
      <div class="alert-banner-actions">
        <button class="alert-btn-action" onclick="switchTab('alerts')">View Alerts Log</button>
        <button class="alert-btn-dismiss" onclick="dismissBannerAlert()">Dismiss ✕</button>
      </div>
    </div>

    <!-- Rules Confirmation Banner -->
    <div class="rules-banner">
      <span style="font-size: 18px;">ℹ️</span>
      <div>
        <strong>Execution Specifications:</strong> 4x Leverage | Initial Capital: 1,000.00 USDT each | Opening Fee: 0.1% (0.3% for 3-Day mode) | Slippage: Buy +0.01% / Sell -0.01% | Liquidation: -23% adverse price move triggers full liquidation.
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tabs-bar">
      <button class="tab-btn active" onclick="switchTab('overview')">
        <span>📊 All Periods Overview</span>
        <span class="tab-badge">5 Parallel</span>
      </button>
      <button class="tab-btn" onclick="switchTab('3h')">
        <span>⏱️ 3 Hours</span>
        <span id="tab-pnl-3h" class="tab-badge mono">--%</span>
      </button>
      <button class="tab-btn" onclick="switchTab('6h')">
        <span>⏱️ 6 Hours</span>
        <span id="tab-pnl-6h" class="tab-badge mono">--%</span>
      </button>
      <button class="tab-btn" onclick="switchTab('12h')">
        <span>⏱️ 12 Hours</span>
        <span id="tab-pnl-12h" class="tab-badge mono">--%</span>
      </button>
      <button class="tab-btn" onclick="switchTab('1d')">
        <span>📅 1 Day (24h)</span>
        <span id="tab-pnl-1d" class="tab-badge mono">--%</span>
      </button>
      <button class="tab-btn" onclick="switchTab('3d')">
        <span>📅 3 Days (72h)</span>
        <span id="tab-pnl-3d" class="tab-badge mono">--%</span>
      </button>
      <button class="tab-btn" id="tab-btn-alerts" onclick="switchTab('alerts')">
        <span>🔔 Alerts</span>
        <span id="tab-alerts-badge" class="tab-badge mono" style="background: rgba(255, 59, 105, 0.25); color: #ff3b69; font-weight: 700; display: none;">0</span>
      </button>
    </div>

    <!-- VIEW 1: All Periods Overview Matrix -->
    <div id="view-overview">
      <div class="overview-grid" id="overview-cards-container">
        <!-- Rendered via JavaScript -->
      </div>

      <!-- Recent Cross-Strategy Completed Trades -->
      <div class="table-card">
        <div class="table-title">
          <span>📜</span> Recent Closed Trades Across All Periods
        </div>
        <table>
          <thead>
            <tr>
              <th>Period</th>
              <th>Trade ID</th>
              <th>Direction</th>
              <th>Entry Price</th>
              <th>Exit Price</th>
              <th>Net PnL (USD)</th>
              <th>Return %</th>
              <th>Fee Paid</th>
              <th>Confidence</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="global-trades-tbody">
            <!-- Rendered via JavaScript -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- VIEW 2: Single Period Detail View -->
    <div id="view-detail" class="detail-view">
      <div class="detail-hero-grid">
        <!-- Left: Active Trade Hero -->
        <div class="hero-card" id="detail-active-trade-card">
          <!-- Rendered via JavaScript -->
        </div>

        <!-- Right: Strategy Performance & Chart -->
        <div class="hero-card" id="detail-strategy-metrics-card">
          <!-- Rendered via JavaScript -->
        </div>
      </div>

      <!-- Period Trade History -->
      <div class="table-card">
        <div class="table-title">
          <span>📜</span> <span id="detail-table-title">Period</span> Trade Execution History
        </div>
        <table>
          <thead>
            <tr>
              <th>Trade ID</th>
              <th>Direction</th>
              <th>Entry Price</th>
              <th>Exit Price</th>
              <th>Realized PnL ($)</th>
              <th>PnL % (4x)</th>
              <th>Fee (BTC)</th>
              <th>Balance After</th>
              <th>Confidence</th>
              <th>Outcome</th>
            </tr>
          </thead>
          <tbody id="detail-trades-tbody">
            <!-- Rendered via JavaScript -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- VIEW 3: Strategy Direction & Hit Prob Alerts History -->
    <div id="view-alerts" class="detail-view">
      <div class="table-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 12px;">
          <div class="table-title" style="margin-bottom: 0;">
            <span>🔔</span> Live Strategy Alerts Log (Direction Reversals & Significant Hit Prob Swings)
          </div>
          <div style="display: flex; gap: 10px;">
            <button onclick="clearAllAlerts()" style="background: rgba(255, 59, 105, 0.15); border: 1px solid var(--accent-red); color: var(--accent-red); padding: 7px 14px; border-radius: 8px; font-weight: 700; cursor: pointer; font-size: 12px;">
              Clear All Alerts
            </button>
          </div>
        </div>
        <table>
          <thead>
            <tr>
              <th>Timestamp (UTC)</th>
              <th>Period</th>
              <th>Severity</th>
              <th>Alert Type</th>
              <th>Direction Change</th>
              <th>Hit Prob Shift</th>
              <th>BTC Price</th>
              <th>Details</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="alerts-tbody">
            <!-- Rendered via JavaScript -->
          </tbody>
        </table>
      </div>
    </div>
  </main>

  <script>
    let globalData = null;
    let currentTab = 'overview';

    function escapeHtml(value) {
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    async function fetchStatus() {
      try {
        const resp = await fetch('/api/status');
        if (!resp.ok) return;
        globalData = await resp.json();
        renderHeader();
        if (currentTab === 'overview') {
          renderOverview();
        } else if (currentTab === 'alerts') {
          await renderAlertsView();
        } else {
          renderDetailView(currentTab);
        }
      } catch (e) {
        console.error("Failed to fetch live status:", e);
      }
    }

    function renderHeader() {
      if (!globalData) return;
      document.getElementById('header-btc-price').textContent = `$${globalData.live_btc_price.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
      
      const maxBalEl = document.getElementById('header-max-balance');
      if (maxBalEl && globalData.max_balance_usdt !== undefined) {
        maxBalEl.textContent = `$${globalData.max_balance_usdt.toLocaleString('en-US', {minimumFractionDigits: 2})} USDT`;
      }
      const maxBalPeriodEl = document.getElementById('header-max-balance-period');
      if (maxBalPeriodEl && globalData.max_balance_period) {
        maxBalPeriodEl.textContent = `(${globalData.max_balance_period})`;
      }

      const retEl = document.getElementById('header-max-return');
      const ret = globalData.max_return_pct;
      if (retEl && ret !== undefined) {
        retEl.textContent = `${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%`;
        retEl.className = `metric-value-lg mono ${ret >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
      }
      const retPeriodEl = document.getElementById('header-max-return-period');
      if (retPeriodEl && globalData.max_return_period) {
        retPeriodEl.textContent = `(${globalData.max_return_period})`;
      }

      // Update Top Dynamic Alert Banner
      const banner = document.getElementById('top-alert-banner');
      const alerts = globalData.alerts || [];
      const activeAlerts = alerts.filter(a => a && !a.dismissed);
      if (activeAlerts.length > 0 && banner) {
        const topAlert = activeAlerts[0];
        const priceAtAlert = Number(topAlert.price_at_alert) || 0;
        const oldHitProb = Number(topAlert.old_hit_prob) || 0;
        const newHitProb = Number(topAlert.new_hit_prob) || 0;
        const probChangePct = Number(topAlert.prob_change_pct) || 0;
        banner.style.display = 'flex';
        document.getElementById('alert-banner-title').textContent = topAlert.title || '🚨 Direction Reversal Alert!';
        document.getElementById('alert-banner-desc').innerHTML = `
          <strong>${escapeHtml(topAlert.period_name)}:</strong> Flipped from <strong>${escapeHtml(topAlert.old_direction)} (${oldHitProb}%)</strong> ➔ 
          <strong style="color: ${topAlert.new_direction === 'LONG' ? 'var(--accent-green)' : 'var(--accent-red)'}; font-size: 14px;">${escapeHtml(topAlert.new_direction)} (${newHitProb}% hit prob)</strong>! 
          BTC Price: <strong>$${priceAtAlert ? priceAtAlert.toLocaleString() : '---'}</strong> (${probChangePct >= 0 ? '+' : ''}${probChangePct}% prob shift).
        `;
        banner.setAttribute('data-top-alert-id', topAlert.alert_id);
      } else if (banner) {
        banner.style.display = 'none';
      }

      // Update Tab Badges
      for (const [pid, p] of Object.entries(globalData.periods)) {
        const badge = document.getElementById(`tab-pnl-${pid}`);
        if (badge) {
          const r = p.total_return_pct;
          badge.textContent = `${r >= 0 ? '+' : ''}${r.toFixed(1)}%`;
          badge.style.color = r >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
        }
      }

      // Update Alerts Tab Badge
      const alertBadge = document.getElementById('tab-alerts-badge');
      if (alertBadge) {
        const cnt = globalData.active_alerts_count !== undefined ? globalData.active_alerts_count : activeAlerts.length;
        alertBadge.textContent = cnt;
        alertBadge.style.display = cnt > 0 ? 'inline-block' : 'none';
      }
    }

    function renderOverview() {
      if (!globalData) return;
      const container = document.getElementById('overview-cards-container');
      container.innerHTML = '';

      for (const [pid, p] of Object.entries(globalData.periods)) {
        const t = p.active_trade;
        const isLong = t && t.direction === 'LONG';
        const pnl = t ? (t.unrealized_pnl_usdt !== undefined ? t.unrealized_pnl_usdt : (t.unrealized_pnl_usd || 0)) : 0;
        const pnlPct = t ? t.unrealized_pnl_pct : 0;
        const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';

        // Real-time Hit Probabilities & Deltas
        const liveHit = p.live_hit_probability_pct !== undefined ? p.live_hit_probability_pct : (t ? t.live_hit_probability_pct : 50.0);
        const entryHit = p.entry_hit_probability_pct !== undefined ? p.entry_hit_probability_pct : (t ? t.entry_hit_probability_pct : 50.0);
        const deltaHit = p.live_hit_prob_delta !== undefined ? p.live_hit_prob_delta : (t ? t.live_hit_prob_delta : 0.0);
        const deltaClass = deltaHit >= 0 ? 'pnl-positive' : 'pnl-negative';
        const deltaSign = deltaHit >= 0 ? '+' : '';
        const deltaIcon = deltaHit >= 0 ? '▲' : '▼';
        const liveConfidence = p.live_confidence || (t ? t.confidence : 'Active');
        const probUp = p.live_prob_up !== undefined && p.live_prob_up !== null ? p.live_prob_up : (t ? t.prob_up : 50.0);
        const probDown = p.live_prob_down !== undefined && p.live_prob_down !== null ? p.live_prob_down : (t ? t.prob_down : 50.0);

        const hasAlert = p.active_alert && !p.active_alert.dismissed;
        const card = document.createElement('div');
        card.className = `period-card ${hasAlert ? 'has-alert' : ''}`;
        card.onclick = () => switchTab(pid);

        card.innerHTML = `
          <div class="period-card-header">
            <div class="period-title">
              <span>${p.name}</span>
              <span class="leverage-pill">${p.leverage}</span>
            </div>
            ${t ? `
              <div class="action-badge ${isLong ? 'badge-long' : 'badge-short'}">
                ${t.direction_icon}
              </div>
            ` : '<span style="font-size: 11px; color: var(--text-dim);">NO ACTIVE TRADE</span>'}
          </div>

          ${hasAlert ? `
            <div class="card-alert-badge">
              <span>🚨</span>
              <span><strong>${p.active_alert.type === 'DIRECTION_REVERSAL' ? 'Direction Reversal' : 'Prob Swing'}:</strong> ${escapeHtml(p.active_alert.old_direction)} ➔ <span style="color: ${p.active_alert.new_direction === 'LONG' ? 'var(--accent-green)' : 'var(--accent-red)'}; font-weight: 800;">${escapeHtml(p.active_alert.new_direction)}</span> (${Number(p.active_alert.new_hit_prob) || 0}% Hit Prob)</span>
            </div>
          ` : ''}

          <!-- PROMINENT LIVE HIT PROBABILITY HERO BOX -->
          <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 10px 14px; margin: 8px 0; display: flex; justify-content: space-between; align-items: center;">
            <div>
              <div style="font-size: 10px; color: var(--text-dim); font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 5px;">
                <span class="pulsing-dot" style="width: 7px; height: 7px; background-color: var(--accent-green);"></span>
                <span>🎯 LIVE HIT PROB</span>
              </div>
              <div style="display: flex; align-items: baseline; gap: 6px; margin-top: 3px;">
                <span class="mono" style="font-size: 21px; font-weight: 800; color: var(--accent-green);">${liveHit.toFixed(1)}%</span>
                <span class="mono ${deltaClass}" style="font-size: 11px; font-weight: 700;">
                  (${deltaIcon} ${deltaSign}${deltaHit.toFixed(1)}%)
                </span>
              </div>
            </div>
            <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 3px;">
              <div style="font-size: 11px; color: var(--text-muted);">Entry Hit: <strong class="mono" style="color: var(--text-main);">${entryHit.toFixed(1)}%</strong></div>
              <span class="confidence-pill" style="font-size: 10px; padding: 2px 7px;">${liveConfidence}</span>
            </div>
          </div>

          <div class="pnl-hero-box">
            <span class="metric-label">Live Unrealized PnL (4x)</span>
            <div class="pnl-num mono ${pnlClass}">
              <span>${pnl >= 0 ? '+' : ''}$${pnl.toLocaleString('en-US', {minimumFractionDigits: 2})} USDT</span>
              <span style="font-size: 14px;">(${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%)</span>
            </div>
          </div>

          <!-- Live Signal Up/Down Probability Distribution -->
          <div style="font-size: 10px; color: var(--text-dim); display: flex; justify-content: space-between; margin: 6px 0 3px 0;" class="mono">
            <span>🟢 Up: <strong style="color: var(--accent-green);">${probUp.toFixed(1)}%</strong></span>
            <span>🔴 Down: <strong style="color: var(--accent-red);">${probDown.toFixed(1)}%</strong></span>
          </div>
          <div class="progress-bar-container" style="height: 4px; background: rgba(255, 59, 105, 0.4); margin-bottom: 12px; border-radius: 2px;">
            <div class="progress-bar-fill" style="width: ${probUp}%; background: var(--accent-green); border-radius: 2px;"></div>
          </div>

          <div class="card-meta-grid mono">
            <div class="meta-item">
              <span class="meta-k">Strategy Equity</span>
              <span class="meta-v" style="color: var(--accent-green);">$${p.current_equity_usdt.toLocaleString('en-US', {minimumFractionDigits: 2})} USDT</span>
            </div>
            <div class="meta-item">
              <span class="meta-k">Total Return</span>
              <span class="meta-v ${p.total_return_pct >= 0 ? 'pnl-positive' : 'pnl-negative'}">${p.total_return_pct >= 0 ? '+' : ''}${p.total_return_pct.toFixed(2)}%</span>
            </div>
            <div class="meta-item">
              <span class="meta-k">Entry Price</span>
              <span class="meta-v">${t ? '$' + t.entry_price.toLocaleString() : '---'}</span>
            </div>
            <div class="meta-item">
              <span class="meta-k">Liquidation</span>
              <span class="meta-v liq-danger">${t ? '$' + t.liquidation_price.toLocaleString() : '---'}</span>
            </div>
          </div>

          ${t ? `
            <div style="font-size: 11px; color: var(--text-dim); display: flex; justify-content: space-between; margin: 10px 0 6px 0;">
              <span>🕒 Opened: <strong class="mono" style="color: var(--text-main);">${t.opened_at_exact || t.start_time}</strong></span>
              <span>Target: <strong class="mono" style="color: var(--accent-amber);">$${t.target_price ? t.target_price.toLocaleString() : '---'}</strong></span>
            </div>
            <div>
              <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
                <span style="color: var(--text-dim);">Period Progress</span>
                <span class="mono">${t.time_remaining_str} left</span>
              </div>
              <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: ${t.progress_pct}%;"></div>
              </div>
            </div>
          ` : ''}

          <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--border-subtle); padding-top: 12px; margin-top: 10px; font-size: 11px;">
            <span style="color: var(--text-dim);">Win Rate: <strong class="mono" style="color: var(--text-main);">${p.win_rate_pct}%</strong> (${p.winning_trades}/${p.total_trades})</span>
            <span style="color: var(--accent-blue); font-weight: 600;">View Details →</span>
          </div>
        `;
        container.appendChild(card);
      }

      // Render Global Trades
      const tbody = document.getElementById('global-trades-tbody');
      tbody.innerHTML = '';
      const recent = globalData.recent_global_trades || [];
      if (recent.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: var(--text-dim);">No closed trades yet</td></tr>';
        return;
      }

      recent.forEach(tr => {
        const netPnl = tr.net_pnl_usdt !== undefined ? tr.net_pnl_usdt : (tr.net_pnl_usd || 0);
        const isWin = netPnl >= 0;
        const fee = tr.fee_usdt !== undefined ? `$${tr.fee_usdt.toFixed(2)} USDT` : `${tr.fee_btc} BTC`;
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><strong style="text-transform: uppercase;">${tr.period}</strong></td>
          <td class="mono" style="font-size: 11px; color: var(--text-dim);">${tr.trade_id}</td>
          <td><span class="action-badge ${tr.direction === 'LONG' ? 'badge-long' : 'badge-short'}" style="display: inline-flex;">${tr.direction_icon}</span></td>
          <td class="mono">$${tr.entry_price ? tr.entry_price.toLocaleString() : '---'}</td>
          <td class="mono">$${tr.exit_price ? tr.exit_price.toLocaleString() : '---'}</td>
          <td class="mono ${isWin ? 'pnl-positive' : 'pnl-negative'}"><strong>${isWin ? '+' : ''}$${netPnl.toLocaleString('en-US', {minimumFractionDigits: 2})} USDT</strong></td>
          <td class="mono ${isWin ? 'pnl-positive' : 'pnl-negative'}">${isWin ? '+' : ''}${tr.pnl_pct.toFixed(2)}%</td>
          <td class="mono" style="color: var(--text-dim);">${fee}</td>
          <td><span class="confidence-pill" style="font-size: 10px;">${tr.confidence || 'Model Agreed'}</span></td>
          <td><span style="font-size: 11px; font-weight: 700; color: ${tr.status === 'LIQUIDATED' ? 'var(--accent-red)' : 'var(--accent-green)'};">${tr.status}</span></td>
        `;
        tbody.appendChild(row);
      });
    }

    function renderDetailView(pid) {
      if (!globalData || !globalData.periods[pid]) return;
      const p = globalData.periods[pid];
      const t = p.active_trade;
      const isLong = t && t.direction === 'LONG';
      const livePrice = globalData.live_btc_price;

      document.getElementById('detail-table-title').textContent = `${p.name}`;

      // Left Card: Active Trade Hero
      const leftCard = document.getElementById('detail-active-trade-card');
      if (t) {
        const pnl = t.unrealized_pnl_usdt !== undefined ? t.unrealized_pnl_usdt : (t.unrealized_pnl_usd || 0);
        const pnlPct = t.unrealized_pnl_pct;
        const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
        const liqDistance = Math.abs((livePrice - t.liquidation_price) / livePrice * 100);

        // Real-time Hit Probabilities & Deltas
        const liveHit = p.live_hit_probability_pct !== undefined ? p.live_hit_probability_pct : (t.live_hit_probability_pct || 50.0);
        const entryHit = p.entry_hit_probability_pct !== undefined ? p.entry_hit_probability_pct : (t.entry_hit_probability_pct || 50.0);
        const deltaHit = p.live_hit_prob_delta !== undefined ? p.live_hit_prob_delta : (t.live_hit_prob_delta || 0.0);
        const deltaClass = deltaHit >= 0 ? 'pnl-positive' : 'pnl-negative';
        const deltaSign = deltaHit >= 0 ? '+' : '';
        const deltaIcon = deltaHit >= 0 ? '▲' : '▼';
        const liveConfidence = p.live_confidence || t.confidence;
        const probUp = p.live_prob_up !== undefined && p.live_prob_up !== null ? p.live_prob_up : t.prob_up;
        const probDown = p.live_prob_down !== undefined && p.live_prob_down !== null ? p.live_prob_down : t.prob_down;

        leftCard.innerHTML = `
          ${p.active_alert && !p.active_alert.dismissed ? `
            <div style="background: rgba(255, 59, 105, 0.16); border: 1px solid rgba(255, 59, 105, 0.5); border-radius: 12px; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 8px;">
              <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 22px;">🚨</span>
                <div>
                  <div style="font-size: 13px; font-weight: 800; color: #ff3b69;">${escapeHtml(p.active_alert.title)}</div>
                  <div style="font-size: 12px; color: var(--text-main); margin-top: 2px;">${escapeHtml(p.active_alert.message)}</div>
                </div>
              </div>
              <button onclick="dismissSingleAlert('${escapeHtml(p.active_alert.alert_id)}')" class="alert-btn-dismiss" style="padding: 4px 10px;">Dismiss</button>
            </div>
          ` : ''}
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 10px;">
              <span style="font-size: 20px; font-weight: 800;">${p.name} Active Position</span>
              <span class="leverage-pill">${p.leverage}</span>
            </div>
            <div class="action-badge ${isLong ? 'badge-long' : 'badge-short'}" style="font-size: 14px; padding: 6px 14px;">
              ${t.direction_icon}
            </div>
          </div>

          <div style="font-size: 12px; color: var(--text-dim); display: flex; gap: 16px; margin: 4px 0 8px 0;">
            <span>🕒 <strong>Exact Open Time:</strong> <span class="mono" style="color: var(--accent-blue);">${t.opened_at_exact || t.start_time}</span></span>
            <span>⏳ <strong>Target Close:</strong> <span class="mono">${t.end_at_exact || t.end_time}</span></span>
          </div>

          <div class="pnl-hero-box" style="padding: 16px 20px;">
            <span class="metric-label" style="font-size: 12px;">Live 4x Leveraged Unrealized PnL</span>
            <div class="pnl-num mono ${pnlClass}" style="font-size: 32px;">
              <span>${pnl >= 0 ? '+' : ''}$${pnl.toLocaleString('en-US', {minimumFractionDigits: 2})} USDT</span>
              <span style="font-size: 18px;">(${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%)</span>
            </div>
            <span style="font-size: 12px; color: var(--text-dim);" class="mono">
              Margin: $${t.margin_usdt.toLocaleString()} USDT | Notional: $${(t.position_notional_usdt || t.margin_usdt * 4).toLocaleString()} USDT (${t.position_qty_btc.toFixed(4)} BTC)
            </span>
          </div>

          <div class="levels-strip mono">
            <div class="level-box">
              <span class="meta-k">Entry Price (+0.01% Slip)</span>
              <span class="level-val">$${t.entry_price.toLocaleString()}</span>
            </div>
            <div class="level-box">
              <span class="meta-k">Current Live Mark</span>
              <span class="level-val" style="color: var(--accent-amber);">$${livePrice.toLocaleString()}</span>
            </div>
            <div class="level-box">
              <span class="meta-k">Liquidation Level</span>
              <span class="level-val liq-danger">$${t.liquidation_price.toLocaleString()} (${liqDistance.toFixed(1)}% away)</span>
            </div>
            <div class="level-box">
              <span class="meta-k">Model Target</span>
              <span class="level-val" style="color: var(--accent-green);">$${t.target_price.toLocaleString()}</span>
            </div>
          </div>

          <!-- LIVE HIT PROBABILITIES & CONVICTION PANEL -->
          <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 14px; padding: 16px; display: flex; flex-direction: column; gap: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <div style="display: flex; align-items: center; gap: 8px;">
                <span class="pulsing-dot" style="width: 8px; height: 8px; background-color: var(--accent-green);"></span>
                <span style="font-size: 14px; font-weight: 800;">Real-Time Signal & Live Hit Probabilities</span>
              </div>
              <div style="display: flex; gap: 8px; align-items: center;">
                <span class="confidence-pill" style="background: rgba(16, 185, 129, 0.2); color: var(--accent-green); font-weight: 800; font-size: 12px;">
                  🎯 Live Hit Prob: ${liveHit.toFixed(1)}%
                </span>
                <span class="confidence-pill">${liveConfidence}</span>
              </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 12px;" class="mono">
              <div class="level-box">
                <span class="meta-k">🎯 Live Hit Prob</span>
                <span class="level-val" style="color: var(--accent-green); font-size: 17px;">
                  ${liveHit.toFixed(1)}%
                  <span style="font-size: 11px;" class="${deltaClass}">(${deltaIcon} ${deltaSign}${deltaHit.toFixed(1)}%)</span>
                </span>
              </div>
              <div class="level-box">
                <span class="meta-k">Entry Hit Prob</span>
                <span class="level-val" style="color: var(--text-main); font-size: 17px;">${entryHit.toFixed(1)}%</span>
              </div>
              <div class="level-box">
                <span class="meta-k">Live Prob Up / Down</span>
                <span class="level-val" style="font-size: 14px;">
                  <span style="color: var(--accent-green);">${probUp.toFixed(1)}%</span> / <span style="color: var(--accent-red);">${probDown.toFixed(1)}%</span>
                </span>
              </div>
              <div class="level-box">
                <span class="meta-k">Expected Delta</span>
                <span class="level-val" style="color: var(--accent-amber); font-size: 15px;">$${t.expected_delta_usd >= 0 ? '+' : ''}${t.expected_delta_usd.toLocaleString()}</span>
              </div>
            </div>

            <!-- Visual Up/Down Split Gauge -->
            <div>
              <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;" class="mono">
                <span style="color: var(--accent-green); font-weight: 700;">🟢 BULLISH CONVICTION: ${probUp.toFixed(1)}%</span>
                <span style="color: var(--accent-red); font-weight: 700;">🔴 BEARISH CONVICTION: ${probDown.toFixed(1)}%</span>
              </div>
              <div class="progress-bar-container" style="height: 6px; background: rgba(255, 59, 105, 0.45); border-radius: 4px;">
                <div class="progress-bar-fill" style="width: ${probUp}%; background: var(--accent-green); border-radius: 4px;"></div>
              </div>
            </div>
          </div>

          <div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;">
              <span style="color: var(--text-dim);">Period Progress (${t.progress_pct.toFixed(1)}%)</span>
              <span class="mono" style="color: var(--accent-blue); font-weight: 600;">Time Remaining: ${t.time_remaining_str}</span>
            </div>
            <div class="progress-bar-container" style="height: 8px;">
              <div class="progress-bar-fill" style="width: ${t.progress_pct}%;"></div>
            </div>
          </div>
        `;
      } else {
        leftCard.innerHTML = `
          <div style="text-align: center; padding: 40px; color: var(--text-dim);">
            <h3>No Active Position</h3>
            <p style="margin-top: 8px;">The engine will open the next period position on the next cycle.</p>
          </div>
        `;
      }

      // Right Card: Strategy Performance & Equity
      const rightCard = document.getElementById('detail-strategy-metrics-card');
      rightCard.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <h3 style="font-size: 18px; font-weight: 700;">Strategy Performance Summary</h3>
          <span style="font-size: 12px; color: var(--text-dim);">Initial: 1,000.00 USDT</span>
        </div>

        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;" class="mono">
          <div class="pnl-hero-box" style="padding: 12px;">
            <span class="meta-k">Current Equity</span>
            <span style="font-size: 18px; font-weight: 800; color: var(--accent-green);">$${p.current_equity_usdt.toLocaleString('en-US', {minimumFractionDigits: 2})} USDT</span>
            <span style="font-size: 11px; color: var(--text-dim);">${(p.current_equity_usdt / livePrice).toFixed(4)} BTC equiv</span>
          </div>
          <div class="pnl-hero-box" style="padding: 12px;">
            <span class="meta-k">Cumulative Return</span>
            <span style="font-size: 18px; font-weight: 800;" class="${p.total_return_pct >= 0 ? 'pnl-positive' : 'pnl-negative'}">${p.total_return_pct >= 0 ? '+' : ''}${p.total_return_pct.toFixed(2)}%</span>
            <span style="font-size: 11px; color: var(--text-dim);">${p.total_trades} Closed Trades</span>
          </div>
          <div class="pnl-hero-box" style="padding: 12px;">
            <span class="meta-k">Win Rate</span>
            <span style="font-size: 18px; font-weight: 800; color: var(--accent-blue);">${p.win_rate_pct}%</span>
            <span style="font-size: 11px; color: var(--text-dim);">${p.winning_trades}W / ${p.total_trades - p.winning_trades}L</span>
          </div>
        </div>

        <div style="display: flex; justify-content: space-between; padding: 10px 14px; background: var(--bg-card); border-radius: 10px; font-size: 12px;" class="mono">
          <span>Fee Deducted per Trade: <strong>${p.fee_rate_pct}%</strong></span>
          <span>Total Fees Paid: <strong style="color: var(--accent-amber);">$${(p.total_fees_paid_usdt || 0).toFixed(2)} USDT</strong></span>
        </div>

        <div>
          <span class="metric-label">Equity Growth Curve (Last ${p.equity_history.length} Snapshots)</span>
          <div class="chart-container" id="equity-chart-box">
            ${generateSvgChart(p.equity_history)}
          </div>
        </div>
      `;

      // Render Period History
      const tbody = document.getElementById('detail-trades-tbody');
      tbody.innerHTML = '';
      const trades = p.recent_trades || [];
      if (trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: var(--text-dim);">No trades recorded yet</td></tr>';
        return;
      }

      trades.forEach(tr => {
        const netPnl = tr.net_pnl_usdt !== undefined ? tr.net_pnl_usdt : (tr.net_pnl_usd || 0);
        const isWin = netPnl >= 0;
        const fee = tr.fee_usdt !== undefined ? `$${tr.fee_usdt.toFixed(2)} USDT` : `${tr.fee_btc} BTC`;
        const balAfter = tr.balance_after_usdt !== undefined ? `$${tr.balance_after_usdt.toFixed(2)} USDT` : `${tr.balance_after_btc} BTC`;
        const row = document.createElement('tr');
        row.innerHTML = `
          <td class="mono" style="font-size: 11px; color: var(--text-dim);">${tr.trade_id}</td>
          <td><span class="action-badge ${tr.direction === 'LONG' ? 'badge-long' : 'badge-short'}" style="display: inline-flex;">${tr.direction_icon}</span></td>
          <td class="mono">$${tr.entry_price ? tr.entry_price.toLocaleString() : '---'}</td>
          <td class="mono">$${tr.exit_price ? tr.exit_price.toLocaleString() : '---'}</td>
          <td class="mono ${isWin ? 'pnl-positive' : 'pnl-negative'}"><strong>${isWin ? '+' : ''}$${netPnl.toLocaleString('en-US', {minimumFractionDigits: 2})}</strong></td>
          <td class="mono ${isWin ? 'pnl-positive' : 'pnl-negative'}">${isWin ? '+' : ''}${tr.pnl_pct.toFixed(2)}%</td>
          <td class="mono" style="color: var(--text-dim);">${fee}</td>
          <td class="mono" style="color: var(--accent-green);">${balAfter}</td>
          <td><span class="confidence-pill" style="font-size: 10px;">${tr.confidence || 'High'}</span></td>
          <td><span style="font-size: 11px; font-weight: 700; color: ${tr.status === 'LIQUIDATED' ? 'var(--accent-red)' : 'var(--accent-green)'};">${tr.status}</span></td>
        `;
        tbody.appendChild(row);
      });
    }

    function generateSvgChart(history) {
      if (!history || history.length < 2) {
        return '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-dim); font-size: 12px;">Accumulating trade history...</div>';
      }
      const vals = history.map(h => (h.equity_usdt !== undefined ? h.equity_usdt : (h.equity_btc ? h.equity_btc * 63700 : 1000.0)));
      const minV = Math.min(...vals) * 0.98;
      const maxV = Math.max(...vals) * 1.02;
      const range = maxV - minV || 1;

      const w = 500;
      const h = 160;
      const pts = vals.map((v, i) => {
        const x = (i / (vals.length - 1)) * (w - 20) + 10;
        const y = h - ((v - minV) / range) * (h - 30) - 15;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');

      const isPositive = vals[vals.length - 1] >= vals[0];
      const strokeColor = isPositive ? '#00f090' : '#ff3b69';
      const gradientId = `grad-${Math.random().toString(36).substr(2, 9)}`;

      return `
        <svg viewBox="0 0 ${w} ${h}" style="width: 100%; height: 100%; overflow: visible;">
          <defs>
            <linearGradient id="${gradientId}" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="${strokeColor}" stop-opacity="0.3"/>
              <stop offset="100%" stop-color="${strokeColor}" stop-opacity="0.0"/>
            </linearGradient>
          </defs>
          <polyline fill="none" stroke="${strokeColor}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${pts}"/>
          <circle cx="${pts.split(' ').slice(-1)[0].split(',')[0]}" cy="${pts.split(' ').slice(-1)[0].split(',')[1]}" r="5" fill="${strokeColor}"/>
        </svg>
      `;
    }

    async function renderAlertsView() {
      if (!globalData) return;
      const tbody = document.getElementById('alerts-tbody');
      tbody.innerHTML = '';
      let alerts = globalData.alerts || [];
      try {
        const resp = await fetch('/api/alerts');
        if (resp.ok) {
          const data = await resp.json();
          if (Array.isArray(data.alerts)) alerts = data.alerts;
        }
      } catch (e) {
        console.error("Failed to fetch alert history, using status data:", e);
      }
      if (currentTab !== 'alerts') return;

      if (alerts.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; padding: 36px; color: var(--text-dim); font-size: 13px;">No alerts triggered yet.<br><span style="font-size: 11px;">Alerts trigger automatically when any period changes direction with significant hit probability or probability swings drastically.</span></td></tr>';
        return;
      }

      alerts.forEach(a => {
        if (!a) return;
        const isCrit = a.severity === 'CRITICAL';
        const row = document.createElement('tr');
        row.style.background = a.dismissed ? 'transparent' : 'rgba(255, 59, 105, 0.04)';
        const oldHitProb = Number(a.old_hit_prob) || 0;
        const newHitProb = Number(a.new_hit_prob) || 0;
        const probChangePct = Number(a.prob_change_pct) || 0;
        const priceAtAlert = Number(a.price_at_alert) || 0;
        const probDiffStr = `${probChangePct >= 0 ? '+' : ''}${probChangePct.toFixed(1)}%`;

        row.innerHTML = `
          <td class="mono" style="font-size: 11px; color: var(--text-dim); white-space: nowrap;">${escapeHtml(a.timestamp)}</td>
          <td><strong style="text-transform: uppercase;">${escapeHtml(a.period_name || a.period_id)}</strong></td>
          <td><span class="${isCrit ? 'badge-critical' : 'badge-warning'}">${escapeHtml(a.severity)}</span></td>
          <td style="font-weight: 600; font-size: 12px;">${escapeHtml((a.type || '').replace('_', ' '))}</td>
          <td>
            <span class="mono" style="font-weight: 700;">
              ${escapeHtml(a.old_direction)} (${oldHitProb}%) ➔ <span style="color: ${a.new_direction === 'LONG' ? 'var(--accent-green)' : 'var(--accent-red)'}; font-weight: 800;">${escapeHtml(a.new_direction)} (${newHitProb}%)</span>
            </span>
          </td>
          <td class="mono" style="font-weight: 700; color: ${probChangePct >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'};">${probDiffStr}</td>
          <td class="mono">$${priceAtAlert ? priceAtAlert.toLocaleString() : '---'}</td>
          <td style="font-size: 12px; max-width: 320px; color: var(--text-muted);">${escapeHtml(a.message)}</td>
          <td>
            ${!a.dismissed ? `
              <button onclick="dismissSingleAlert('${escapeHtml(a.alert_id)}')" class="alert-btn-dismiss" style="padding: 4px 10px; font-size: 11px;">Dismiss</button>
            ` : '<span style="font-size: 11px; color: var(--text-dim);">Dismissed</span>'}
          </td>
        `;
        tbody.appendChild(row);
      });
    }

    async function dismissSingleAlert(alertId) {
      try {
        await fetch(`/api/alerts/dismiss/${alertId}`, { method: 'POST' });
        fetchStatus();
      } catch (e) {
        console.error("Failed to dismiss alert:", e);
      }
    }

    async function dismissBannerAlert() {
      const banner = document.getElementById('top-alert-banner');
      const id = banner.getAttribute('data-top-alert-id');
      if (id) {
        await dismissSingleAlert(id);
      }
    }

    async function clearAllAlerts() {
      if (!confirm("Are you sure you want to clear all strategy alerts across all periods?")) return;
      try {
        await fetch('/api/alerts/clear', { method: 'POST' });
        fetchStatus();
      } catch (e) {
        console.error("Failed to clear alerts:", e);
      }
    }

    function switchTab(tabId) {
      currentTab = tabId;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

      if (tabId === 'overview') {
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
        document.getElementById('view-overview').style.display = 'block';
        document.getElementById('view-detail').style.display = 'none';
        document.getElementById('view-alerts').style.display = 'none';
        renderOverview();
      } else if (tabId === 'alerts') {
        const btn = document.getElementById('tab-btn-alerts');
        if (btn) btn.classList.add('active');
        document.getElementById('view-overview').style.display = 'none';
        document.getElementById('view-detail').style.display = 'none';
        document.getElementById('view-alerts').style.display = 'block';
        renderAlertsView();
      } else {
        const idx = ['3h', '6h', '12h', '1d', '3d'].indexOf(tabId);
        if (idx >= 0) {
          document.querySelectorAll('.tab-btn')[idx + 1].classList.add('active');
        }
        document.getElementById('view-overview').style.display = 'none';
        document.getElementById('view-detail').style.display = 'flex';
        document.getElementById('view-alerts').style.display = 'none';
        renderDetailView(tabId);
      }
    }

    // Auto-poll every 2.5 seconds
    fetchStatus();
    setInterval(fetchStatus, 2500);

    async function triggerReset() {
      if (!confirm("Are you sure you want to reset all 5 period engines to 1,000 USDT and start clean fresh trades from this exact second?")) return;
      try {
        const btn = document.getElementById('btn-reset');
        btn.textContent = "⏳ Resetting...";
        btn.disabled = true;
        const resp = await fetch('/api/reset', { method: 'POST' });
        const data = await resp.json();
        alert(`Reset complete!\n\nExact Start Time: ${data.exact_start_time_utc}\nLive BTC Price: $${data.live_btc_price.toLocaleString()}\n\nAll 5 strategies are now running fresh with 1,000 USDT each.`);
        btn.innerHTML = "<span>🔄</span> Reset to 1,000 USDT";
        btn.disabled = false;
        fetchStatus();
      } catch (e) {
        alert("Reset failed: " + e);
        const btn = document.getElementById('btn-reset');
        btn.innerHTML = "<span>🔄</span> Reset to 1,000 USDT";
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the modern dark-themed live terminal dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n" + "=" * 75)
    print(f"  ⚡ BITCOIN 4X LEVERAGED LIVE TRADING WEB SERVER STARTING")
    print(f"  🌐 Access the live web dashboard at: http://localhost:{port}")
    print(f"  📊 5 Parallel Period Engines: 3h, 6h, 12h, 1d, 3d")
    print("=" * 75 + "\n")
    uvicorn.run("web_server:app", host="0.0.0.0", port=port, log_level="info")
