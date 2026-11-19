"""Tests for multi-provider live price fetching and 1,000 USDT 4x leveraged trading engine."""

import unittest
from datetime import datetime, timezone
from trading_engine import (
    fetch_live_btc_price_fast,
    PeriodStrategyTracker,
    MultiPeriodTradingManager,
    PERIOD_CONFIGS,
    _last_valid_live_btc_price
)


class TestTradingEngineUSDT(unittest.TestCase):
    def test_live_price_fetcher_sanity(self):
        """Verify that fetch_live_btc_price_fast returns a realistic BTC price without dropouts."""
        price = fetch_live_btc_price_fast()
        self.assertIsInstance(price, float)
        self.assertGreater(price, 10000.0, "BTC price should be reasonable (> $10k)")
        self.assertLess(price, 500000.0, "BTC price should be reasonable (< $500k)")

    def test_period_strategy_tracker_1000_usdt(self):
        """Verify 1,000 USDT capital, 0.1% fee deduction, and 4x leverage."""
        tracker = PeriodStrategyTracker("3h", PERIOD_CONFIGS["3h"])
        self.assertEqual(tracker.initial_usdt, 1000.0)
        self.assertEqual(tracker.balance_usdt, 1000.0)

        mock_time = datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc)
        live_price = 60000.0
        mock_model = {
            "action": "BUY / LONG",
            "confidence": "High Conviction",
            "prob_up": 70.0,
            "prob_down": 30.0,
            "pred_total_delta_dollars": 1500.0
        }

        tracker.open_new_trade(live_price, mock_time, model_result=mock_model)

        self.assertIsNotNone(tracker.active_trade)
        # 0.1% fee on 1,000 USDT = $1.00 USDT
        self.assertAlmostEqual(tracker.total_fees_paid_usdt, 1.0, places=2)
        # Balance becomes $999.00 USDT
        self.assertAlmostEqual(tracker.balance_usdt, 999.0, places=2)
        # Margin = $999.00 USDT, Position Notional = 4 * 999 = $3,996.00 USDT
        self.assertAlmostEqual(tracker.active_trade["margin_usdt"], 999.0, places=2)
        self.assertAlmostEqual(tracker.active_trade["position_notional_usdt"], 3996.0, places=2)

        # Slippage: Buy price is +0.01% higher
        expected_entry = live_price * 1.0001
        self.assertAlmostEqual(tracker.active_trade["entry_price"], expected_entry, places=2)

        # Liquidation price: ~0.77x of entry
        self.assertLess(tracker.active_trade["liquidation_price"], expected_entry)

    def test_3d_fee_rate_03_pct(self):
        """Verify 3-day mode specifically deducts 0.3% fee ($3.00 on $1,000)."""
        tracker = PeriodStrategyTracker("3d", PERIOD_CONFIGS["3d"])
        mock_time = datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc)
        live_price = 60000.0
        tracker.open_new_trade(live_price, mock_time)

        self.assertAlmostEqual(tracker.total_fees_paid_usdt, 3.0, places=2)
        self.assertAlmostEqual(tracker.balance_usdt, 997.0, places=2)
        self.assertAlmostEqual(tracker.active_trade["margin_usdt"], 997.0, places=2)

    def test_manager_max_metrics(self):
        """Verify max balance and max return calculations across the 5 periods."""
        mgr = MultiPeriodTradingManager()
        dashboard = mgr.get_full_dashboard_data()

        self.assertIn("max_balance_usdt", dashboard)
        self.assertIn("max_balance_period", dashboard)
        self.assertIn("max_return_pct", dashboard)
        self.assertIn("max_return_period", dashboard)
        self.assertIsInstance(dashboard["max_balance_usdt"], (int, float))
        self.assertIsInstance(dashboard["max_return_pct"], (int, float))

    def test_direction_reversal_alert(self):
        """Verify that evaluate_alert triggers a critical alert when direction flips with high hit probability."""
        tracker = PeriodStrategyTracker("3h", PERIOD_CONFIGS["3h"])
        t1 = datetime(2026, 11, 19, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 19, 11, 0, 0, tzinfo=timezone.utc)

        # Baseline signal
        a1 = tracker.evaluate_alert("LONG", 83.6, 63000.0, t1)
        self.assertIsNone(a1, "Initial baseline signal should not trigger alert")

        # Direction flip from LONG (83.6%) to SHORT (72.0%)
        a2 = tracker.evaluate_alert("SHORT", 72.0, 63400.0, t2)
        self.assertIsNotNone(a2, "Direction flip should trigger alert")
        self.assertEqual(a2["type"], "DIRECTION_REVERSAL")
        self.assertEqual(a2["severity"], "CRITICAL")
        self.assertEqual(a2["old_direction"], "LONG")
        self.assertEqual(a2["new_direction"], "SHORT")
        self.assertEqual(a2["old_hit_prob"], 83.6)
        self.assertEqual(a2["new_hit_prob"], 72.0)
        self.assertFalse(a2["dismissed"])
        self.assertIsNotNone(tracker.active_alert)

    def test_no_alert_when_hit_prob_too_low(self):
        """Verify that evaluate_alert does NOT trigger an alert if new hit prob is < 60% (e.g. 54.2% chop)."""
        tracker = PeriodStrategyTracker("6h", PERIOD_CONFIGS["6h"])
        t1 = datetime(2026, 11, 19, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 19, 11, 0, 0, tzinfo=timezone.utc)

        tracker.evaluate_alert("LONG", 82.2, 63000.0, t1)

        # Flipped to SHORT, but hit prob is only 54.2% (<60% threshold)
        alert = tracker.evaluate_alert("SHORT", 54.2, 63100.0, t2)
        self.assertIsNone(alert, "Direction flip with weak hit prob (<60%) must not trigger an alert")

    def test_probability_swing_alert(self):
        """Verify that evaluate_alert triggers a warning alert when probability swings drastically in same direction."""
        tracker = PeriodStrategyTracker("1d", PERIOD_CONFIGS["1d"])
        t1 = datetime(2026, 11, 19, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 19, 11, 0, 0, tzinfo=timezone.utc)

        # Baseline signal
        tracker.evaluate_alert("LONG", 50.0, 63000.0, t1)

        # Same direction, but hit prob jumps from 50.0% to 76.0% (+26%)
        alert = tracker.evaluate_alert("LONG", 76.0, 63100.0, t2)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["type"], "PROBABILITY_SHIFT")
        self.assertEqual(alert["severity"], "WARNING")
        self.assertAlmostEqual(alert["prob_change_pct"], 26.0, places=1)

    def test_manager_dismiss_and_clear_alerts(self):
        """Verify dismissing and clearing alerts in MultiPeriodTradingManager."""
        mgr = MultiPeriodTradingManager()
        test_alert = {
            "alert_id": "test-alert-123",
            "period_id": "3h",
            "type": "DIRECTION_REVERSAL",
            "dismissed": False
        }
        mgr.record_alert(test_alert)
        self.assertTrue(any(a["alert_id"] == "test-alert-123" for a in mgr.alerts))

        # Dismiss alert
        dismissed = mgr.dismiss_alert("test-alert-123")
        self.assertTrue(dismissed)
        matching = [a for a in mgr.alerts if a["alert_id"] == "test-alert-123"]
        self.assertTrue(matching[0]["dismissed"])

        # Clear alerts
        mgr.clear_alerts()
        self.assertEqual(len(mgr.alerts), 0)


if __name__ == "__main__":
    unittest.main()
