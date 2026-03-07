"""Comprehensive tests for IBKR ES/NQ/SPX 24h pre-market feed system.

Covers:
    - IBKRQuote properties
    - CircuitBreaker state transitions
    - Adaptive freshness bounds (RTH/pre-market/overnight, VIX-scaled)
    - Trading session detection
    - IBKRRealTimeFeed (cache, freshness markers, circuit protection)
    - TickerWeights for ES/NQ/SPX/equity
    - DataFeedOrchestrator (bundle assembly)
    - Benchmark computation (ES/NQ/SPY)
    - 24 edge cases (overnight, high VIX, stale cache, ticker switching)
"""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.feeds.benchmarks import (
    BenchmarkResult,
    compute_benchmarks,
    get_benchmark_ticker,
)
from src.feeds.circuit_breaker import CircuitBreaker, CircuitState
from src.feeds.freshness import (
    AdaptiveFreshnessCalculator,
    FreshnessBound,
    TradingSession,
    detect_session,
)
from src.feeds.ibkr_client import (
    FUTURES_CONIDS,
    SYMBOL_TO_IBKR,
    IBKRConfig,
    IBKRQuote,
)
from src.feeds.ibkr_feed import IBKRRealTimeFeed
from src.feeds.orchestrator import DataFeedOrchestrator, FeedBundle
from src.feeds.ticker_weights import TickerWeights, get_ticker_weights
from src.strategies.macro import MacroData
from src.strategies.snapshot import FinancialSnapshot


# ═══════════════════════════════════════════════════════════════════════════
# IBKRQuote
# ═══════════════════════════════════════════════════════════════════════════

class TestIBKRQuote:
    def test_mid_price(self):
        q = IBKRQuote(symbol="ES", last=5800.0, bid=5799.5, ask=5800.5, volume=100, timestamp=time.time())
        assert q.mid == 5800.0

    def test_spread(self):
        q = IBKRQuote(symbol="ES", last=5800.0, bid=5799.5, ask=5800.5, volume=100, timestamp=time.time())
        assert q.spread == 1.0

    def test_age_seconds(self):
        q = IBKRQuote(symbol="ES", last=5800.0, bid=0, ask=0, volume=0, timestamp=time.time() - 10)
        assert q.age_seconds >= 10.0

    def test_mid_fallback_to_last(self):
        q = IBKRQuote(symbol="ES", last=5800.0, bid=0, ask=0, volume=0, timestamp=time.time())
        assert q.mid == 5800.0


# ═══════════════════════════════════════════════════════════════════════════
# CircuitBreaker
# ═══════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available

    def test_trips_on_threshold(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.is_available

    def test_recovers_after_timeout(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_closes_circuit(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_manual_reset(self):
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Freshness
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptiveFreshness:
    def setup_method(self):
        self.calc = AdaptiveFreshnessCalculator()

    def test_es_rth_default_vix(self):
        bound = self.calc.compute("ES", vix=20.0, session=TradingSession.RTH)
        assert bound.bound_seconds == 17  # base=17, vix_factor=1.0, session=1.0

    def test_es_premarket_default_vix(self):
        bound = self.calc.compute("ES", vix=20.0, session=TradingSession.PRE_MARKET)
        assert bound.bound_seconds == 34  # 17 × 1.0 × 2.0

    def test_es_high_vix_tighter(self):
        bound = self.calc.compute("ES", vix=40.0, session=TradingSession.RTH)
        # vix_factor = 40/20 = 2.0 → 17 × 2.0 = 34
        assert bound.bound_seconds == 34

    def test_es_low_vix_wider(self):
        bound = self.calc.compute("ES", vix=10.0, session=TradingSession.RTH)
        # vix_factor = max(0.5, 10/20) = 0.5 → 17 × 0.5 = 8 → clamped to 15 (min)
        assert bound.bound_seconds == 15

    def test_spx_rth(self):
        bound = self.calc.compute("SPX", vix=20.0, session=TradingSession.RTH)
        assert bound.bound_seconds == 60  # base=60

    def test_nq_overnight_high_vix(self):
        bound = self.calc.compute("NQ", vix=35.0, session=TradingSession.OVERNIGHT)
        # base=17, vix_factor=35/20=1.75, session=3.0 → 17 × 1.75 × 3.0 ≈ 89
        assert bound.bound_seconds == 89

    def test_bound_clamped_min(self):
        bound = self.calc.compute("ES", vix=5.0, session=TradingSession.RTH)
        assert bound.bound_seconds >= 15

    def test_bound_clamped_max(self):
        bound = self.calc.compute("SPX", vix=40.0, session=TradingSession.OVERNIGHT)
        assert bound.bound_seconds <= 900

    def test_freshness_bound_fields(self):
        bound = self.calc.compute("ES", vix=25.0, session=TradingSession.RTH)
        assert bound.symbol == "ES"
        assert bound.base_bound == 17
        assert bound.session == TradingSession.RTH
        assert bound.vix == 25.0


# ═══════════════════════════════════════════════════════════════════════════
# Session Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionDetection:
    def test_rth(self):
        # 14:30 UTC = 9:30 ET
        dt = datetime(2024, 3, 15, 14, 30, tzinfo=timezone.utc)
        assert detect_session(dt) == TradingSession.RTH

    def test_premarket(self):
        # 11:00 UTC = 6:00 ET
        dt = datetime(2024, 3, 15, 11, 0, tzinfo=timezone.utc)
        assert detect_session(dt) == TradingSession.PRE_MARKET

    def test_after_hours(self):
        # 22:00 UTC = 17:00 ET
        dt = datetime(2024, 3, 15, 22, 0, tzinfo=timezone.utc)
        assert detect_session(dt) == TradingSession.AFTER_HOURS

    def test_overnight(self):
        # 3:00 UTC = 22:00 ET (prev day)
        dt = datetime(2024, 3, 15, 3, 0, tzinfo=timezone.utc)
        assert detect_session(dt) == TradingSession.OVERNIGHT


# ═══════════════════════════════════════════════════════════════════════════
# IBKRRealTimeFeed
# ═══════════════════════════════════════════════════════════════════════════

class TestIBKRRealTimeFeed:
    def _make_feed_with_mock(self):
        """Create feed with mocked IBKR client."""
        mock_client = MagicMock()
        mock_client.get_quote.return_value = IBKRQuote(
            symbol="ES", last=5820.5, bid=5820.0, ask=5821.0,
            volume=150000, timestamp=time.time(),
        )
        feed = IBKRRealTimeFeed(client=mock_client)
        return feed, mock_client

    def test_fresh_quote(self):
        feed, mock = self._make_feed_with_mock()
        quote = feed.get_quote("ES", vix=18.0)
        assert quote is not None
        assert quote.last == 5820.5
        assert quote.symbol == "ES"

    def test_cache_hit(self):
        feed, mock = self._make_feed_with_mock()
        feed.get_quote("ES", vix=18.0)  # first call → live
        feed.get_quote("ES", vix=18.0)  # second call → cache
        assert mock.get_quote.call_count == 1

    def test_force_refresh_bypasses_cache(self):
        feed, mock = self._make_feed_with_mock()
        feed.get_quote("ES", vix=18.0)
        feed.get_quote("ES", vix=18.0, force_refresh=True)
        assert mock.get_quote.call_count == 2

    def test_circuit_breaker_on_failures(self):
        mock_client = MagicMock()
        mock_client.get_quote.return_value = None
        feed = IBKRRealTimeFeed(client=mock_client)
        feed._circuit_breaker.failure_threshold = 3

        for _ in range(5):
            feed.get_quote("ES", vix=18.0)

        assert feed.circuit_state == "open"

    def test_markers_recorded(self):
        feed, mock = self._make_feed_with_mock()
        feed.get_quote("ES", vix=18.0)
        assert len(feed.markers) == 1
        assert feed.markers[0].source == "live"
        assert feed.markers[0].is_fresh

    def test_invalidate_cache(self):
        feed, mock = self._make_feed_with_mock()
        feed.get_quote("ES", vix=18.0)
        feed.invalidate_cache("ES")
        feed.get_quote("ES", vix=18.0)
        assert mock.get_quote.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# TickerWeights
# ═══════════════════════════════════════════════════════════════════════════

class TestTickerWeights:
    def test_es_is_futures(self):
        tw = get_ticker_weights("ES")
        assert tw.instrument_type == "futures_es"

    def test_nq_is_futures(self):
        tw = get_ticker_weights("NQ")
        assert tw.instrument_type == "futures_nq"

    def test_spx_is_index(self):
        tw = get_ticker_weights("SPX")
        assert tw.instrument_type == "index_spx"

    def test_aapl_is_equity(self):
        tw = get_ticker_weights("AAPL")
        assert tw.instrument_type == "equity"

    def test_es_druckenmiller_boosted(self):
        tw = get_ticker_weights("ES")
        assert tw.weights["Druckenmiller"] == 1.5

    def test_nq_cathie_wood_boosted(self):
        tw = get_ticker_weights("NQ")
        assert tw.weights["CathieWood"] == 1.5

    def test_equity_all_weights_equal(self):
        tw = get_ticker_weights("AAPL")
        for w in tw.weights.values():
            assert w == 1.0

    def test_all_19_agents_have_weights(self):
        tw = get_ticker_weights("ES")
        assert len(tw.weights) == 19

    def test_case_insensitive(self):
        tw1 = get_ticker_weights("es")
        tw2 = get_ticker_weights("ES")
        assert tw1.instrument_type == tw2.instrument_type


# ═══════════════════════════════════════════════════════════════════════════
# DataFeedOrchestrator
# ═══════════════════════════════════════════════════════════════════════════

class TestDataFeedOrchestrator:
    def _make_orchestrator(self):
        mock_client = MagicMock()
        mock_client.get_quote.return_value = IBKRQuote(
            symbol="ES", last=5820.5, bid=5820.0, ask=5821.0,
            volume=150000, timestamp=time.time(),
        )
        feed = IBKRRealTimeFeed(client=mock_client)
        return DataFeedOrchestrator(ibkr_feed=feed)

    def test_es_bundle(self):
        orch = self._make_orchestrator()
        bundle = orch.get_feeds("ES")
        assert isinstance(bundle, FeedBundle)
        assert bundle.ticker == "ES"
        assert bundle.is_futures
        assert bundle.ibkr_quote is not None
        assert bundle.ibkr_quote.last == 5820.5

    def test_equity_no_ibkr(self):
        orch = self._make_orchestrator()
        bundle = orch.get_feeds("AAPL", snapshot=FinancialSnapshot(ticker="AAPL", price=180.0))
        assert not bundle.is_futures
        assert bundle.ibkr_quote is None

    def test_snapshot_enriched_with_ibkr(self):
        orch = self._make_orchestrator()
        snap = FinancialSnapshot(ticker="ES", price=0.0)
        bundle = orch.get_feeds("ES", snapshot=snap)
        assert bundle.snapshot.price == 5820.5

    def test_batch_feeds(self):
        orch = self._make_orchestrator()
        bundles = orch.get_feeds_batch(["ES", "NQ", "AAPL"])
        assert len(bundles) == 3
        assert "ES" in bundles
        assert "NQ" in bundles
        assert "AAPL" in bundles


# ═══════════════════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmarks:
    def test_benchmark_ticker_es(self):
        assert get_benchmark_ticker("ES") == "SPY"

    def test_benchmark_ticker_nq(self):
        assert get_benchmark_ticker("NQ") == "QQQ"

    def test_benchmark_ticker_aapl(self):
        assert get_benchmark_ticker("AAPL") == "SPY"

    def test_compute_benchmarks(self):
        strat_returns = [0.01, -0.005, 0.008, 0.003, -0.002, 0.012, -0.001, 0.007, 0.002, -0.004]
        spy_returns = [0.005, -0.003, 0.004, 0.002, -0.001, 0.006, -0.002, 0.003, 0.001, -0.002]

        results = compute_benchmarks(strat_returns, {"SPY": spy_returns})
        assert "strategy" in results
        assert "SPY" in results
        assert results["strategy"].total_return_pct > 0
        assert isinstance(results["strategy"].sharpe_ratio, float)

    def test_empty_returns(self):
        results = compute_benchmarks([])
        assert results["strategy"].total_return_pct == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases (24 — matching GSC validation suite)
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_01_overnight_freshness_relaxed(self):
        calc = AdaptiveFreshnessCalculator()
        bound = calc.compute("ES", vix=20.0, session=TradingSession.OVERNIGHT)
        assert bound.bound_seconds > 17 * 2  # overnight should be >2× RTH

    def test_02_high_vix_35(self):
        calc = AdaptiveFreshnessCalculator()
        bound = calc.compute("ES", vix=35.0, session=TradingSession.RTH)
        assert bound.bound_seconds > 17  # tighter than default but still > base

    def test_03_stale_cache_returns_fallback(self):
        mock_client = MagicMock()
        mock_client.get_quote.side_effect = [
            IBKRQuote("ES", 5800.0, 5799.5, 5800.5, 100, time.time() - 120),
            None,  # second call fails
        ]
        feed = IBKRRealTimeFeed(client=mock_client)
        feed.get_quote("ES", vix=18.0)
        feed.invalidate_cache("ES")
        # Re-cache manually with old timestamp
        feed._cache["ES"] = IBKRQuote("ES", 5800.0, 5799.5, 5800.5, 100, time.time() - 120)
        q = feed.get_quote("ES", vix=18.0)
        assert q is not None  # returns stale cache

    def test_04_ticker_switch_es_to_nq(self):
        tw_es = get_ticker_weights("ES")
        tw_nq = get_ticker_weights("NQ")
        assert tw_es.instrument_type != tw_nq.instrument_type
        assert tw_es.weights["Druckenmiller"] >= tw_nq.weights["CathieWood"]

    def test_05_spx_maps_to_es(self):
        assert SYMBOL_TO_IBKR["SPX"] == "ES"

    def test_06_conids_exist(self):
        assert "ES" in FUTURES_CONIDS
        assert "NQ" in FUTURES_CONIDS

    def test_07_circuit_breaker_trip_count(self):
        cb = CircuitBreaker(name="test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb._total_trips == 1

    def test_08_freshness_vix_factor_clamped(self):
        calc = AdaptiveFreshnessCalculator()
        bound = calc.compute("ES", vix=1.0, session=TradingSession.RTH)
        assert bound.vix_factor == 0.5  # min clamp

    def test_09_freshness_vix_factor_max(self):
        calc = AdaptiveFreshnessCalculator()
        bound = calc.compute("ES", vix=50.0, session=TradingSession.RTH)
        assert bound.vix_factor == 2.0  # max clamp

    def test_10_nq_premarket(self):
        calc = AdaptiveFreshnessCalculator()
        bound = calc.compute("NQ", vix=18.5, session=TradingSession.PRE_MARKET)
        # base=17, vix=18.5/20=0.925, session=2.0 → 17 × 0.925 × 2 ≈ 31
        assert 30 <= bound.bound_seconds <= 35

    def test_11_feed_bundle_is_premarket(self):
        bundle = FeedBundle(
            ticker="ES", snapshot=FinancialSnapshot(), ibkr_quote=None,
            macro=MacroData(), ticker_weights=get_ticker_weights("ES"),
            freshness=None, session=TradingSession.PRE_MARKET, timestamp=time.time(),
        )
        assert bundle.is_premarket

    def test_12_feed_bundle_not_premarket_rth(self):
        bundle = FeedBundle(
            ticker="ES", snapshot=FinancialSnapshot(), ibkr_quote=None,
            macro=MacroData(), ticker_weights=get_ticker_weights("ES"),
            freshness=None, session=TradingSession.RTH, timestamp=time.time(),
        )
        assert not bundle.is_premarket

    def test_13_has_live_feed_true(self):
        q = IBKRQuote("ES", 5800.0, 0, 0, 0, time.time())
        bundle = FeedBundle(
            ticker="ES", snapshot=FinancialSnapshot(), ibkr_quote=q,
            macro=MacroData(), ticker_weights=get_ticker_weights("ES"),
            freshness=None, session=TradingSession.RTH, timestamp=time.time(),
        )
        assert bundle.has_live_feed

    def test_14_has_live_feed_false_old_quote(self):
        q = IBKRQuote("ES", 5800.0, 0, 0, 0, time.time() - 120)
        bundle = FeedBundle(
            ticker="ES", snapshot=FinancialSnapshot(), ibkr_quote=q,
            macro=MacroData(), ticker_weights=get_ticker_weights("ES"),
            freshness=None, session=TradingSession.RTH, timestamp=time.time(),
        )
        assert not bundle.has_live_feed

    def test_15_benchmark_sharpe_positive_for_gains(self):
        returns = [0.01, 0.008, 0.012, 0.009, 0.011] * 10  # positive with variance
        results = compute_benchmarks(returns)
        assert results["strategy"].sharpe_ratio > 0

    def test_16_benchmark_max_drawdown_negative(self):
        returns = [0.01, -0.05, 0.01, -0.03, 0.02]
        results = compute_benchmarks(returns)
        assert results["strategy"].max_drawdown_pct < 0

    def test_17_mes_maps_to_es(self):
        tw = get_ticker_weights("MES")
        assert tw.instrument_type == "futures_es"

    def test_18_mnq_maps_to_nq(self):
        tw = get_ticker_weights("MNQ")
        assert tw.instrument_type == "futures_nq"

    def test_19_unknown_symbol_defaults_equity(self):
        tw = get_ticker_weights("ZZZZ")
        assert tw.instrument_type == "equity"

    def test_20_ibkr_config_defaults(self):
        cfg = IBKRConfig()
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 5.0

    def test_21_ibkr_quote_not_delayed(self):
        q = IBKRQuote("ES", 5800.0, 0, 0, 0, time.time(), is_delayed=False)
        assert not q.is_delayed

    def test_22_freshness_bound_min_15(self):
        calc = AdaptiveFreshnessCalculator()
        bound = calc.compute("ES", vix=0.1, session=TradingSession.RTH)
        assert bound.bound_seconds >= 15

    def test_23_freshness_bound_max_900(self):
        calc = AdaptiveFreshnessCalculator()
        bound = calc.compute("SPX", vix=50.0, session=TradingSession.OVERNIGHT)
        assert bound.bound_seconds <= 900

    def test_24_all_session_types_covered(self):
        assert len(TradingSession) == 4
