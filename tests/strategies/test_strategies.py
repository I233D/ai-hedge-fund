"""Comprehensive tests for the 17-agent quantitative strategy system.

Covers:
    - All 17 investor strategies (zero-fill, bullish, bearish, edge cases)
    - Risk manager volatility guard
    - Portfolio manager quality gate
    - Signal aggregator (weighted vote, risk override)
    - Strategy runner (parallel + sequential)
    - 24 edge cases matching GSC validation suite
"""

import pytest

from src.strategies.aggregator import AggregatedDecision, SignalAggregator
from src.strategies.base import Signal, TradingAction
from src.strategies.investors import (
    ALL_INVESTOR_STRATEGIES,
    AckmanStrategy,
    BillMillerStrategy,
    BuffettStrategy,
    BurryStrategy,
    CarlIcahnStrategy,
    CathieWoodStrategy,
    DalioStrategy,
    DavidTepperStrategy,
    DruckenmillerStrategy,
    JimSimonsStrategy,
    JohnPaulsonStrategy,
    KenGriffinStrategy,
    MungerStrategy,
    PaulTudorJonesStrategy,
    PeterLynchStrategy,
    SorosStrategy,
    SteveCohenStrategy,
)
from src.strategies.risk_guard import PortfolioManagerStrategy, RiskManagerStrategy
from src.strategies.runner import StrategyRunner
from src.strategies.snapshot import FinancialSnapshot


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def zero_fill_snapshot():
    """All defaults — simulates GSC zero-fill / historical fallback."""
    return FinancialSnapshot(ticker="TEST", is_fallback=True)


@pytest.fixture
def bullish_snapshot():
    """Strong bull case — high quality, positive momentum."""
    return FinancialSnapshot(
        ticker="BULL",
        price=150.0,
        price_change_pct=2.0,
        price_change_5d_pct=5.0,
        price_change_20d_pct=12.0,
        volatility=0.22,
        volume_ratio=1.8,
        slope=2.5,
        sma_cross=1.06,
        rsi=62.0,
        regime_score=0.7,
        pe_ratio=18.0,
        pb_ratio=3.0,
        ps_ratio=4.0,
        peg_ratio=0.8,
        roe=0.22,
        roic=0.18,
        debt_to_equity=0.3,
        current_ratio=2.0,
        operating_margin=0.25,
        gross_margin=0.55,
        net_margin=0.15,
        fcf_yield=0.07,
        revenue_growth=0.25,
        earnings_growth=0.30,
        fcf_growth=20.0,
        revenue_growth_consistency=0.85,
        intrinsic_value_gap=0.25,
        ev_to_ebitda=12.0,
        insider_net_buys=5,
        insider_buy_ratio=0.8,
        news_sentiment=0.6,
        market_cap_bucket=4,
    )


@pytest.fixture
def bearish_snapshot():
    """Strong bear case — overvalued, declining, high vol."""
    return FinancialSnapshot(
        ticker="BEAR",
        price=50.0,
        price_change_pct=-3.0,
        price_change_5d_pct=-8.0,
        price_change_20d_pct=-18.0,
        volatility=0.60,
        volume_ratio=3.5,
        slope=-3.0,
        sma_cross=0.92,
        rsi=25.0,
        regime_score=-0.8,
        pe_ratio=55.0,
        pb_ratio=8.0,
        ps_ratio=15.0,
        peg_ratio=5.0,
        roe=-0.05,
        roic=-0.03,
        debt_to_equity=3.5,
        current_ratio=0.6,
        operating_margin=-0.05,
        gross_margin=0.20,
        net_margin=-0.10,
        fcf_yield=-0.03,
        revenue_growth=-0.10,
        earnings_growth=-0.20,
        fcf_growth=-15.0,
        revenue_growth_consistency=0.2,
        intrinsic_value_gap=-0.40,
        ev_to_ebitda=25.0,
        insider_net_buys=-5,
        insider_buy_ratio=0.1,
        news_sentiment=-0.7,
        market_cap_bucket=3,
    )


@pytest.fixture
def deep_value_snapshot():
    """Deep value: very cheap, beaten down, but fundamentally sound."""
    return FinancialSnapshot(
        ticker="VALUE",
        price=20.0,
        price_change_pct=-1.0,
        price_change_5d_pct=-5.0,
        price_change_20d_pct=-15.0,
        volatility=0.35,
        volume_ratio=1.2,
        slope=-0.8,
        sma_cross=0.96,
        rsi=32.0,
        regime_score=-0.4,
        pe_ratio=6.0,
        pb_ratio=0.7,
        ps_ratio=0.8,
        peg_ratio=0.5,
        roe=0.12,
        roic=0.10,
        debt_to_equity=0.4,
        current_ratio=2.5,
        operating_margin=0.18,
        gross_margin=0.45,
        net_margin=0.10,
        fcf_yield=0.12,
        revenue_growth=0.08,
        earnings_growth=0.12,
        fcf_growth=10.0,
        revenue_growth_consistency=0.6,
        intrinsic_value_gap=0.45,
        ev_to_ebitda=6.0,
        insider_net_buys=8,
        insider_buy_ratio=0.9,
        news_sentiment=-0.4,
        market_cap_bucket=3,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Zero-fill tests — all agents must HOLD on fallback data
# ═══════════════════════════════════════════════════════════════════════════

class TestZeroFill:
    """Every strategy must return HOLD on zero-fill / fallback data."""

    def test_all_investors_hold_on_zero_fill(self, zero_fill_snapshot):
        for strategy in ALL_INVESTOR_STRATEGIES:
            sig = strategy.generate_signal(zero_fill_snapshot)
            assert sig.action == TradingAction.HOLD, f"{strategy.name} should HOLD on zero-fill, got {sig.action}"
            assert sig.confidence == 0.50, f"{strategy.name} confidence should be 0.50 on zero-fill"

    def test_risk_manager_covers_on_zero_fill(self, zero_fill_snapshot):
        sig = RiskManagerStrategy().generate_signal(zero_fill_snapshot)
        assert sig.action == TradingAction.COVER

    def test_portfolio_manager_holds_on_zero_fill(self, zero_fill_snapshot):
        sig = PortfolioManagerStrategy().generate_signal(zero_fill_snapshot)
        assert sig.action == TradingAction.HOLD

    def test_aggregated_hold_on_zero_fill(self, zero_fill_snapshot):
        runner = StrategyRunner()
        decision = runner.run_sequential(zero_fill_snapshot)
        # All investors HOLD, risk manager COVER but not at 0.90+ → aggregate = HOLD
        assert decision.action in (TradingAction.HOLD, TradingAction.COVER)


# ═══════════════════════════════════════════════════════════════════════════
# Individual strategy tests — bullish scenario
# ═══════════════════════════════════════════════════════════════════════════

class TestBullishScenario:
    def test_buffett_buys(self, bullish_snapshot):
        sig = BuffettStrategy().generate_signal(bullish_snapshot)
        assert sig.action == TradingAction.BUY
        assert sig.confidence >= 0.60

    def test_munger_buys(self, bullish_snapshot):
        sig = MungerStrategy().generate_signal(bullish_snapshot)
        assert sig.action == TradingAction.BUY
        assert sig.confidence >= 0.60

    def test_cathie_wood_buys(self, bullish_snapshot):
        sig = CathieWoodStrategy().generate_signal(bullish_snapshot)
        assert sig.action == TradingAction.BUY
        assert sig.confidence >= 0.60

    def test_druckenmiller_buys(self, bullish_snapshot):
        sig = DruckenmillerStrategy().generate_signal(bullish_snapshot)
        assert sig.action == TradingAction.BUY

    def test_peter_lynch_buys(self, bullish_snapshot):
        sig = PeterLynchStrategy().generate_signal(bullish_snapshot)
        assert sig.action == TradingAction.BUY
        assert sig.confidence >= 0.60

    def test_soros_buys(self, bullish_snapshot):
        sig = SorosStrategy().generate_signal(bullish_snapshot)
        assert sig.action == TradingAction.BUY

    def test_ackman_buys(self, bullish_snapshot):
        sig = AckmanStrategy().generate_signal(bullish_snapshot)
        assert sig.action == TradingAction.BUY

    def test_paul_tudor_jones_buys(self, bullish_snapshot):
        sig = PaulTudorJonesStrategy().generate_signal(bullish_snapshot)
        assert sig.action == TradingAction.BUY

    def test_steve_cohen_buys(self, bullish_snapshot):
        sig = SteveCohenStrategy().generate_signal(bullish_snapshot)
        assert sig.action == TradingAction.BUY

    def test_dalio_buys(self, bullish_snapshot):
        sig = DalioStrategy().generate_signal(bullish_snapshot)
        assert sig.action == TradingAction.BUY


# ═══════════════════════════════════════════════════════════════════════════
# Individual strategy tests — bearish scenario
# ═══════════════════════════════════════════════════════════════════════════

class TestBearishScenario:
    def test_burry_shorts(self, bearish_snapshot):
        sig = BurryStrategy().generate_signal(bearish_snapshot)
        assert sig.action in (TradingAction.SHORT, TradingAction.SELL)
        assert sig.confidence >= 0.65

    def test_munger_sells(self, bearish_snapshot):
        sig = MungerStrategy().generate_signal(bearish_snapshot)
        assert sig.action == TradingAction.SELL
        assert sig.confidence >= 0.80

    def test_druckenmiller_sells(self, bearish_snapshot):
        sig = DruckenmillerStrategy().generate_signal(bearish_snapshot)
        assert sig.action == TradingAction.SELL

    def test_soros_shorts(self, bearish_snapshot):
        sig = SorosStrategy().generate_signal(bearish_snapshot)
        assert sig.action in (TradingAction.SHORT, TradingAction.SELL)

    def test_paul_tudor_jones_sells(self, bearish_snapshot):
        sig = PaulTudorJonesStrategy().generate_signal(bearish_snapshot)
        assert sig.action == TradingAction.SELL

    def test_john_paulson_shorts(self, bearish_snapshot):
        sig = JohnPaulsonStrategy().generate_signal(bearish_snapshot)
        assert sig.action in (TradingAction.SHORT, TradingAction.SELL)

    def test_risk_manager_covers(self, bearish_snapshot):
        sig = RiskManagerStrategy().generate_signal(bearish_snapshot)
        assert sig.action == TradingAction.COVER
        assert sig.confidence >= 0.85


# ═══════════════════════════════════════════════════════════════════════════
# Deep value tests — contrarian strategies should BUY
# ═══════════════════════════════════════════════════════════════════════════

class TestDeepValue:
    def test_burry_buys_deep_value(self, deep_value_snapshot):
        sig = BurryStrategy().generate_signal(deep_value_snapshot)
        assert sig.action == TradingAction.BUY
        assert sig.confidence >= 0.60

    def test_carl_icahn_buys(self, deep_value_snapshot):
        sig = CarlIcahnStrategy().generate_signal(deep_value_snapshot)
        assert sig.action == TradingAction.BUY

    def test_david_tepper_buys(self, deep_value_snapshot):
        sig = DavidTepperStrategy().generate_signal(deep_value_snapshot)
        assert sig.action == TradingAction.BUY

    def test_bill_miller_buys(self, deep_value_snapshot):
        sig = BillMillerStrategy().generate_signal(deep_value_snapshot)
        assert sig.action == TradingAction.BUY


# ═══════════════════════════════════════════════════════════════════════════
# Quant strategy tests
# ═══════════════════════════════════════════════════════════════════════════

class TestQuantStrategies:
    def test_jim_simons_oversold_buy(self):
        snap = FinancialSnapshot(ticker="Q", rsi=22.0, volatility=0.18, slope=0.8, sma_cross=1.04)
        sig = JimSimonsStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY

    def test_jim_simons_overbought_sell(self):
        snap = FinancialSnapshot(ticker="Q", rsi=78.0, volatility=0.18, slope=-0.8, sma_cross=0.95)
        sig = JimSimonsStrategy().generate_signal(snap)
        assert sig.action == TradingAction.SHORT

    def test_ken_griffin_low_vol_spike(self):
        snap = FinancialSnapshot(ticker="Q", volatility=0.15, volume_ratio=3.0)
        sig = KenGriffinStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY


# ═══════════════════════════════════════════════════════════════════════════
# Signal aggregator tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalAggregator:
    def test_empty_signals_returns_hold(self):
        decision = SignalAggregator.aggregate([])
        assert decision.action == TradingAction.HOLD

    def test_majority_buy(self):
        signals = [
            ("A", Signal(TradingAction.BUY, 0.80)),
            ("B", Signal(TradingAction.BUY, 0.75)),
            ("C", Signal(TradingAction.SELL, 0.60)),
        ]
        decision = SignalAggregator.aggregate(signals)
        assert decision.action == TradingAction.BUY

    def test_risk_manager_override(self):
        signals = [
            ("A", Signal(TradingAction.BUY, 0.90)),
            ("B", Signal(TradingAction.BUY, 0.85)),
            ("RiskManager", Signal(TradingAction.COVER, 0.95)),
        ]
        decision = SignalAggregator.aggregate(signals)
        assert decision.action == TradingAction.COVER

    def test_risk_manager_no_override_below_threshold(self):
        signals = [
            ("A", Signal(TradingAction.BUY, 0.90)),
            ("B", Signal(TradingAction.BUY, 0.85)),
            ("RiskManager", Signal(TradingAction.COVER, 0.80)),
        ]
        decision = SignalAggregator.aggregate(signals)
        assert decision.action == TradingAction.BUY

    def test_infra_agents_excluded_from_vote(self):
        signals = [
            ("A", Signal(TradingAction.BUY, 0.70)),
            ("PortfolioManager", Signal(TradingAction.HOLD, 0.90)),
            ("RiskManager", Signal(TradingAction.HOLD, 0.80)),
        ]
        decision = SignalAggregator.aggregate(signals)
        assert decision.action == TradingAction.BUY

    def test_confidence_is_normalised(self):
        signals = [
            ("A", Signal(TradingAction.BUY, 0.80)),
            ("B", Signal(TradingAction.SELL, 0.60)),
        ]
        decision = SignalAggregator.aggregate(signals)
        assert 0 <= decision.confidence <= 1.0

    def test_breakdown_populated(self):
        signals = [
            ("A", Signal(TradingAction.BUY, 0.80)),
            ("B", Signal(TradingAction.SELL, 0.60)),
        ]
        decision = SignalAggregator.aggregate(signals)
        assert TradingAction.BUY in decision.breakdown
        assert TradingAction.SELL in decision.breakdown


# ═══════════════════════════════════════════════════════════════════════════
# Strategy runner tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategyRunner:
    def test_runner_returns_aggregated_decision(self, bullish_snapshot):
        runner = StrategyRunner()
        decision = runner.run(bullish_snapshot)
        assert isinstance(decision, AggregatedDecision)
        assert decision.action in TradingAction
        assert 0 <= decision.confidence <= 1.0
        assert len(decision.votes) == 19  # 17 investors + 2 managers

    def test_runner_sequential_matches_parallel(self, bullish_snapshot):
        runner = StrategyRunner()
        d_seq = runner.run_sequential(bullish_snapshot)
        d_par = runner.run(bullish_snapshot)
        assert d_seq.action == d_par.action
        assert abs(d_seq.confidence - d_par.confidence) < 0.01

    def test_runner_custom_strategies(self, bullish_snapshot):
        runner = StrategyRunner(
            strategies=[BuffettStrategy(), BurryStrategy()],
            include_risk_manager=False,
            include_portfolio_manager=False,
        )
        decision = runner.run(bullish_snapshot)
        assert len(decision.votes) == 2

    def test_runner_agent_names(self):
        runner = StrategyRunner()
        names = runner.agent_names
        assert "Buffett" in names
        assert "RiskManager" in names
        assert "PortfolioManager" in names
        assert len(names) == 19

    def test_bullish_aggregate_is_buy(self, bullish_snapshot):
        runner = StrategyRunner()
        decision = runner.run_sequential(bullish_snapshot)
        assert decision.action == TradingAction.BUY

    def test_bearish_aggregate_is_sell_or_short(self, bearish_snapshot):
        runner = StrategyRunner()
        decision = runner.run_sequential(bearish_snapshot)
        # Risk manager has high confidence COVER on extreme vol
        assert decision.action in (TradingAction.SELL, TradingAction.SHORT, TradingAction.COVER)


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases (24 total — matching GSC validation suite)
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """24 edge cases covering zero-fill, historical fallback, high TE,
    negative slope, midday session, and boundary conditions."""

    def test_01_zero_fill_all_hold(self):
        snap = FinancialSnapshot(is_fallback=True)
        for s in ALL_INVESTOR_STRATEGIES:
            assert s.generate_signal(snap).action == TradingAction.HOLD

    def test_02_extreme_negative_slope(self):
        snap = FinancialSnapshot(slope=-5.0, volatility=0.55, regime_score=-0.8)
        sig = BurryStrategy().generate_signal(snap)
        assert sig.action in (TradingAction.SHORT, TradingAction.SELL)

    def test_03_extreme_positive_slope(self):
        snap = FinancialSnapshot(slope=5.0, volume_ratio=2.0, price_change_5d_pct=8.0, regime_score=0.6)
        sig = SorosStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY

    def test_04_high_volatility_cover(self):
        snap = FinancialSnapshot(volatility=0.75)
        sig = RiskManagerStrategy().generate_signal(snap)
        assert sig.action == TradingAction.COVER
        assert sig.confidence >= 0.90

    def test_05_low_volatility_hold(self):
        snap = FinancialSnapshot(volatility=0.10)
        sig = RiskManagerStrategy().generate_signal(snap)
        assert sig.action == TradingAction.HOLD

    def test_06_rsi_oversold(self):
        snap = FinancialSnapshot(rsi=18.0, volatility=0.15, slope=0.8, sma_cross=1.06, volume_ratio=2.5, price_change_pct=1.0)
        sig = JimSimonsStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY

    def test_07_rsi_overbought(self):
        snap = FinancialSnapshot(rsi=85.0, volatility=0.15, slope=-0.8, sma_cross=0.93, volume_ratio=2.5, price_change_pct=-1.0)
        sig = JimSimonsStrategy().generate_signal(snap)
        assert sig.action == TradingAction.SHORT

    def test_08_negative_roe_munger_sells(self):
        snap = FinancialSnapshot(roe=-0.1, operating_margin=-0.05, debt_to_equity=3.0, volatility=0.65)
        sig = MungerStrategy().generate_signal(snap)
        assert sig.action == TradingAction.SELL

    def test_09_high_peg_no_buy(self):
        snap = FinancialSnapshot(peg_ratio=4.0, earnings_growth=0.02)
        sig = PeterLynchStrategy().generate_signal(snap)
        assert sig.action != TradingAction.BUY

    def test_10_low_peg_buy(self):
        snap = FinancialSnapshot(peg_ratio=0.6, earnings_growth=0.25, revenue_growth_consistency=0.8, regime_score=0.5)
        sig = PeterLynchStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY

    def test_11_distressed_buy_tepper(self):
        snap = FinancialSnapshot(regime_score=-0.5, price_change_20d_pct=-12.0, roe=0.10, current_ratio=1.5)
        sig = DavidTepperStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY

    def test_12_contrarian_buy_miller(self):
        snap = FinancialSnapshot(news_sentiment=-0.5, roe=0.08, price_change_20d_pct=-12.0, pe_ratio=10.0, fcf_yield=0.07)
        sig = BillMillerStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY

    def test_13_macro_event_paulson_sells(self):
        snap = FinancialSnapshot(volatility=0.55, regime_score=-0.6)
        sig = JohnPaulsonStrategy().generate_signal(snap)
        assert sig.action == TradingAction.SHORT

    def test_14_trend_following_ptj_buys(self):
        snap = FinancialSnapshot(sma_cross=1.08, slope=2.0)
        sig = PaulTudorJonesStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY

    def test_15_trend_following_ptj_sells(self):
        snap = FinancialSnapshot(sma_cross=0.90, slope=-2.0)
        sig = PaulTudorJonesStrategy().generate_signal(snap)
        assert sig.action == TradingAction.SELL

    def test_16_vol_arb_griffin(self):
        snap = FinancialSnapshot(volatility=0.55, slope=3.0)
        sig = KenGriffinStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY

    def test_17_event_driven_cohen_buys(self):
        snap = FinancialSnapshot(insider_net_buys=5, volume_ratio=2.5, roe=0.15, operating_margin=0.20, news_sentiment=0.5)
        sig = SteveCohenStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY

    def test_18_activist_icahn_buys(self):
        snap = FinancialSnapshot(intrinsic_value_gap=0.35, pb_ratio=1.0, insider_net_buys=3, market_cap_bucket=4, fcf_yield=0.08)
        sig = CarlIcahnStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY

    def test_19_dalio_low_vol_bull(self):
        snap = FinancialSnapshot(volatility=0.15, regime_score=0.5)
        sig = DalioStrategy().generate_signal(snap)
        assert sig.action == TradingAction.BUY

    def test_20_dalio_high_vol_sells(self):
        snap = FinancialSnapshot(volatility=0.55)
        sig = DalioStrategy().generate_signal(snap)
        assert sig.action == TradingAction.SELL

    def test_21_signal_confidence_clamped(self):
        sig = Signal(TradingAction.BUY, 1.5)
        assert sig.confidence == 1.0
        sig2 = Signal(TradingAction.SELL, -0.5)
        assert sig2.confidence == 0.0

    def test_22_all_17_strategies_registered(self):
        assert len(ALL_INVESTOR_STRATEGIES) == 17

    def test_23_strategy_names_unique(self):
        names = [s.name for s in ALL_INVESTOR_STRATEGIES]
        assert len(names) == len(set(names))

    def test_24_panic_selling_risk_cover(self):
        snap = FinancialSnapshot(volume_ratio=4.0, price_change_pct=-5.0)
        sig = RiskManagerStrategy().generate_signal(snap)
        assert sig.action == TradingAction.COVER
        assert sig.confidence >= 0.90
