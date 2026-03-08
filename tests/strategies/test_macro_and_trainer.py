"""Tests for macro data provider, domain multipliers, and SPX trainer."""

import pytest

from src.strategies.macro import (
    DOMAIN_MULTIPLIERS,
    MACRO_BENEFIT_MAP,
    MacroData,
    macro_adjustment,
)
from src.strategies.snapshot import FinancialSnapshot
from src.strategies.trainer import SPXOptionsTrainer, TrainingResult


class TestMacroData:
    def test_default_macro_data(self):
        m = MacroData()
        assert m.vix == 18.0
        assert m.ten_year_yield == 4.0

    def test_custom_macro_data(self):
        m = MacroData(vix=35.0, ten_year_yield=5.5)
        assert m.vix == 35.0
        assert m.ten_year_yield == 5.5


class TestDomainMultipliers:
    def test_all_17_agents_have_multipliers(self):
        expected = [
            "Buffett", "Munger", "Burry", "CathieWood", "Dalio", "Soros",
            "Ackman", "Druckenmiller", "PeterLynch", "JimSimons", "CarlIcahn",
            "DavidTepper", "BillMiller", "JohnPaulson", "PaulTudorJones",
            "KenGriffin", "SteveCohen", "RiskManager", "PortfolioManager",
        ]
        for name in expected:
            assert name in DOMAIN_MULTIPLIERS, f"Missing multiplier for {name}"

    def test_burry_slope_weight_is_highest(self):
        assert DOMAIN_MULTIPLIERS["Burry"]["slope"] == 2.0

    def test_all_multipliers_positive(self):
        for agent, fields in DOMAIN_MULTIPLIERS.items():
            for field, mult in fields.items():
                assert mult > 0, f"{agent}.{field} has non-positive multiplier {mult}"


class TestMacroBenefitMap:
    def test_all_agents_have_benefits(self):
        assert len(MACRO_BENEFIT_MAP) == 19

    def test_dalio_has_most_macro_fields(self):
        # Dalio (risk parity) should benefit from most macro fields
        assert len(MACRO_BENEFIT_MAP["Dalio"]) >= 4


class TestMacroAdjustment:
    def test_calm_market_positive(self):
        adj = macro_adjustment("Burry", MacroData(vix=12.0))
        assert adj > 0

    def test_fearful_market_negative(self):
        adj = macro_adjustment("Burry", MacroData(vix=35.0))
        assert adj < 0

    def test_adjustment_clamped(self):
        # Even with extreme macro, adjustment should be bounded
        adj = macro_adjustment("Dalio", MacroData(vix=50.0, ten_year_yield=8.0, oil=150.0, dxy=120.0, gold=1500.0))
        assert -0.10 <= adj <= 0.10

    def test_unknown_agent_zero(self):
        adj = macro_adjustment("UnknownAgent", MacroData())
        assert adj == 0.0

    def test_low_rates_boost_cathie(self):
        adj = macro_adjustment("CathieWood", MacroData(ten_year_yield=2.0, gdp_growth=4.0))
        assert adj > 0


class TestSPXOptionsTrainer:
    def test_basic_training(self):
        snapshots = [
            FinancialSnapshot(
                ticker="SPX", regime_score=0.5, slope=1.0, volatility=0.2,
                roe=0.15, pe_ratio=18.0, revenue_growth=0.10,
            ),
            FinancialSnapshot(
                ticker="SPX", regime_score=-0.5, slope=-1.5, volatility=0.4,
                roe=0.05, pe_ratio=30.0, revenue_growth=-0.05,
            ),
            FinancialSnapshot(
                ticker="SPX", regime_score=0.3, slope=0.5, volatility=0.25,
                roe=0.12, pe_ratio=20.0, revenue_growth=0.08,
            ),
        ]
        forward_returns = [2.0, -3.0, 1.5]  # %

        trainer = SPXOptionsTrainer()
        result = trainer.train(snapshots, forward_returns)

        assert isinstance(result, TrainingResult)
        assert result.total_snapshots == 3
        assert len(result.agent_metrics) == 19  # 17 + 2 managers

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            SPXOptionsTrainer().train(
                [FinancialSnapshot()],
                [1.0, 2.0],
            )

    def test_single_snapshot_training(self):
        trainer = SPXOptionsTrainer()
        result = trainer.train(
            [FinancialSnapshot(ticker="SPX", regime_score=0.8, slope=2.0)],
            [5.0],
        )
        assert result.total_snapshots == 1
        for perf in result.agent_metrics.values():
            assert perf.total_signals == 1

    def test_accuracy_reasonable(self):
        # Bull snapshots with positive returns → agents should generally be correct
        snapshots = [
            FinancialSnapshot(
                ticker="SPX", regime_score=0.7, slope=2.0, volatility=0.2,
                roe=0.20, pe_ratio=15.0, revenue_growth=0.25, earnings_growth=0.30,
                sma_cross=1.05, rsi=60.0, peg_ratio=0.8, gross_margin=0.55,
                intrinsic_value_gap=0.20, fcf_yield=0.08,
            )
        ] * 10
        forward_returns = [3.0] * 10

        result = SPXOptionsTrainer().train(snapshots, forward_returns)
        # Most agents should have decent accuracy on clearly bullish data
        accuracies = [p.accuracy for p in result.agent_metrics.values()]
        avg_accuracy = sum(accuracies) / len(accuracies)
        assert avg_accuracy > 0.5
