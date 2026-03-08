"""Risk Manager + Portfolio Manager strategy agents.

RiskManagerStrategy:  Volatility guard — high vol / high exposure → COVER.
PortfolioManagerStrategy:  Position sizing via weighted vote of all signals.
"""

from __future__ import annotations

from src.strategies.base import Signal, StrategyAgent, TradingAction
from src.strategies.snapshot import FinancialSnapshot


class RiskManagerStrategy(StrategyAgent):
    """Volatility guard.

    Maps to GSC ``RiskManagerAgent`` (high DEX/Vega → COVER).
    In financial terms: if annualised volatility or volume spike exceeds
    safety thresholds, emit COVER with high confidence.
    """

    name = "RiskManager"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.COVER, 0.70)

        # Extreme volatility → force cover
        if s.volatility > 0.70:
            return Signal(TradingAction.COVER, 0.95)
        if s.volatility > 0.55:
            return Signal(TradingAction.COVER, 0.88)

        # Very high volume spike + negative movement = panic selling
        if s.volume_ratio > 3.0 and s.price_change_pct < -3:
            return Signal(TradingAction.COVER, 0.90)

        # High debt + high vol = fragile position
        if s.debt_to_equity > 3.0 and s.volatility > 0.40:
            return Signal(TradingAction.COVER, 0.82)

        # Moderate risk — no action needed
        if s.volatility > 0.40:
            return Signal(TradingAction.HOLD, 0.65)

        return Signal(TradingAction.HOLD, 0.50)


class PortfolioManagerStrategy(StrategyAgent):
    """Position sizing manager.

    This agent doesn't generate a directional signal — it validates that
    fundamental quality supports taking a position.  A weak fundamental
    profile produces HOLD (effectively a veto on sizing up).
    """

    name = "PortfolioManager"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        quality = 0.0
        # Profitability
        if s.roe > 0.10:
            quality += 1.0
        if s.operating_margin > 0.10:
            quality += 0.8
        # Liquidity
        if s.current_ratio > 1.2:
            quality += 0.5
        # Growth
        if s.revenue_growth > 0.05:
            quality += 0.7
        # Valuation sanity
        if 0 < s.pe_ratio < 35:
            quality += 0.5
        # Cash generation
        if s.fcf_yield > 0.03:
            quality += 0.5

        # High quality → supports position (pass-through)
        if quality >= 2.5:
            return Signal(TradingAction.BUY, min(0.80, 0.55 + quality * 0.05))
        # Low quality → veto (hold)
        return Signal(TradingAction.HOLD, 0.60)
