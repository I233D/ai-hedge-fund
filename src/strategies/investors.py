"""17 Concrete Investor Strategies – Production Implementation.

Each strategy is pure math on ``FinancialSnapshot`` — no LLM calls, no I/O.
Total sweep of all 17 completes in < 1 ms.

Strategy mapping to GSC concepts:
    regime_score   → market regime (bull/bear composite)
    slope          → price trend slope (20-day linear regression)
    volatility     → annualised daily vol  (≈ TE)
    revenue_growth → growth momentum       (≈ gm/gn)
    volume_ratio   → volume spike          (≈ dex_px)
    price_change_* → signed bar area proxy (≈ signedBarAreaPx)
"""

from src.strategies.base import Signal, StrategyAgent, TradingAction
from src.strategies.snapshot import FinancialSnapshot


# ────────────────────────────────────────────────────────────────────────────
# 1. Warren Buffett – Value Investing (moat + intrinsic value + quality)
# ────────────────────────────────────────────────────────────────────────────
class BuffettStrategy(StrategyAgent):
    name = "Buffett"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        score = 0.0
        # High ROE indicates moat
        if s.roe > 0.15:
            score += 1.5
        elif s.roe > 0.10:
            score += 0.8
        # Low debt
        if 0 < s.debt_to_equity < 0.5:
            score += 1.0
        elif s.debt_to_equity < 1.0:
            score += 0.5
        # Consistent margins
        if s.operating_margin > 0.20:
            score += 1.0
        # Trading below intrinsic value
        if s.intrinsic_value_gap > 0.20:
            score += 1.5
        elif s.intrinsic_value_gap > 0.05:
            score += 0.8
        # Positive regime
        if s.regime_score > 0.3:
            score += 0.5
        # Positive price trend
        if s.slope > 0:
            score += 0.3

        if score >= 3.5:
            return Signal(TradingAction.BUY, min(0.95, 0.60 + score * 0.06))
        elif score <= 1.0:
            return Signal(TradingAction.SELL, min(0.85, 0.55 + (5.0 - score) * 0.05))
        return Signal(TradingAction.HOLD, 0.55)


# ────────────────────────────────────────────────────────────────────────────
# 2. Charlie Munger – Inversion + Quality Businesses
# ────────────────────────────────────────────────────────────────────────────
class MungerStrategy(StrategyAgent):
    name = "Munger"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        score = 0.0
        # Inversion: avoid bad businesses (high debt, negative margins)
        red_flags = 0
        if s.debt_to_equity > 2.0:
            red_flags += 1
        if s.operating_margin < 0:
            red_flags += 1
        if s.roe < 0:
            red_flags += 1
        if s.volatility > 0.60:
            red_flags += 1

        if red_flags >= 3:
            return Signal(TradingAction.SELL, 0.85)

        # Quality: strong ROIC, predictable earnings, reasonable price
        if s.roic > 0.15:
            score += 1.5
        elif s.roic > 0.10:
            score += 0.8
        if s.gross_margin > 0.40:
            score += 1.0
        if s.revenue_growth_consistency > 0.7:
            score += 1.0
        if s.pe_ratio > 0 and s.pe_ratio < 25:
            score += 0.8
        if s.current_ratio > 1.5:
            score += 0.5

        if score >= 3.5:
            return Signal(TradingAction.BUY, min(0.90, 0.58 + score * 0.06))
        return Signal(TradingAction.HOLD, 0.55)


# ────────────────────────────────────────────────────────────────────────────
# 3. Michael Burry – Contrarian Short (extreme negative slope + high vol)
# ────────────────────────────────────────────────────────────────────────────
class BurryStrategy(StrategyAgent):
    name = "Burry"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        short_score = 0.0
        # Extreme negative price slope
        if s.slope < -2.0:
            short_score += 2.0
        elif s.slope < -0.5:
            short_score += 1.0
        # High volatility (convexity opportunity)
        if s.volatility > 0.50:
            short_score += 1.5
        elif s.volatility > 0.35:
            short_score += 0.8
        # Overvalued
        if s.pe_ratio > 40:
            short_score += 1.0
        elif s.pe_ratio > 30:
            short_score += 0.5
        # Negative FCF
        if s.fcf_yield < 0:
            short_score += 1.0
        # Bearish regime
        if s.regime_score < -0.3:
            short_score += 0.8

        # Deep value long (contrarian buy)
        long_score = 0.0
        if s.pe_ratio > 0 and s.pe_ratio < 8:
            long_score += 1.5
        if s.pb_ratio > 0 and s.pb_ratio < 1.0:
            long_score += 1.0
        if s.fcf_yield > 0.10:
            long_score += 1.5
        if s.insider_net_buys > 3:
            long_score += 1.0

        if short_score >= 4.0:
            return Signal(TradingAction.SHORT, min(0.95, 0.65 + short_score * 0.05))
        if long_score >= 3.5:
            return Signal(TradingAction.BUY, min(0.90, 0.60 + long_score * 0.06))
        return Signal(TradingAction.HOLD, 0.55)


# ────────────────────────────────────────────────────────────────────────────
# 4. Cathie Wood – Disruptive Growth (positive regime + high growth)
# ────────────────────────────────────────────────────────────────────────────
class CathieWoodStrategy(StrategyAgent):
    name = "CathieWood"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        score = 0.0
        # High revenue growth
        if s.revenue_growth > 0.30:
            score += 2.0
        elif s.revenue_growth > 0.15:
            score += 1.0
        # Positive regime
        if s.regime_score > 0.3:
            score += 1.0
        elif s.regime_score > 0:
            score += 0.5
        # Earnings growth momentum
        if s.earnings_growth > 0.25:
            score += 1.5
        elif s.earnings_growth > 0.10:
            score += 0.8
        # Price momentum
        if s.price_change_20d_pct > 5:
            score += 0.8
        # Innovation proxy: high gross margin
        if s.gross_margin > 0.50:
            score += 0.7

        if score >= 3.5:
            return Signal(TradingAction.BUY, min(0.95, 0.60 + score * 0.06))
        elif score <= 1.0:
            return Signal(TradingAction.SELL, 0.65)
        return Signal(TradingAction.HOLD, 0.55)


# ────────────────────────────────────────────────────────────────────────────
# 5. Ray Dalio – Risk Parity (balanced, low volatility preference)
# ────────────────────────────────────────────────────────────────────────────
class DalioStrategy(StrategyAgent):
    name = "Dalio"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        # Risk parity: prefer low-volatility assets with stable returns
        if s.volatility < 0.20 and s.regime_score > 0:
            return Signal(TradingAction.BUY, 0.78)
        if s.volatility < 0.25 and abs(s.slope) < 1.0:
            return Signal(TradingAction.HOLD, 0.70)
        if s.volatility > 0.50:
            return Signal(TradingAction.SELL, 0.75)
        # Moderate vol — follow trend
        if s.slope > 0 and s.regime_score > 0:
            return Signal(TradingAction.BUY, 0.68)
        if s.slope < 0 and s.regime_score < 0:
            return Signal(TradingAction.SELL, 0.68)
        return Signal(TradingAction.HOLD, 0.60)


# ────────────────────────────────────────────────────────────────────────────
# 6. George Soros – Reflexivity / Breakout (momentum + volume spike)
# ────────────────────────────────────────────────────────────────────────────
class SorosStrategy(StrategyAgent):
    name = "Soros"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        # Breakout: strong positive slope + volume confirmation
        if s.slope > 1.5 and s.volume_ratio > 1.5 and s.price_change_5d_pct > 3:
            return Signal(TradingAction.BUY, 0.90)
        # Breakdown: strong negative slope + volume
        if s.slope < -1.5 and s.volume_ratio > 1.5 and s.price_change_5d_pct < -3:
            return Signal(TradingAction.SHORT, 0.88)
        # Moderate breakout
        if s.slope > 1.0 and s.regime_score > 0.3:
            return Signal(TradingAction.BUY, 0.75)
        # Moderate breakdown
        if s.slope < -1.0 and s.regime_score < -0.3:
            return Signal(TradingAction.SELL, 0.72)
        return Signal(TradingAction.HOLD, 0.55)


# ────────────────────────────────────────────────────────────────────────────
# 7. Bill Ackman – Activist / Concentrated (strong brand + undervalued)
# ────────────────────────────────────────────────────────────────────────────
class AckmanStrategy(StrategyAgent):
    name = "Ackman"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        score = 0.0
        # Strong brand proxy: high gross margin + large cap
        if s.gross_margin > 0.40 and s.market_cap_bucket >= 4:
            score += 1.5
        # Undervalued
        if s.intrinsic_value_gap > 0.15:
            score += 1.5
        elif s.intrinsic_value_gap > 0.05:
            score += 0.8
        # Financial discipline (low debt, high FCF)
        if s.debt_to_equity < 1.0 and s.fcf_yield > 0.05:
            score += 1.0
        # Volume spike (activist catalyst proxy)
        if s.volume_ratio > 2.0:
            score += 1.0

        if score >= 3.5:
            return Signal(TradingAction.BUY, min(0.90, 0.62 + score * 0.05))
        return Signal(TradingAction.HOLD, 0.58)


# ────────────────────────────────────────────────────────────────────────────
# 8. Stanley Druckenmiller – Macro Timing (regime + momentum)
# ────────────────────────────────────────────────────────────────────────────
class DruckenmillerStrategy(StrategyAgent):
    name = "Druckenmiller"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        # Strong macro conviction: positive regime + high momentum
        if s.regime_score > 0.5 and s.price_change_20d_pct > 5:
            return Signal(TradingAction.BUY, 0.89)
        # Negative macro: bearish regime
        if s.regime_score < -0.5 and s.price_change_20d_pct < -5:
            return Signal(TradingAction.SELL, 0.85)
        # Moderate signals
        if s.regime_score > 0.3 and s.slope > 0:
            return Signal(TradingAction.BUY, 0.72)
        if s.regime_score < -0.3 and s.slope < 0:
            return Signal(TradingAction.SELL, 0.70)
        return Signal(TradingAction.HOLD, 0.58)


# ────────────────────────────────────────────────────────────────────────────
# 9. Peter Lynch – Growth at Reasonable Price (PEG + earnings growth)
# ────────────────────────────────────────────────────────────────────────────
class PeterLynchStrategy(StrategyAgent):
    name = "PeterLynch"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        score = 0.0
        # PEG ratio sweet spot
        if 0 < s.peg_ratio < 1.0:
            score += 2.0
        elif 0 < s.peg_ratio < 1.5:
            score += 1.0
        # Earnings growth
        if s.earnings_growth > 0.20:
            score += 1.5
        elif s.earnings_growth > 0.10:
            score += 0.8
        # Revenue growth consistency
        if s.revenue_growth_consistency > 0.7:
            score += 1.0
        # Low debt
        if s.debt_to_equity < 0.5:
            score += 0.5
        # Positive regime
        if s.regime_score > 0.3:
            score += 0.5

        if score >= 3.5:
            return Signal(TradingAction.BUY, min(0.92, 0.60 + score * 0.06))
        elif score <= 1.0:
            return Signal(TradingAction.SELL, 0.62)
        return Signal(TradingAction.HOLD, 0.58)


# ────────────────────────────────────────────────────────────────────────────
# 10. Jim Simons – Quant Pattern (statistical signals + vol arb)
# ────────────────────────────────────────────────────────────────────────────
class JimSimonsStrategy(StrategyAgent):
    name = "JimSimons"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        score = 0.0
        # RSI mean reversion
        if s.rsi < 30:
            score += 1.5  # oversold → buy
        elif s.rsi > 70:
            score -= 1.5  # overbought → sell
        # Volume anomaly
        if s.volume_ratio > 2.0:
            score += 0.8 if s.price_change_pct > 0 else -0.8
        # Low vol + trend = exploitable pattern
        if s.volatility < 0.25 and abs(s.slope) > 0.5:
            score += 1.0 if s.slope > 0 else -1.0
        # SMA cross momentum
        if s.sma_cross > 1.03:
            score += 0.8
        elif s.sma_cross < 0.97:
            score -= 0.8

        if score >= 2.5:
            return Signal(TradingAction.BUY, min(0.92, 0.65 + abs(score) * 0.06))
        elif score <= -2.5:
            return Signal(TradingAction.SHORT, min(0.92, 0.65 + abs(score) * 0.06))
        return Signal(TradingAction.HOLD, 0.60)


# ────────────────────────────────────────────────────────────────────────────
# 11. Carl Icahn – Activist Value (undervalued + catalyst)
# ────────────────────────────────────────────────────────────────────────────
class CarlIcahnStrategy(StrategyAgent):
    name = "CarlIcahn"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        score = 0.0
        # Deep undervaluation
        if s.intrinsic_value_gap > 0.30:
            score += 2.0
        elif s.intrinsic_value_gap > 0.15:
            score += 1.0
        # Low P/B (asset breakup value)
        if s.pb_ratio > 0 and s.pb_ratio < 1.5:
            score += 1.0
        # Insider buying as catalyst signal
        if s.insider_net_buys > 2:
            score += 1.0
        # Large cap (activist target)
        if s.market_cap_bucket >= 3:
            score += 0.5
        # FCF generation
        if s.fcf_yield > 0.06:
            score += 0.8

        if score >= 3.5:
            return Signal(TradingAction.BUY, min(0.90, 0.62 + score * 0.05))
        return Signal(TradingAction.HOLD, 0.58)


# ────────────────────────────────────────────────────────────────────────────
# 12. David Tepper – Distressed Conviction (buy fear, bearish regime + value)
# ────────────────────────────────────────────────────────────────────────────
class DavidTepperStrategy(StrategyAgent):
    name = "DavidTepper"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        # Buy when others are fearful — bearish regime + strong fundamentals
        distressed_buy = (
            s.regime_score < -0.3
            and s.price_change_20d_pct < -10
            and s.roe > 0.08
            and s.current_ratio > 1.0
        )
        if distressed_buy:
            return Signal(TradingAction.BUY, 0.88)

        # Strong conviction: cheap + beaten down
        if s.pe_ratio > 0 and s.pe_ratio < 12 and s.price_change_20d_pct < -5:
            return Signal(TradingAction.BUY, 0.80)

        # Avoid obvious overvaluation
        if s.pe_ratio > 50 and s.regime_score > 0.5:
            return Signal(TradingAction.SELL, 0.72)

        return Signal(TradingAction.HOLD, 0.58)


# ────────────────────────────────────────────────────────────────────────────
# 13. Bill Miller – Contrarian Value (buy hated sectors, long-term mean reversion)
# ────────────────────────────────────────────────────────────────────────────
class BillMillerStrategy(StrategyAgent):
    name = "BillMiller"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        score = 0.0
        # Contrarian: negative sentiment + decent fundamentals
        if s.news_sentiment < -0.3 and s.roe > 0.05:
            score += 1.5
        # Beaten-down price
        if s.price_change_20d_pct < -10:
            score += 1.0
        # Low P/E contrarian play
        if s.pe_ratio > 0 and s.pe_ratio < 15:
            score += 1.0
        # Cash-generative
        if s.fcf_yield > 0.05:
            score += 0.8
        # Regime negative but not crashing
        if -0.5 < s.regime_score < 0:
            score += 0.5

        if score >= 3.0:
            return Signal(TradingAction.BUY, min(0.90, 0.62 + score * 0.06))
        return Signal(TradingAction.HOLD, 0.58)


# ────────────────────────────────────────────────────────────────────────────
# 14. John Paulson – Macro Event (extreme vol + directional bet)
# ────────────────────────────────────────────────────────────────────────────
class JohnPaulsonStrategy(StrategyAgent):
    name = "JohnPaulson"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        # Macro event: very high volatility = opportunity
        if s.volatility > 0.50:
            if s.regime_score < -0.5:
                return Signal(TradingAction.SHORT, 0.87)
            if s.regime_score > 0.3:
                return Signal(TradingAction.BUY, 0.82)
            return Signal(TradingAction.HOLD, 0.65)

        # Moderate vol + strong trend
        if s.volatility > 0.30 and abs(s.price_change_20d_pct) > 10:
            action = TradingAction.BUY if s.price_change_20d_pct > 0 else TradingAction.SELL
            return Signal(action, 0.78)

        return Signal(TradingAction.HOLD, 0.55)


# ────────────────────────────────────────────────────────────────────────────
# 15. Paul Tudor Jones – Macro Trend (strong trend following)
# ────────────────────────────────────────────────────────────────────────────
class PaulTudorJonesStrategy(StrategyAgent):
    name = "PaulTudorJones"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        # Strong trend following
        if s.sma_cross > 1.05 and s.slope > 1.0:
            return Signal(TradingAction.BUY, 0.90)
        if s.sma_cross < 0.95 and s.slope < -1.0:
            return Signal(TradingAction.SELL, 0.88)

        # Moderate trend
        if s.sma_cross > 1.02 and s.regime_score > 0:
            return Signal(TradingAction.BUY, 0.72)
        if s.sma_cross < 0.98 and s.regime_score < 0:
            return Signal(TradingAction.SELL, 0.70)

        return Signal(TradingAction.HOLD, 0.55)


# ────────────────────────────────────────────────────────────────────────────
# 16. Ken Griffin – Quant Volatility Arbitrage (vol surface exploitation)
# ────────────────────────────────────────────────────────────────────────────
class KenGriffinStrategy(StrategyAgent):
    name = "KenGriffin"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        # Vol arb: exploit mean-reverting volatility
        # Low vol + volume spike = vol expansion expected
        if s.volatility < 0.20 and s.volume_ratio > 2.0:
            return Signal(TradingAction.BUY, 0.82)
        # High vol + declining volume = vol compression expected
        if s.volatility > 0.45 and s.volume_ratio < 0.7:
            return Signal(TradingAction.BUY, 0.78)
        # Extreme vol with trend = ride the wave
        if s.volatility > 0.50 and abs(s.slope) > 2.0:
            action = TradingAction.BUY if s.slope > 0 else TradingAction.SHORT
            return Signal(action, 0.85)

        return Signal(TradingAction.HOLD, 0.60)


# ────────────────────────────────────────────────────────────────────────────
# 17. Steve Cohen – Event Driven (catalyst + fundamental quality)
# ────────────────────────────────────────────────────────────────────────────
class SteveCohenStrategy(StrategyAgent):
    name = "SteveCohen"

    def generate_signal(self, s: FinancialSnapshot) -> Signal:
        if s.is_fallback:
            return Signal(TradingAction.HOLD, 0.50)

        score = 0.0
        # Insider activity as event catalyst
        if s.insider_net_buys > 3:
            score += 1.5
        elif s.insider_net_buys > 1:
            score += 0.8
        # Volume spike (event proxy)
        if s.volume_ratio > 2.0:
            score += 1.0
        # Quality fundamentals
        if s.roe > 0.12 and s.operating_margin > 0.15:
            score += 1.0
        # Positive news sentiment
        if s.news_sentiment > 0.3:
            score += 0.8
        # Earnings beat proxy (recent positive price action)
        if s.price_change_pct > 3:
            score += 0.8

        if score >= 3.0:
            return Signal(TradingAction.BUY, min(0.90, 0.62 + score * 0.05))
        elif s.insider_net_buys < -3 and s.news_sentiment < -0.3:
            return Signal(TradingAction.SELL, 0.78)
        return Signal(TradingAction.HOLD, 0.58)


# ────────────────────────────────────────────────────────────────────────────
# Registry – all 17 strategies in execution order
# ────────────────────────────────────────────────────────────────────────────
ALL_INVESTOR_STRATEGIES: list[StrategyAgent] = [
    BuffettStrategy(),
    MungerStrategy(),
    BurryStrategy(),
    CathieWoodStrategy(),
    DalioStrategy(),
    SorosStrategy(),
    AckmanStrategy(),
    DruckenmillerStrategy(),
    PeterLynchStrategy(),
    JimSimonsStrategy(),
    CarlIcahnStrategy(),
    DavidTepperStrategy(),
    BillMillerStrategy(),
    JohnPaulsonStrategy(),
    PaulTudorJonesStrategy(),
    KenGriffinStrategy(),
    SteveCohenStrategy(),
]
