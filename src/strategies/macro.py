"""MacroDataProvider + domain multipliers for agent weighting.

Each agent applies a domain-specific multiplier to its primary data fields,
producing a weighted confidence score in the aggregator.

Macro data (VIX, 10Y yield, oil, DXY, gold) is injected per-agent to enhance
signal quality.  Production-ready, pure math, < 0.1 ms overhead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MacroData:
    """External macro indicators injected into each agent."""

    vix: float = 18.0           # CBOE VIX
    ten_year_yield: float = 4.0 # US 10Y Treasury
    oil: float = 72.0           # WTI crude
    dxy: float = 103.0          # US Dollar Index
    gold: float = 2650.0        # Gold spot
    cpi_yoy: float = 3.0        # CPI year-over-year %
    gdp_growth: float = 2.5     # GDP growth %
    credit_spread: float = 1.5  # IG credit spread bps/100


# ────────────────────────────────────────────────────────────────────────────
# Domain multipliers — per-agent data weight overrides
# ────────────────────────────────────────────────────────────────────────────
# Maps agent name → { data_field: multiplier }
# Used in aggregator: final_confidence = raw_confidence * domain_multiplier
DOMAIN_MULTIPLIERS: dict[str, dict[str, float]] = {
    "Buffett":         {"regime_score": 1.2, "intrinsic_value_gap": 1.5, "roe": 1.3},
    "Munger":          {"roic": 1.5, "gross_margin": 1.3, "debt_to_equity": 1.2},
    "Burry":           {"slope": 2.0, "volatility": 1.8, "fcf_yield": 1.5},
    "CathieWood":      {"revenue_growth": 1.8, "earnings_growth": 1.5, "regime_score": 1.2},
    "Dalio":           {"volatility": 1.5, "regime_score": 1.3, "slope": 1.0},
    "Soros":           {"slope": 1.8, "volume_ratio": 1.6, "regime_score": 1.2},
    "Ackman":          {"intrinsic_value_gap": 1.5, "gross_margin": 1.3, "volume_ratio": 1.4},
    "Druckenmiller":   {"regime_score": 1.8, "price_change_20d_pct": 1.5, "slope": 1.2},
    "PeterLynch":      {"peg_ratio": 1.8, "earnings_growth": 1.5, "revenue_growth_consistency": 1.3},
    "JimSimons":       {"rsi": 1.6, "volatility": 1.4, "sma_cross": 1.5, "volume_ratio": 1.3},
    "CarlIcahn":       {"intrinsic_value_gap": 1.8, "pb_ratio": 1.4, "insider_net_buys": 1.5},
    "DavidTepper":     {"regime_score": 1.6, "price_change_20d_pct": 1.5, "pe_ratio": 1.3},
    "BillMiller":      {"news_sentiment": 1.5, "price_change_20d_pct": 1.4, "fcf_yield": 1.3},
    "JohnPaulson":     {"volatility": 2.0, "regime_score": 1.5},
    "PaulTudorJones":  {"sma_cross": 1.8, "slope": 1.6},
    "KenGriffin":      {"volatility": 1.8, "volume_ratio": 1.5, "slope": 1.3},
    "SteveCohen":      {"insider_net_buys": 1.6, "volume_ratio": 1.5, "news_sentiment": 1.4},
    "RiskManager":     {"volatility": 2.0, "volume_ratio": 1.5},
    "PortfolioManager": {"roe": 1.0, "fcf_yield": 1.0},
}


# ────────────────────────────────────────────────────────────────────────────
# Macro benefit mapping — which macro fields benefit each agent most
# ────────────────────────────────────────────────────────────────────────────
MACRO_BENEFIT_MAP: dict[str, list[str]] = {
    "Buffett":         ["ten_year_yield", "cpi_yoy"],
    "Munger":          ["vix", "credit_spread"],
    "Burry":           ["vix", "credit_spread"],
    "CathieWood":      ["ten_year_yield", "gdp_growth"],
    "Dalio":           ["dxy", "oil", "gold", "ten_year_yield"],
    "Soros":           ["dxy", "gold"],
    "Ackman":          ["credit_spread"],
    "Druckenmiller":   ["cpi_yoy", "gdp_growth", "oil"],
    "PeterLynch":      ["gdp_growth", "cpi_yoy"],
    "JimSimons":       ["vix"],
    "CarlIcahn":       ["credit_spread"],
    "DavidTepper":     ["credit_spread", "vix"],
    "BillMiller":      ["vix"],
    "JohnPaulson":     ["vix", "gold"],
    "PaulTudorJones":  ["dxy", "oil"],
    "KenGriffin":      ["vix"],
    "SteveCohen":      ["vix"],
    "RiskManager":     ["vix"],
    "PortfolioManager": ["vix", "ten_year_yield"],
}


def macro_adjustment(agent_name: str, macro: MacroData) -> float:
    """Compute a macro-based confidence adjustment for a given agent.

    Returns a value in [-0.10, +0.10] that can be added to the agent's
    raw confidence to incorporate macro conditions.
    """
    adj = 0.0

    benefits = MACRO_BENEFIT_MAP.get(agent_name, [])
    if not benefits:
        return 0.0

    if "vix" in benefits:
        if macro.vix < 15:
            adj += 0.03   # calm market
        elif macro.vix > 30:
            adj -= 0.05   # fearful market

    if "ten_year_yield" in benefits:
        if macro.ten_year_yield < 3.0:
            adj += 0.03   # low rates favour growth
        elif macro.ten_year_yield > 5.0:
            adj -= 0.03   # high rates hurt valuations

    if "credit_spread" in benefits:
        if macro.credit_spread > 3.0:
            adj -= 0.04   # stress
        elif macro.credit_spread < 1.0:
            adj += 0.02   # benign

    if "oil" in benefits:
        if macro.oil > 100:
            adj -= 0.02   # inflation pressure
        elif macro.oil < 50:
            adj += 0.02

    if "dxy" in benefits:
        if macro.dxy > 110:
            adj -= 0.02   # strong dollar hurts
        elif macro.dxy < 95:
            adj += 0.02

    if "gold" in benefits:
        if macro.gold > 3000:
            adj += 0.02   # flight to safety
        elif macro.gold < 2000:
            adj -= 0.01

    if "gdp_growth" in benefits:
        if macro.gdp_growth > 3.0:
            adj += 0.03
        elif macro.gdp_growth < 1.0:
            adj -= 0.03

    if "cpi_yoy" in benefits:
        if macro.cpi_yoy > 5.0:
            adj -= 0.03
        elif macro.cpi_yoy < 2.0:
            adj += 0.02

    return max(-0.10, min(0.10, adj))
