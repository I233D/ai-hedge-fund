"""Ticker-aware agent weighting for ES/NQ/SPX.

Different instruments have different characteristics that shift which
agents are most relevant:
    - ES: high-liquidity futures, favours macro/momentum/vol agents
    - NQ: tech-heavy, favours growth/innovation agents
    - SPX: cash index, favours value/fundamental agents
    - Individual stocks: standard weighting (1.0 for all)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True, slots=True)
class TickerWeights:
    """Per-agent confidence multipliers for a specific ticker class."""

    ticker: str
    instrument_type: str  # "futures_es" | "futures_nq" | "index_spx" | "equity"
    weights: Dict[str, float]  # agent_name → multiplier (0.5–1.5)


# ────────────────────────────────────────────────────────────────────────────
# Weight profiles per instrument
# ────────────────────────────────────────────────────────────────────────────

_ES_WEIGHTS: Dict[str, float] = {
    # Macro/momentum agents emphasised for ES
    "Buffett": 0.8,
    "Munger": 0.8,
    "Burry": 1.2,
    "CathieWood": 0.7,
    "Dalio": 1.4,          # Risk parity loves index futures
    "Soros": 1.3,           # Reflexivity + breakout
    "Ackman": 0.7,
    "Druckenmiller": 1.5,   # Macro king → ES is macro
    "PeterLynch": 0.6,
    "JimSimons": 1.3,       # Quant pattern + vol arb
    "CarlIcahn": 0.5,
    "DavidTepper": 1.1,
    "BillMiller": 0.8,
    "JohnPaulson": 1.4,     # Macro event
    "PaulTudorJones": 1.5,  # Trend following on futures
    "KenGriffin": 1.4,      # Vol arb
    "SteveCohen": 0.9,
    "RiskManager": 1.2,
    "PortfolioManager": 1.0,
}

_NQ_WEIGHTS: Dict[str, float] = {
    # Growth/tech/innovation agents emphasised for NQ
    "Buffett": 0.7,
    "Munger": 0.8,
    "Burry": 1.1,
    "CathieWood": 1.5,     # Innovation → NQ
    "Dalio": 1.1,
    "Soros": 1.2,
    "Ackman": 0.8,
    "Druckenmiller": 1.3,
    "PeterLynch": 1.0,
    "JimSimons": 1.3,
    "CarlIcahn": 0.5,
    "DavidTepper": 1.0,
    "BillMiller": 0.9,
    "JohnPaulson": 1.2,
    "PaulTudorJones": 1.3,
    "KenGriffin": 1.3,
    "SteveCohen": 1.0,
    "RiskManager": 1.2,
    "PortfolioManager": 1.0,
}

_SPX_WEIGHTS: Dict[str, float] = {
    # Balanced — fundamental agents slightly emphasised
    "Buffett": 1.1,
    "Munger": 1.1,
    "Burry": 1.0,
    "CathieWood": 0.9,
    "Dalio": 1.2,
    "Soros": 1.1,
    "Ackman": 0.9,
    "Druckenmiller": 1.3,
    "PeterLynch": 0.9,
    "JimSimons": 1.1,
    "CarlIcahn": 0.7,
    "DavidTepper": 1.0,
    "BillMiller": 0.9,
    "JohnPaulson": 1.2,
    "PaulTudorJones": 1.3,
    "KenGriffin": 1.2,
    "SteveCohen": 0.9,
    "RiskManager": 1.1,
    "PortfolioManager": 1.0,
}

_EQUITY_WEIGHTS: Dict[str, float] = {
    # Standard 1.0 for all — individual stocks use default strategy weighting
    "Buffett": 1.0,
    "Munger": 1.0,
    "Burry": 1.0,
    "CathieWood": 1.0,
    "Dalio": 1.0,
    "Soros": 1.0,
    "Ackman": 1.0,
    "Druckenmiller": 1.0,
    "PeterLynch": 1.0,
    "JimSimons": 1.0,
    "CarlIcahn": 1.0,
    "DavidTepper": 1.0,
    "BillMiller": 1.0,
    "JohnPaulson": 1.0,
    "PaulTudorJones": 1.0,
    "KenGriffin": 1.0,
    "SteveCohen": 1.0,
    "RiskManager": 1.0,
    "PortfolioManager": 1.0,
}

# Futures/index symbols
_FUTURES_SYMBOLS = {"ES", "NQ", "MES", "MNQ", "RTY", "YM"}
_INDEX_SYMBOLS = {"SPX", "NDX", "DJI", "RUT"}


def get_ticker_weights(ticker: str) -> TickerWeights:
    """Get agent confidence multipliers for a given ticker.

    Parameters
    ----------
    ticker:
        Instrument ticker (ES, NQ, SPX, AAPL, etc.).

    Returns
    -------
    TickerWeights with per-agent multipliers.
    """
    ticker = ticker.upper()

    if ticker in ("ES", "MES"):
        return TickerWeights(ticker=ticker, instrument_type="futures_es", weights=_ES_WEIGHTS)
    elif ticker in ("NQ", "MNQ"):
        return TickerWeights(ticker=ticker, instrument_type="futures_nq", weights=_NQ_WEIGHTS)
    elif ticker in _INDEX_SYMBOLS:
        return TickerWeights(ticker=ticker, instrument_type="index_spx", weights=_SPX_WEIGHTS)
    elif ticker in _FUTURES_SYMBOLS:
        return TickerWeights(ticker=ticker, instrument_type="futures_es", weights=_ES_WEIGHTS)
    else:
        return TickerWeights(ticker=ticker, instrument_type="equity", weights=_EQUITY_WEIGHTS)
