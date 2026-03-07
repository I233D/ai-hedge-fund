"""Adaptive Freshness Bounds for ES/NQ/SPX feeds.

Computes per-symbol, session-aware, VIX-scaled freshness thresholds:
    - RTH (Regular Trading Hours): tighter bounds (17s for futures)
    - Pre-market / overnight: relaxed bounds (2× RTH)
    - VIX scaling: high VIX → tighter (faster updates needed)

Maps to GSC ``DataFeedOrchestrator.getFreshnessBound()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TradingSession(str, Enum):
    RTH = "rth"              # Regular Trading Hours (9:30–16:00 ET)
    PRE_MARKET = "pre_market"  # 4:00–9:30 ET
    AFTER_HOURS = "after_hours"  # 16:00–20:00 ET
    OVERNIGHT = "overnight"    # 20:00–4:00 ET (futures only)


# Base freshness bounds in seconds per instrument type
BASE_BOUNDS: dict[str, int] = {
    "ES": 17,     # E-mini S&P — tightest (most liquid)
    "NQ": 17,     # E-mini Nasdaq — same liquidity tier
    "SPX": 60,    # Cash index — less frequent updates
    "MES": 20,    # Micro — slightly wider
    "MNQ": 20,    # Micro — slightly wider
}

# Session multipliers
SESSION_MULTIPLIERS: dict[TradingSession, float] = {
    TradingSession.RTH: 1.0,
    TradingSession.PRE_MARKET: 2.0,
    TradingSession.AFTER_HOURS: 2.0,
    TradingSession.OVERNIGHT: 3.0,
}


@dataclass(frozen=True, slots=True)
class FreshnessBound:
    """Computed freshness bound for a specific symbol and moment."""

    symbol: str
    bound_seconds: int
    session: TradingSession
    vix: float
    base_bound: int
    vix_factor: float
    session_multiplier: float


def detect_session(now: Optional[datetime] = None) -> TradingSession:
    """Detect current trading session based on Eastern Time.

    Uses UTC internally; adjusts for ET (UTC-5 / UTC-4 DST).
    For simplicity, assumes ET = UTC-5 (standard).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Convert to approximate ET (UTC-5)
    et_hour = (now.hour - 5) % 24
    et_minute = now.minute

    et_time = et_hour * 60 + et_minute

    if 570 <= et_time < 960:      # 9:30 – 16:00
        return TradingSession.RTH
    elif 240 <= et_time < 570:    # 4:00 – 9:30
        return TradingSession.PRE_MARKET
    elif 960 <= et_time < 1200:   # 16:00 – 20:00
        return TradingSession.AFTER_HOURS
    else:                          # 20:00 – 4:00
        return TradingSession.OVERNIGHT


class AdaptiveFreshnessCalculator:
    """Compute VIX-scaled, session-aware freshness bounds.

    Usage:
        calc = AdaptiveFreshnessCalculator()
        bound = calc.compute("ES", vix=18.5)
        # → FreshnessBound(symbol="ES", bound_seconds=17, ...)
    """

    def compute(
        self,
        symbol: str,
        vix: float = 18.0,
        session: Optional[TradingSession] = None,
        now: Optional[datetime] = None,
    ) -> FreshnessBound:
        """Compute adaptive freshness bound.

        Parameters
        ----------
        symbol:
            Instrument ticker (ES, NQ, SPX, MES, MNQ).
        vix:
            Current VIX level.  Default 18.0.
        session:
            Override session detection.  Auto-detected if None.
        now:
            Override current time for testing.
        """
        symbol = symbol.upper()
        if session is None:
            session = detect_session(now)

        base = BASE_BOUNDS.get(symbol, 30)
        vix_factor = max(0.5, min(2.0, vix / 20.0))
        session_mult = SESSION_MULTIPLIERS.get(session, 1.0)

        raw_bound = base * vix_factor * session_mult
        bound_seconds = max(15, min(900, int(raw_bound)))

        result = FreshnessBound(
            symbol=symbol,
            bound_seconds=bound_seconds,
            session=session,
            vix=vix,
            base_bound=base,
            vix_factor=round(vix_factor, 3),
            session_multiplier=session_mult,
        )

        logger.debug(
            "Freshness: %s bound=%ds session=%s vix=%.1f (base=%d × vix_factor=%.2f × session=%.1f)",
            symbol, bound_seconds, session.value, vix, base, vix_factor, session_mult,
        )

        return result
