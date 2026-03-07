"""IBKRRealTimeFeed – cached, circuit-breaker-protected futures feed.

Wraps IBKRClient with:
- In-memory quote cache (per-symbol)
- Adaptive freshness: only re-fetches when cache exceeds bound
- CircuitBreaker: blocks requests on repeated failures
- Freshness markers for downstream logging
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.feeds.circuit_breaker import CircuitBreaker
from src.feeds.freshness import AdaptiveFreshnessCalculator, FreshnessBound
from src.feeds.ibkr_client import IBKRClient, IBKRConfig, IBKRQuote

logger = logging.getLogger(__name__)


@dataclass
class FreshnessMarker:
    """Log record for feed freshness tracking."""

    symbol: str
    bound_seconds: int
    actual_age_seconds: float
    is_fresh: bool
    source: str  # "live" | "cache" | "fallback"
    timestamp: float = field(default_factory=time.time)


class IBKRRealTimeFeed:
    """Real-time ES/NQ/SPX feed with adaptive freshness.

    Usage:
        feed = IBKRRealTimeFeed()
        quote = feed.get_quote("ES", vix=18.5)
        # Returns fresh quote or cached if within freshness bound
    """

    def __init__(
        self,
        client: Optional[IBKRClient] = None,
        config: Optional[IBKRConfig] = None,
    ) -> None:
        self._client = client or IBKRClient(config)
        self._cache: Dict[str, IBKRQuote] = {}
        self._freshness_calc = AdaptiveFreshnessCalculator()
        self._circuit_breaker = CircuitBreaker(name="IBKR", failure_threshold=5, recovery_timeout=30.0)
        self._markers: List[FreshnessMarker] = []

    @property
    def circuit_state(self) -> str:
        return self._circuit_breaker.state.value

    @property
    def markers(self) -> List[FreshnessMarker]:
        return list(self._markers)

    def _record_marker(self, symbol: str, bound: FreshnessBound, age: float, fresh: bool, source: str) -> None:
        marker = FreshnessMarker(
            symbol=symbol,
            bound_seconds=bound.bound_seconds,
            actual_age_seconds=round(age, 2),
            is_fresh=fresh,
            source=source,
        )
        self._markers.append(marker)
        # Keep last 1000 markers to prevent memory growth
        if len(self._markers) > 1000:
            self._markers = self._markers[-500:]

        logger.info(
            "IBKR_FEED_%s symbol=%s last=%.2f age=%.1fs bound=%ds",
            "FRESH" if fresh else "STALE",
            symbol,
            self._cache.get(symbol, IBKRQuote(symbol, 0, 0, 0, 0, 0)).last,
            age,
            bound.bound_seconds,
        )

    def get_quote(
        self,
        symbol: str,
        vix: float = 18.0,
        force_refresh: bool = False,
    ) -> Optional[IBKRQuote]:
        """Get a futures quote, using cache if within freshness bound.

        Parameters
        ----------
        symbol:
            Ticker (ES, NQ, SPX).
        vix:
            Current VIX for adaptive freshness scaling.
        force_refresh:
            Bypass cache and fetch live data.
        """
        symbol = symbol.upper()
        bound = self._freshness_calc.compute(symbol, vix=vix)

        # Check cache freshness
        cached = self._cache.get(symbol)
        if cached and not force_refresh:
            age = cached.age_seconds
            if age < bound.bound_seconds:
                self._record_marker(symbol, bound, age, True, "cache")
                return cached

        # Circuit breaker check
        if not self._circuit_breaker.is_available:
            logger.warning("IBKR circuit open for %s — returning stale cache", symbol)
            if cached:
                self._record_marker(symbol, bound, cached.age_seconds, False, "fallback")
                return cached
            return None

        # Fetch live quote
        quote = self._client.get_quote(symbol)
        if quote:
            self._cache[symbol] = quote
            self._circuit_breaker.record_success()
            self._record_marker(symbol, bound, 0.0, True, "live")
            return quote
        else:
            self._circuit_breaker.record_failure()
            if cached:
                self._record_marker(symbol, bound, cached.age_seconds, False, "fallback")
                return cached
            return None

    def get_quotes_batch(
        self,
        symbols: List[str],
        vix: float = 18.0,
    ) -> Dict[str, IBKRQuote]:
        """Fetch quotes for multiple symbols."""
        results: Dict[str, IBKRQuote] = {}
        for symbol in symbols:
            quote = self.get_quote(symbol, vix=vix)
            if quote:
                results[symbol.upper()] = quote
        return results

    def invalidate_cache(self, symbol: Optional[str] = None) -> None:
        """Invalidate cached quotes."""
        if symbol:
            self._cache.pop(symbol.upper(), None)
        else:
            self._cache.clear()

    def close(self) -> None:
        self._client.close()
