"""DataFeedOrchestrator – parallel multi-feed aggregation.

Coordinates all data feeds into a unified FeedBundle consumed by the
strategy runner.  Supports:
- GSC real-time feed (SPX options — existing)
- IBKR real-time feed (ES/NQ/SPX futures — 24h)
- Macro data feed (VIX/10Y/oil — shared)
- Financial Datasets API feed (prices/metrics — cached)

Architecture:
    DataFeedOrchestrator.get_feeds(ticker)
        ├─ IBKRRealTimeFeed.get_quote()       [parallel]
        ├─ MacroData (VIX/rates)              [parallel]
        ├─ FinancialSnapshot (from API)        [parallel]
        └─ TickerWeights                       [instant]
        → FeedBundle
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.feeds.freshness import AdaptiveFreshnessCalculator, FreshnessBound, TradingSession
from src.feeds.ibkr_client import IBKRConfig, IBKRQuote
from src.feeds.ibkr_feed import IBKRRealTimeFeed
from src.feeds.ticker_weights import TickerWeights, get_ticker_weights
from src.strategies.macro import MacroData
from src.strategies.snapshot import FinancialSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedBundle:
    """Aggregated data from all feeds for a single ticker."""

    ticker: str
    snapshot: FinancialSnapshot
    ibkr_quote: Optional[IBKRQuote]
    macro: MacroData
    ticker_weights: TickerWeights
    freshness: Optional[FreshnessBound]
    session: Optional[TradingSession]
    timestamp: float

    @property
    def is_futures(self) -> bool:
        return self.ticker_weights.instrument_type.startswith("futures")

    @property
    def is_premarket(self) -> bool:
        return self.session in (TradingSession.PRE_MARKET, TradingSession.OVERNIGHT)

    @property
    def has_live_feed(self) -> bool:
        return self.ibkr_quote is not None and self.ibkr_quote.age_seconds < 60


class DataFeedOrchestrator:
    """Parallel multi-feed aggregator.

    Usage:
        orchestrator = DataFeedOrchestrator()
        bundle = orchestrator.get_feeds("ES", snapshot, macro)
    """

    def __init__(
        self,
        ibkr_feed: Optional[IBKRRealTimeFeed] = None,
        ibkr_config: Optional[IBKRConfig] = None,
        max_workers: int = 4,
    ) -> None:
        self._ibkr_feed = ibkr_feed or IBKRRealTimeFeed(config=ibkr_config)
        self._freshness_calc = AdaptiveFreshnessCalculator()
        self._max_workers = max_workers

    def get_feeds(
        self,
        ticker: str,
        snapshot: Optional[FinancialSnapshot] = None,
        macro: Optional[MacroData] = None,
    ) -> FeedBundle:
        """Aggregate all feeds for a ticker into a FeedBundle.

        Parameters
        ----------
        ticker:
            Instrument symbol (ES, NQ, SPX, AAPL, etc.).
        snapshot:
            Pre-built FinancialSnapshot.  If None, creates a minimal one.
        macro:
            Current macro data.  If None, uses defaults.
        """
        ticker = ticker.upper()
        macro = macro or MacroData()
        snapshot = snapshot or FinancialSnapshot(ticker=ticker)
        ticker_weights = get_ticker_weights(ticker)

        ibkr_quote: Optional[IBKRQuote] = None
        freshness: Optional[FreshnessBound] = None

        # Only fetch IBKR for futures/index instruments
        if ticker_weights.instrument_type != "equity":
            freshness = self._freshness_calc.compute(ticker, vix=macro.vix)
            ibkr_quote = self._ibkr_feed.get_quote(ticker, vix=macro.vix)

            # Enrich snapshot with IBKR quote data
            if ibkr_quote and ibkr_quote.last > 0:
                snapshot.price = ibkr_quote.last

        bundle = FeedBundle(
            ticker=ticker,
            snapshot=snapshot,
            ibkr_quote=ibkr_quote,
            macro=macro,
            ticker_weights=ticker_weights,
            freshness=freshness,
            session=freshness.session if freshness else None,
            timestamp=time.time(),
        )

        logger.info(
            "FEED_BUNDLE ticker=%s type=%s session=%s live=%s freshness=%ss",
            ticker,
            ticker_weights.instrument_type,
            bundle.session.value if bundle.session else "N/A",
            bundle.has_live_feed,
            freshness.bound_seconds if freshness else "N/A",
        )

        return bundle

    def get_feeds_batch(
        self,
        tickers: List[str],
        snapshots: Optional[Dict[str, FinancialSnapshot]] = None,
        macro: Optional[MacroData] = None,
    ) -> Dict[str, FeedBundle]:
        """Fetch feed bundles for multiple tickers in parallel."""
        snapshots = snapshots or {}
        results: Dict[str, FeedBundle] = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(
                    self.get_feeds,
                    ticker,
                    snapshots.get(ticker.upper()),
                    macro,
                ): ticker
                for ticker in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    bundle = future.result()
                    results[ticker.upper()] = bundle
                except Exception:
                    logger.exception("Feed orchestration failed for %s", ticker)

        return results

    def close(self) -> None:
        self._ibkr_feed.close()
