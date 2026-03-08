"""
IBKR ES/NQ/SPX 24h Pre-Market Feed System

Provides real-time futures data via IBKR Client Portal API with:
- Adaptive freshness bounds (17s RTH, VIX-scaled, pre-market relaxed)
- CircuitBreaker protection for stale data
- Ticker-aware agent weighting (ES/NQ/SPX)
- DataFeedOrchestrator for parallel multi-feed aggregation
"""

from src.feeds.ibkr_client import IBKRClient, IBKRQuote, IBKRConfig
from src.feeds.ibkr_feed import IBKRRealTimeFeed
from src.feeds.freshness import FreshnessBound, AdaptiveFreshnessCalculator
from src.feeds.circuit_breaker import CircuitBreaker, CircuitState
from src.feeds.orchestrator import DataFeedOrchestrator, FeedBundle
from src.feeds.ticker_weights import TickerWeights, get_ticker_weights

__all__ = [
    "IBKRClient",
    "IBKRQuote",
    "IBKRConfig",
    "IBKRRealTimeFeed",
    "FreshnessBound",
    "AdaptiveFreshnessCalculator",
    "CircuitBreaker",
    "CircuitState",
    "DataFeedOrchestrator",
    "FeedBundle",
    "TickerWeights",
    "get_ticker_weights",
]
