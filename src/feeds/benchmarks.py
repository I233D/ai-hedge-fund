"""ES/NQ/SPY benchmark comparison for backtesting.

Extends the existing SPY benchmark to support ES and NQ futures as
benchmark instruments.  Used by the backtester for relative performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Performance metrics for a single benchmark."""

    ticker: str
    total_return_pct: float
    annualised_return_pct: float
    volatility: float
    sharpe_ratio: float
    max_drawdown_pct: float
    sortino_ratio: float


# Benchmark ticker mapping
BENCHMARK_MAP: Dict[str, str] = {
    "ES": "SPY",     # ES benchmarked against SPY
    "NQ": "QQQ",     # NQ benchmarked against QQQ
    "SPX": "SPY",    # SPX → SPY
    "MES": "SPY",
    "MNQ": "QQQ",
}


def _compute_metrics(returns: np.ndarray, name: str) -> BenchmarkResult:
    """Compute performance metrics from daily returns."""
    if len(returns) < 2:
        return BenchmarkResult(
            ticker=name, total_return_pct=0.0, annualised_return_pct=0.0,
            volatility=0.0, sharpe_ratio=0.0, max_drawdown_pct=0.0, sortino_ratio=0.0,
        )

    cumulative = np.cumprod(1 + returns)
    total_return = float(cumulative[-1] - 1) * 100
    n_days = len(returns)
    ann_return = float((cumulative[-1] ** (252 / max(n_days, 1)) - 1) * 100)
    vol = float(np.std(returns) * np.sqrt(252))

    # Sharpe (risk-free = 0 for simplicity)
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    max_dd = float(np.min(drawdowns) * 100)

    # Sortino
    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if len(downside) > 0 else 0.0
    sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    return BenchmarkResult(
        ticker=name,
        total_return_pct=round(total_return, 2),
        annualised_return_pct=round(ann_return, 2),
        volatility=round(vol, 4),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown_pct=round(max_dd, 2),
        sortino_ratio=round(sortino, 2),
    )


def compute_benchmarks(
    strategy_returns: List[float],
    benchmark_returns: Optional[Dict[str, List[float]]] = None,
    ticker: str = "SPY",
) -> Dict[str, BenchmarkResult]:
    """Compute strategy and benchmark performance metrics.

    Parameters
    ----------
    strategy_returns:
        Daily returns for the strategy (as decimals, e.g., 0.02 = 2%).
    benchmark_returns:
        Dict of {benchmark_name: daily_returns}.
    ticker:
        Primary ticker being traded (for benchmark selection).

    Returns
    -------
    Dict with "strategy" key plus benchmark keys.
    """
    results: Dict[str, BenchmarkResult] = {}

    strat_arr = np.array(strategy_returns, dtype=np.float64)
    results["strategy"] = _compute_metrics(strat_arr, "Strategy")

    if benchmark_returns:
        for name, rets in benchmark_returns.items():
            arr = np.array(rets, dtype=np.float64)
            results[name] = _compute_metrics(arr, name)

    return results


def get_benchmark_ticker(trading_ticker: str) -> str:
    """Get the appropriate benchmark ticker for comparison."""
    return BENCHMARK_MAP.get(trading_ticker.upper(), "SPY")
