"""SPX Options Training Hook – domain fine-tuning for all 19 agents.

Provides a training loop that evaluates agent performance on historical
snapshots and computes per-agent Sharpe ratios.  Used for calibration
and strategy validation.

This is a pure-Python, zero-LLM implementation.  No external dependencies
beyond numpy/pandas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.strategies.base import Signal, TradingAction
from src.strategies.investors import ALL_INVESTOR_STRATEGIES
from src.strategies.risk_guard import PortfolioManagerStrategy, RiskManagerStrategy
from src.strategies.snapshot import FinancialSnapshot

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformance:
    """Per-agent backtested metrics."""

    name: str
    total_signals: int = 0
    correct_signals: int = 0
    pnl_series: List[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct_signals / self.total_signals if self.total_signals > 0 else 0.0

    @property
    def sharpe_ratio(self) -> float:
        if len(self.pnl_series) < 2:
            return 0.0
        arr = np.array(self.pnl_series)
        mean_ret = np.mean(arr)
        std_ret = np.std(arr)
        if std_ret == 0:
            return 0.0
        return float(mean_ret / std_ret * np.sqrt(252))


@dataclass
class TrainingResult:
    """Aggregate training results across all agents."""

    agent_metrics: Dict[str, AgentPerformance]
    ensemble_sharpe: float = 0.0
    total_snapshots: int = 0


class SPXOptionsTrainer:
    """Train/validate all 19 agents on historical FinancialSnapshot data.

    Usage:
        trainer = SPXOptionsTrainer()
        result = trainer.train(snapshots, forward_returns)
    """

    def __init__(self) -> None:
        self._agents = list(ALL_INVESTOR_STRATEGIES) + [RiskManagerStrategy(), PortfolioManagerStrategy()]

    def train(
        self,
        snapshots: List[FinancialSnapshot],
        forward_returns: List[float],
    ) -> TrainingResult:
        """Evaluate all agents against historical snapshots.

        Parameters
        ----------
        snapshots:
            List of ``FinancialSnapshot`` objects (one per time period).
        forward_returns:
            Corresponding forward returns (%) for each snapshot.
            Positive = stock went up, negative = stock went down.

        Returns
        -------
        TrainingResult with per-agent and ensemble metrics.
        """
        if len(snapshots) != len(forward_returns):
            raise ValueError("snapshots and forward_returns must have the same length")

        agent_perf: Dict[str, AgentPerformance] = {
            a.name: AgentPerformance(name=a.name) for a in self._agents
        }

        ensemble_pnl: List[float] = []

        for snap, fwd_ret in zip(snapshots, forward_returns):
            period_signals: Dict[str, Signal] = {}

            for agent in self._agents:
                sig = agent.generate_signal(snap)
                perf = agent_perf[agent.name]
                perf.total_signals += 1

                # Map action to position: BUY=+1, SELL/SHORT=-1, HOLD/COVER=0
                position = 0.0
                if sig.action == TradingAction.BUY:
                    position = 1.0
                elif sig.action in (TradingAction.SELL, TradingAction.SHORT):
                    position = -1.0

                # P&L = position * forward_return * confidence
                pnl = position * fwd_ret * sig.confidence
                perf.pnl_series.append(pnl)

                # Correct if direction matches
                if (position > 0 and fwd_ret > 0) or (position < 0 and fwd_ret < 0) or (position == 0):
                    perf.correct_signals += 1

                period_signals[agent.name] = sig

            # Ensemble P&L (equal-weighted average)
            period_pnls = [p.pnl_series[-1] for p in agent_perf.values()]
            ensemble_pnl.append(sum(period_pnls) / len(period_pnls))

        # Compute ensemble Sharpe
        ensemble_sharpe = 0.0
        if len(ensemble_pnl) >= 2:
            arr = np.array(ensemble_pnl)
            mean_ret = np.mean(arr)
            std_ret = np.std(arr)
            if std_ret > 0:
                ensemble_sharpe = float(mean_ret / std_ret * np.sqrt(252))

        result = TrainingResult(
            agent_metrics=agent_perf,
            ensemble_sharpe=ensemble_sharpe,
            total_snapshots=len(snapshots),
        )

        logger.info("Training complete: %d snapshots, ensemble Sharpe=%.2f", len(snapshots), ensemble_sharpe)
        for name, perf in sorted(agent_perf.items()):
            logger.info("  %s: accuracy=%.1f%%, Sharpe=%.2f", name, perf.accuracy * 100, perf.sharpe_ratio)

        return result
