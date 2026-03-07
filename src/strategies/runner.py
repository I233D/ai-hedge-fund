"""StrategyRunner – parallel orchestrator for all 19 agents.

Mirrors the GSC ``AIAnalystSystem`` actor model.  Uses
``concurrent.futures.ThreadPoolExecutor`` for parallel execution of all
strategy agents (17 investors + RiskManager + PortfolioManager).

Usage:
    runner = StrategyRunner()
    decision = runner.run(snapshot)
    # decision.action, decision.confidence, decision.votes, decision.breakdown
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from src.strategies.aggregator import AggregatedDecision, SignalAggregator
from src.strategies.base import Signal, StrategyAgent
from src.strategies.investors import ALL_INVESTOR_STRATEGIES
from src.strategies.risk_guard import PortfolioManagerStrategy, RiskManagerStrategy
from src.strategies.snapshot import FinancialSnapshot

logger = logging.getLogger(__name__)


class StrategyRunner:
    """Orchestrates parallel execution of all strategy agents.

    Parameters
    ----------
    strategies:
        Custom list of investor strategies.  Defaults to all 17.
    include_risk_manager:
        Whether to include the volatility guard.  Default ``True``.
    include_portfolio_manager:
        Whether to include the quality gate.  Default ``True``.
    max_workers:
        Thread pool size.  Default ``19`` (one per agent).
    """

    def __init__(
        self,
        strategies: Optional[List[StrategyAgent]] = None,
        include_risk_manager: bool = True,
        include_portfolio_manager: bool = True,
        max_workers: int = 19,
    ) -> None:
        self._agents: List[StrategyAgent] = list(strategies or ALL_INVESTOR_STRATEGIES)
        if include_risk_manager:
            self._agents.append(RiskManagerStrategy())
        if include_portfolio_manager:
            self._agents.append(PortfolioManagerStrategy())
        self._max_workers = max_workers

    @property
    def agent_names(self) -> List[str]:
        return [a.name for a in self._agents]

    def _run_agent(self, agent: StrategyAgent, snapshot: FinancialSnapshot) -> Tuple[str, Signal]:
        """Execute a single agent (called inside thread pool)."""
        sig = agent.generate_signal(snapshot)
        return (agent.name, sig)

    def run(self, snapshot: FinancialSnapshot) -> AggregatedDecision:
        """Run all agents in parallel and aggregate signals.

        Returns an ``AggregatedDecision`` with the final action, confidence,
        per-agent votes, and action breakdown.
        """
        signals: List[Tuple[str, Signal]] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._run_agent, agent, snapshot): agent.name
                for agent in self._agents
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result()
                    signals.append(result)
                except Exception:
                    logger.exception("Agent %s failed — defaulting to HOLD", agent_name)
                    from src.strategies.base import TradingAction
                    signals.append((agent_name, Signal(TradingAction.HOLD, 0.50)))

        decision = SignalAggregator.aggregate(signals)

        logger.info(
            "StrategyRunner result: %s (confidence=%.2f) from %d agents",
            decision.action.value,
            decision.confidence,
            len(signals),
        )
        for name, sig in sorted(signals, key=lambda x: x[0]):
            logger.debug("  %s: %s (%.2f)", name, sig.action.value, sig.confidence)

        return decision

    def run_sequential(self, snapshot: FinancialSnapshot) -> AggregatedDecision:
        """Run all agents sequentially (for testing / debugging)."""
        signals: List[Tuple[str, Signal]] = []
        for agent in self._agents:
            sig = agent.generate_signal(snapshot)
            signals.append((agent.name, sig))
        return SignalAggregator.aggregate(signals)
