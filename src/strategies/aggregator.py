"""Confidence-weighted signal aggregator (parallel reduce → final decision).

Maps GSC ``SignalAggregator`` to Python.  Supports the full action set:
BUY / SELL / SHORT / COVER / HOLD.

Algorithm:
    1. Collect (name, Signal) tuples from all 17 investor agents.
    2. Group by action, sum confidence weights.
    3. The action with the highest cumulative confidence wins.
    4. Final confidence = weighted_sum / total_confidence (normalised).
    5. Risk manager can veto: if RiskManager says COVER with ≥ 0.90
       confidence, override to COVER.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.strategies.base import Signal, TradingAction


@dataclass(frozen=True, slots=True)
class AggregatedDecision:
    action: TradingAction
    confidence: float
    votes: Dict[str, Signal]  # agent_name → Signal
    breakdown: Dict[TradingAction, float]  # action → total weighted confidence


def aggregate(
    signals: List[Tuple[str, Signal]],
    risk_override_threshold: float = 0.90,
) -> AggregatedDecision:
    """Parallel-reduce signals into a single trading decision.

    Parameters
    ----------
    signals:
        List of ``(agent_name, Signal)`` pairs from all agents (including
        RiskManager and PortfolioManager).
    risk_override_threshold:
        If RiskManager confidence ≥ this value and action is COVER,
        the final decision is forced to COVER.
    """
    if not signals:
        return AggregatedDecision(
            action=TradingAction.HOLD,
            confidence=0.50,
            votes={},
            breakdown={TradingAction.HOLD: 0.50},
        )

    votes: Dict[str, Signal] = {}
    risk_signal: Signal | None = None

    for name, sig in signals:
        votes[name] = sig
        if name == "RiskManager":
            risk_signal = sig

    # Risk manager veto check
    if risk_signal and risk_signal.action == TradingAction.COVER and risk_signal.confidence >= risk_override_threshold:
        return AggregatedDecision(
            action=TradingAction.COVER,
            confidence=risk_signal.confidence,
            votes=votes,
            breakdown={TradingAction.COVER: risk_signal.confidence},
        )

    # Weighted vote aggregation (exclude infrastructure agents from voting)
    infra_agents = {"RiskManager", "PortfolioManager"}
    action_weights: Dict[TradingAction, float] = defaultdict(float)
    total_weight = 0.0

    for name, sig in signals:
        if name in infra_agents:
            continue
        action_weights[sig.action] += sig.confidence
        total_weight += sig.confidence

    if total_weight == 0:
        return AggregatedDecision(
            action=TradingAction.HOLD,
            confidence=0.50,
            votes=votes,
            breakdown=dict(action_weights),
        )

    # Winner = action with highest cumulative confidence
    winning_action = max(action_weights, key=action_weights.get)  # type: ignore[arg-type]
    winning_weight = action_weights[winning_action]
    final_confidence = winning_weight / total_weight

    return AggregatedDecision(
        action=winning_action,
        confidence=round(final_confidence, 4),
        votes=votes,
        breakdown=dict(action_weights),
    )


class SignalAggregator:
    """Stateless aggregator — call ``aggregate()`` directly or use this wrapper."""

    @staticmethod
    def aggregate(
        signals: List[Tuple[str, Signal]],
        risk_override_threshold: float = 0.90,
    ) -> AggregatedDecision:
        return aggregate(signals, risk_override_threshold)
