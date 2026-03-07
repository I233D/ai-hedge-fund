"""
Quantitative Investor Strategies – Production Implementation

Pure-math, zero-LLM-call strategies modeled after 17 famous investors.
Designed for parallel execution with confidence-weighted signal aggregation.

Architecture:
    FinancialSnapshot (data) → 17 StrategyAgents (parallel) → SignalAggregator → TradingDecision

Includes:
    - MacroDataProvider for VIX/rates/oil/DXY/gold injection
    - Domain multipliers for per-agent data weighting
    - SPXOptionsTrainer for domain fine-tuning on historical snapshots
"""

from src.strategies.snapshot import FinancialSnapshot, build_snapshot
from src.strategies.base import Signal, TradingAction, StrategyAgent
from src.strategies.aggregator import SignalAggregator, AggregatedDecision
from src.strategies.risk_guard import RiskManagerStrategy, PortfolioManagerStrategy
from src.strategies.runner import StrategyRunner
from src.strategies.macro import MacroData, DOMAIN_MULTIPLIERS, MACRO_BENEFIT_MAP, macro_adjustment
from src.strategies.trainer import SPXOptionsTrainer, TrainingResult, AgentPerformance

__all__ = [
    "FinancialSnapshot",
    "build_snapshot",
    "Signal",
    "TradingAction",
    "StrategyAgent",
    "SignalAggregator",
    "AggregatedDecision",
    "RiskManagerStrategy",
    "PortfolioManagerStrategy",
    "StrategyRunner",
    "MacroData",
    "DOMAIN_MULTIPLIERS",
    "MACRO_BENEFIT_MAP",
    "macro_adjustment",
    "SPXOptionsTrainer",
    "TrainingResult",
    "AgentPerformance",
]
