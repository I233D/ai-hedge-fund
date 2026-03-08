"""Base types and abstract strategy for quantitative investor agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.snapshot import FinancialSnapshot


class TradingAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class Signal:
    action: TradingAction
    confidence: float  # 0.0 – 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))


class StrategyAgent:
    """Base class for all quantitative investor strategies.

    Subclasses must implement ``generate_signal`` with pure math — no LLM or
    network calls — so the full 17-agent sweep completes in < 1 ms.
    """

    name: str = "BaseAgent"

    def generate_signal(self, snapshot: FinancialSnapshot) -> Signal:
        raise NotImplementedError
