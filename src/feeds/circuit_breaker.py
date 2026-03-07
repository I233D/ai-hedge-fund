"""CircuitBreaker – protects against stale/failing data feeds.

Three states:
    CLOSED  → normal operation (requests pass through)
    OPEN    → feed is broken (requests blocked, return cached/fallback)
    HALF_OPEN → testing recovery (one request allowed)

Transitions:
    CLOSED → OPEN: failure_count >= threshold
    OPEN → HALF_OPEN: after recovery_timeout
    HALF_OPEN → CLOSED: success
    HALF_OPEN → OPEN: failure
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for data feed protection.

    Parameters
    ----------
    name:
        Feed identifier for logging.
    failure_threshold:
        Number of consecutive failures before opening circuit.
    recovery_timeout:
        Seconds to wait before testing recovery (HALF_OPEN).
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _total_trips: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("CircuitBreaker[%s] → HALF_OPEN (testing recovery)", self.name)
        return self._state

    @property
    def is_available(self) -> bool:
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful request — reset circuit to CLOSED."""
        if self._state != CircuitState.CLOSED:
            logger.info("CircuitBreaker[%s] → CLOSED (recovered)", self.name)
        self._state = CircuitState.CLOSED
        self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request — may trip circuit to OPEN."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                self._total_trips += 1
                logger.warning(
                    "CircuitBreaker[%s] → OPEN (failures=%d, total_trips=%d)",
                    self.name,
                    self._failure_count,
                    self._total_trips,
                )
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Force reset to CLOSED (manual override)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        logger.info("CircuitBreaker[%s] → CLOSED (manual reset)", self.name)
