"""FinancialSnapshot – unified data record consumed by all strategy agents.

Mirrors the GSC ``GreeksByStrikeSnapshot`` concept but maps to financial
market data available through the Financial Datasets API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FinancialSnapshot:
    """Point-in-time financial data for a single ticker.

    Every field has a safe default so strategies never crash on missing data.
    The ``is_fallback`` flag corresponds to the GSC zero-fill / historical-
    fallback marker — agents should respect it and default to HOLD.
    """

    ticker: str = ""

    # Price dynamics
    price: float = 0.0
    price_change_pct: float = 0.0        # 1-day return
    price_change_5d_pct: float = 0.0     # 5-day return
    price_change_20d_pct: float = 0.0    # 20-day return
    volatility: float = 0.0              # annualised daily vol
    volume_ratio: float = 1.0            # today vol / 20d avg vol

    # Trend indicators
    slope: float = 0.0                   # linear regression slope of 20d prices
    sma_cross: float = 1.0              # SMA-10 / SMA-50 ratio  (>1 = bullish)
    rsi: float = 50.0                    # 14-day RSI
    regime_score: float = 0.0            # composite regime  (-1 bear … +1 bull)

    # Fundamentals
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    ps_ratio: float = 0.0
    peg_ratio: float = 0.0
    roe: float = 0.0
    roic: float = 0.0
    debt_to_equity: float = 0.0
    current_ratio: float = 0.0
    operating_margin: float = 0.0
    gross_margin: float = 0.0
    net_margin: float = 0.0
    fcf_yield: float = 0.0              # FCF / market cap

    # Growth
    revenue_growth: float = 0.0
    earnings_growth: float = 0.0
    fcf_growth: float = 0.0
    revenue_growth_consistency: float = 0.0  # 0–1 score

    # Valuation
    intrinsic_value_gap: float = 0.0     # (intrinsic - price) / price  (+ = undervalued)
    ev_to_ebitda: float = 0.0

    # Insider / sentiment
    insider_net_buys: int = 0
    insider_buy_ratio: float = 0.5       # buys / total transactions
    news_sentiment: float = 0.0          # -1 … +1

    # Market cap bucket (micro=1, small=2, mid=3, large=4, mega=5)
    market_cap_bucket: int = 3

    # Fallback / data quality
    is_fallback: bool = False            # True when data is stale / zero-fill


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or math.isnan(b) or math.isinf(b):
        return default
    return a / b


def _safe_get(obj: object, attr: str, default: float = 0.0) -> float:
    val = getattr(obj, attr, None)
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return default
    return float(val)


def build_snapshot(
    ticker: str,
    prices: Optional[List] = None,
    metrics: Optional[List] = None,
    line_items: Optional[List] = None,
    insider_trades: Optional[List] = None,
    news: Optional[List] = None,
    market_cap: Optional[float] = None,
) -> FinancialSnapshot:
    """Build a ``FinancialSnapshot`` from raw API data.

    Accepts the same objects returned by ``src.tools.api`` functions.  All
    inputs are optional — missing data yields safe defaults.
    """
    snap = FinancialSnapshot(ticker=ticker)

    # ── Price dynamics ──────────────────────────────────────────────────
    if prices and len(prices) >= 2:
        closes = np.array([p.close for p in prices if p.close], dtype=np.float64)
        if len(closes) >= 2:
            snap.price = float(closes[-1])
            snap.price_change_pct = float((closes[-1] / closes[-2] - 1) * 100)
            if len(closes) >= 5:
                snap.price_change_5d_pct = float((closes[-1] / closes[-5] - 1) * 100)
            if len(closes) >= 20:
                snap.price_change_20d_pct = float((closes[-1] / closes[-20] - 1) * 100)

            # Volatility (annualised)
            log_returns = np.diff(np.log(closes[closes > 0]))
            if len(log_returns) >= 5:
                snap.volatility = float(np.std(log_returns) * math.sqrt(252))

            # Slope via linear regression on last 20 prices
            window = closes[-20:] if len(closes) >= 20 else closes
            x = np.arange(len(window))
            if len(window) >= 3:
                coeffs = np.polyfit(x, window, 1)
                snap.slope = float(coeffs[0])

            # SMA cross
            if len(closes) >= 50:
                sma10 = float(np.mean(closes[-10:]))
                sma50 = float(np.mean(closes[-50:]))
                snap.sma_cross = _safe_div(sma10, sma50, 1.0)

            # RSI (14-day)
            if len(closes) >= 15:
                deltas = np.diff(closes[-15:])
                gains = np.where(deltas > 0, deltas, 0.0)
                losses = np.where(deltas < 0, -deltas, 0.0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    snap.rsi = float(100 - 100 / (1 + rs))
                else:
                    snap.rsi = 100.0

            # Volume ratio
            volumes = np.array([p.volume for p in prices if p.volume and p.volume > 0], dtype=np.float64)
            if len(volumes) >= 20:
                avg_vol = float(np.mean(volumes[-20:]))
                if avg_vol > 0:
                    snap.volume_ratio = float(volumes[-1] / avg_vol)

        # Regime score (composite: SMA cross + momentum + RSI)
        momentum_score = 1.0 if snap.price_change_20d_pct > 0 else -1.0 if snap.price_change_20d_pct < -5 else 0.0
        sma_score = 1.0 if snap.sma_cross > 1.02 else -1.0 if snap.sma_cross < 0.98 else 0.0
        rsi_score = 1.0 if snap.rsi > 60 else -1.0 if snap.rsi < 40 else 0.0
        snap.regime_score = (momentum_score + sma_score + rsi_score) / 3.0

    # ── Fundamentals from metrics ───────────────────────────────────────
    if metrics and len(metrics) > 0:
        m = metrics[0]  # most recent
        snap.pe_ratio = _safe_get(m, "pe_ratio")
        snap.pb_ratio = _safe_get(m, "price_to_book_value")
        snap.ps_ratio = _safe_get(m, "price_to_sales_ratio")
        snap.peg_ratio = _safe_get(m, "peg_ratio")
        snap.roe = _safe_get(m, "return_on_equity")
        snap.roic = _safe_get(m, "return_on_invested_capital")
        snap.debt_to_equity = _safe_get(m, "debt_to_equity")
        snap.current_ratio = _safe_get(m, "current_ratio")
        snap.operating_margin = _safe_get(m, "operating_margin")
        snap.gross_margin = _safe_get(m, "gross_margin")
        snap.net_margin = _safe_get(m, "net_margin")
        snap.ev_to_ebitda = _safe_get(m, "ev_to_ebitda")
        snap.fcf_yield = _safe_get(m, "free_cash_flow_yield")
        snap.earnings_growth = _safe_get(m, "earnings_growth")
        snap.revenue_growth = _safe_get(m, "revenue_growth")

    # ── Growth consistency from line items ──────────────────────────────
    if line_items and len(line_items) >= 2:
        revenues = [_safe_get(li, "revenue") for li in line_items if _safe_get(li, "revenue") > 0]
        if len(revenues) >= 3:
            growth_rates = [(revenues[i] / revenues[i + 1] - 1) for i in range(len(revenues) - 1) if revenues[i + 1] > 0]
            if growth_rates:
                positive = sum(1 for g in growth_rates if g > 0)
                snap.revenue_growth_consistency = positive / len(growth_rates)

        fcfs = [_safe_get(li, "free_cash_flow") for li in line_items if _safe_get(li, "free_cash_flow") != 0]
        if len(fcfs) >= 2 and fcfs[-1] != 0:
            snap.fcf_growth = (fcfs[0] / abs(fcfs[-1]) - 1) * 100

    # ── Intrinsic value gap (simple earnings-based) ─────────────────────
    if snap.pe_ratio > 0 and snap.earnings_growth > 0 and snap.price > 0:
        fair_pe = min(snap.earnings_growth * 100 * 1.5, 30)  # PEG-based fair P/E, capped
        snap.intrinsic_value_gap = (fair_pe - snap.pe_ratio) / snap.pe_ratio

    # ── Insider activity ────────────────────────────────────────────────
    if insider_trades:
        buys = sum(1 for t in insider_trades if getattr(t, "transaction_type", "") in ("P", "purchase", "P-Purchase"))
        sells = sum(1 for t in insider_trades if getattr(t, "transaction_type", "") in ("S", "sale", "S-Sale"))
        snap.insider_net_buys = buys - sells
        total = buys + sells
        snap.insider_buy_ratio = buys / total if total > 0 else 0.5

    # ── News sentiment ──────────────────────────────────────────────────
    if news:
        sentiments = [getattr(n, "sentiment", None) for n in news if getattr(n, "sentiment", None)]
        if sentiments:
            score_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            scores = [score_map.get(str(s).lower(), 0.0) for s in sentiments]
            snap.news_sentiment = sum(scores) / len(scores) if scores else 0.0

    # ── Market cap bucket ───────────────────────────────────────────────
    if market_cap and market_cap > 0:
        if market_cap < 3e8:
            snap.market_cap_bucket = 1
        elif market_cap < 2e9:
            snap.market_cap_bucket = 2
        elif market_cap < 10e9:
            snap.market_cap_bucket = 3
        elif market_cap < 200e9:
            snap.market_cap_bucket = 4
        else:
            snap.market_cap_bucket = 5

    # ── Fallback detection ──────────────────────────────────────────────
    if not prices or len(prices) < 2:
        snap.is_fallback = True

    return snap
