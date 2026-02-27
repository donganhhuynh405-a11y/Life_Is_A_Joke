"""Arbitrage strategies package."""

from .triangular import TriangularArbitrage
from .cross_exchange import CrossExchangeArbitrage
from .funding_rate import FundingRateArbitrage
from .detector import ArbitrageDetector
from .executor import ArbitrageExecutor

__all__ = [
    "TriangularArbitrage",
    "CrossExchangeArbitrage",
    "FundingRateArbitrage",
    "ArbitrageDetector",
    "ArbitrageExecutor",
]
