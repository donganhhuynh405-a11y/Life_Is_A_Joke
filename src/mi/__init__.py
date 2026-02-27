"""
Machine Learning module for trading bot
Provides trade analysis, pattern detection, and performance optimization
"""

from .trade_analyzer import TradeAnalyzer
from .performance_analyzer import PerformanceAnalyzer
from .signal_scorer import SignalScorer
from .ai_commentary import AICommentaryGenerator, get_commentary_generator
from .adaptive_tactics import AdaptiveTacticsManager

__all__ = [
    'TradeAnalyzer',
    'PerformanceAnalyzer',
    'SignalScorer',
    'AICommentaryGenerator',
    'get_commentary_generator',
    'AdaptiveTacticsManager'
]
