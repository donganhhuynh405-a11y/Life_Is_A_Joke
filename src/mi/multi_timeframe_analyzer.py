"""
Multi-Timeframe Analyzer
Analyzes multiple timeframes for better trade entry confirmation

Based on top performing bot strategies
"""

import logging
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """
    Analyze multiple timeframes for trend confirmation and better entries
    """
    
    def __init__(self, config: Dict):
        """Initialize multi-timeframe analyzer"""
        self.config = config
        self.timeframes = config.get('ANALYSIS_TIMEFRAMES', ['1h', '4h', '1d'])
        logger.info(f"📊 Multi-Timeframe Analyzer initialized: {self.timeframes}")
    
    def analyze_timeframes(
        self,
        symbol: str,
        data_by_timeframe: Dict[str, Dict]
    ) -> Dict:
        """
        Analyze multiple timeframes and provide consolidated view
        
        Args:
            symbol: Trading symbol
            data_by_timeframe: Dict of timeframe -> market data
            
        Returns:
            Consolidated analysis with trend alignment
        """
        analysis = {
            'symbol': symbol,
            'timeframes': {},
            'trend_alignment': 0.0,  # -1 to 1
            'all_aligned': False,
            'bullish_timeframes': 0,
            'bearish_timeframes': 0,
            'neutral_timeframes': 0,
            'recommendation': 'NEUTRAL'
        }
        
        for tf in self.timeframes:
            if tf not in data_by_timeframe:
                continue
            
            tf_data = data_by_timeframe[tf]
            prices = tf_data.get('prices', [])
            
            if len(prices) < 20:
                continue
            
            # Calculate trend
            sma_20 = np.mean(prices[-20:])
            current_price = prices[-1]
            trend_pct = ((current_price - sma_20) / sma_20) * 100
            
            # Classify
            if trend_pct > 1:
                trend = 'BULLISH'
                analysis['bullish_timeframes'] += 1
                score = 1
            elif trend_pct < -1:
                trend = 'BEARISH'
                analysis['bearish_timeframes'] += 1
                score = -1
            else:
                trend = 'NEUTRAL'
                analysis['neutral_timeframes'] += 1
                score = 0
            
            analysis['timeframes'][tf] = {
                'trend': trend,
                'trend_pct': trend_pct,
                'score': score
            }
        
        # Calculate alignment
        if analysis['timeframes']:
            total_score = sum(tf['score'] for tf in analysis['timeframes'].values())
            analysis['trend_alignment'] = total_score / len(analysis['timeframes'])
            
            # Check if all aligned
            if abs(analysis['trend_alignment']) > 0.8:
                analysis['all_aligned'] = True
                if analysis['trend_alignment'] > 0:
                    analysis['recommendation'] = 'STRONG_BUY'
                else:
                    analysis['recommendation'] = 'STRONG_SELL'
            elif analysis['trend_alignment'] > 0.5:
                analysis['recommendation'] = 'BUY'
            elif analysis['trend_alignment'] < -0.5:
                analysis['recommendation'] = 'SELL'
        
        logger.debug(f"MTF {symbol}: {analysis['recommendation']} "
                    f"(alignment: {analysis['trend_alignment']:.2f})")
        
        return analysis
    
    def should_enter_trade(self, mtf_analysis: Dict, direction: str) -> bool:
        """
        Determine if should enter trade based on MTF analysis
        
        Args:
            mtf_analysis: Output from analyze_timeframes()
            direction: 'LONG' or 'SHORT'
            
        Returns:
            True if MTF confirms trade direction
        """
        if direction == 'LONG':
            # Want bullish alignment for longs
            return mtf_analysis['trend_alignment'] > 0.3
        else:
            # Want bearish alignment for shorts
            return mtf_analysis['trend_alignment'] < -0.3
