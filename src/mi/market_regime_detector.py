"""
Market Regime Detector
Identifies market conditions (trending/ranging/volatile/choppy) for adaptive trading

Based on strategies from top performing bots
"""

import logging
import numpy as np
from typing import Dict, List
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    CHOPPY = "choppy"


class MarketRegimeDetector:
    """
    Detect market regime for adaptive trading strategies
    """
    
    def __init__(self, config: Dict):
        """
        Initialize market regime detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        self.adx_strong_trend = config.get('ADX_STRONG_TREND', 40)
        self.adx_weak_trend = config.get('ADX_WEAK_TREND', 25)
        self.volatility_high_threshold = config.get('VOLATILITY_HIGH_THRESHOLD', 3.0)
        
        logger.info("📈 Market Regime Detector initialized")
    
    def detect_regime(
        self,
        prices: np.ndarray,
        adx: float = None,
        di_plus: float = None,
        di_minus: float = None,
        atr: float = None
    ) -> Dict:
        """
        Detect current market regime
        
        Args:
            prices: Array of recent prices
            adx: Average Directional Index
            di_plus: Positive Directional Indicator
            di_minus: Negative Directional Indicator
            atr: Average True Range
            
        Returns:
            Dict with regime information
        """
        if len(prices) < 20:
            return {
                'regime': MarketRegime.RANGING,
                'confidence': 0.5,
                'description': 'Insufficient data',
                'trending': False,
                'volatile': False
            }
        
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
        current_price = prices[-1]
        
        price_vs_sma20 = ((current_price - sma_20) / sma_20) * 100
        trend_strength = ((sma_20 - sma_50) / sma_50) * 100 if sma_50 > 0 else 0
        
        returns = np.diff(prices[-20:]) / prices[-20:-1]
        volatility = np.std(returns) * 100
        
        regime = MarketRegime.RANGING
        confidence = 0.5
        description = ""
        trending = False
        volatile = False
        
        if volatility > self.volatility_high_threshold:
            volatile = True
        
        if adx is not None:
            if adx > self.adx_strong_trend:
                trending = True
                if di_plus and di_minus:
                    if di_plus > di_minus:
                        regime = MarketRegime.STRONG_UPTREND
                        description = "Strong uptrend (high ADX)"
                    else:
                        regime = MarketRegime.STRONG_DOWNTREND
                        description = "Strong downtrend (high ADX)"
                else:
                    # Use price trend
                    if trend_strength > 2:
                        regime = MarketRegime.STRONG_UPTREND
                        description = "Strong uptrend"
                    elif trend_strength < -2:
                        regime = MarketRegime.STRONG_DOWNTREND
                        description = "Strong downtrend"
                confidence = 0.9
                
            elif adx > self.adx_weak_trend:
                trending = True
                if di_plus and di_minus:
                    if di_plus > di_minus:
                        regime = MarketRegime.WEAK_UPTREND
                        description = "Weak uptrend"
                    else:
                        regime = MarketRegime.WEAK_DOWNTREND
                        description = "Weak downtrend"
                else:
                    if trend_strength > 1:
                        regime = MarketRegime.WEAK_UPTREND
                        description = "Weak uptrend"
                    elif trend_strength < -1:
                        regime = MarketRegime.WEAK_DOWNTREND
                        description = "Weak downtrend"
                confidence = 0.7
                
            else:
                if volatile:
                    regime = MarketRegime.CHOPPY
                    description = "Choppy market (low ADX, high volatility)"
                else:
                    regime = MarketRegime.RANGING
                    description = "Ranging market (low ADX)"
                confidence = 0.8
        else:
            if abs(trend_strength) > 3:
                trending = True
                if trend_strength > 3:
                    regime = MarketRegime.STRONG_UPTREND
                    description = "Strong uptrend (price-based)"
                else:
                    regime = MarketRegime.STRONG_DOWNTREND
                    description = "Strong downtrend (price-based)"
                confidence = 0.7
            elif abs(trend_strength) > 1:
                trending = True
                if trend_strength > 1:
                    regime = MarketRegime.WEAK_UPTREND
                    description = "Weak uptrend (price-based)"
                else:
                    regime = MarketRegime.WEAK_DOWNTREND
                    description = "Weak downtrend (price-based)"
                confidence = 0.6
            else:
                if volatile:
                    regime = MarketRegime.CHOPPY
                    description = "Choppy market (high volatility)"
                else:
                    regime = MarketRegime.RANGING
                    description = "Ranging market (sideways)"
                confidence = 0.6
        
        if volatile and volatility > self.volatility_high_threshold * 1.5:
            regime = MarketRegime.HIGH_VOLATILITY
            description = f"High volatility market ({volatility:.2f}%)"
            confidence = 0.9
        
        result = {
            'regime': regime,
            'confidence': confidence,
            'description': description,
            'trending': trending,
            'volatile': volatile,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'adx': adx,
            'price_vs_sma20': price_vs_sma20
        }
        
        logger.debug(f"Regime: {regime.value} ({description}, conf: {confidence:.0%})")
        
        return result
    
    def get_regime_trading_advice(self, regime_info: Dict) -> Dict:
        """
        Get trading recommendations based on regime
        
        Args:
            regime_info: Output from detect_regime()
            
        Returns:
            Trading advice dict
        """
        regime = regime_info['regime']
        
        advice = {
            'regime': regime.value,
            'should_trade': True,
            'preferred_strategy': 'trend_following',
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'confidence_threshold_adj': 0.0
        }
        
        if regime == MarketRegime.STRONG_UPTREND:
            advice.update({
                'should_trade': True,
                'preferred_strategy': 'trend_following',
                'bias': 'LONG',
                'position_size_multiplier': 1.2,  # Increase size in strong trends
                'stop_loss_multiplier': 1.5,  # Wider stops
                'confidence_threshold_adj': -5  # Lower threshold (more trades)
            })
        
        elif regime == MarketRegime.STRONG_DOWNTREND:
            advice.update({
                'should_trade': True,
                'preferred_strategy': 'trend_following',
                'bias': 'SHORT',
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.5,
                'confidence_threshold_adj': -5
            })
        
        elif regime == MarketRegime.WEAK_UPTREND:
            advice.update({
                'should_trade': True,
                'preferred_strategy': 'trend_following',
                'bias': 'LONG',
                'position_size_multiplier': 0.8,  # Reduce size in weak trends
                'stop_loss_multiplier': 1.2,
                'confidence_threshold_adj': 0
            })
        
        elif regime == MarketRegime.WEAK_DOWNTREND:
            advice.update({
                'should_trade': True,
                'preferred_strategy': 'trend_following',
                'bias': 'SHORT',
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 1.2,
                'confidence_threshold_adj': 0
            })
        
        elif regime == MarketRegime.RANGING:
            advice.update({
                'should_trade': True,
                'preferred_strategy': 'mean_reversion',
                'bias': 'NEUTRAL',
                'position_size_multiplier': 0.7,  # Smaller positions in ranging
                'stop_loss_multiplier': 1.0,
                'confidence_threshold_adj': 5  # Higher threshold (fewer trades)
            })
        
        elif regime == MarketRegime.CHOPPY:
            advice.update({
                'should_trade': False,  # Avoid choppy markets
                'preferred_strategy': 'wait',
                'bias': 'NEUTRAL',
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 0.8,  # Tighter stops if trading
                'confidence_threshold_adj': 10  # Much higher threshold
            })
        
        elif regime == MarketRegime.HIGH_VOLATILITY:
            advice.update({
                'should_trade': True,
                'preferred_strategy': 'volatility_breakout',
                'bias': 'NEUTRAL',
                'position_size_multiplier': 0.6,  # Reduce size in high volatility
                'stop_loss_multiplier': 2.0,  # Much wider stops
                'confidence_threshold_adj': 5
            })
        
        return advice
