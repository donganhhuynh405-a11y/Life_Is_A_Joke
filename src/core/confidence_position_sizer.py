"""
Confidence-Based Position Sizer
Dynamically adjusts position size based on signal confidence and market conditions
"""

import logging
from typing import Tuple


class ConfidencePositionSizer:
    """
    Calculate position size based on multiple confidence factors:
    1. Signal confidence score (from technical indicators)
    2. Indicator agreement (multiple indicators confirming the same signal)
    3. Trend strength
    4. Market volatility
    5. Historical success rate in similar conditions
    """
    
    def __init__(self, config):
        """
        Initialize confidence-based position sizer
        
        Args:
            config: Configuration object with min/max position sizes
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get position size bounds from config
        self.min_position_pct = getattr(config, 'min_position_size_pct', 0.5)  # % of balance
        self.max_position_pct = getattr(config, 'max_position_size_pct', 5.0)  # % of balance
        
        # Confidence thresholds
        self.min_confidence = 50  # Minimum confidence to trade
        self.max_confidence = 100  # Perfect confidence
        
        self.logger.info(
            f"Confidence Position Sizer initialized: "
            f"min={self.min_position_pct}%, max={self.max_position_pct}%"
        )
    
    def calculate_position_size(
        self,
        balance: float,
        price: float,
        confidence_score: float,
        trend_strength: float = None,
        volatility: float = None
    ) -> Tuple[float, float]:
        """
        Calculate position size based on confidence and market conditions
        
        Args:
            balance: Available balance (USDT)
            price: Current price of the asset
            confidence_score: Signal confidence (0-100)
            trend_strength: Trend strength factor (0-1), optional
            volatility: Market volatility factor (0-1), optional
        
        Returns:
            Tuple of (quantity in base currency, position size in USDT)
        """
        # Validate confidence score
        if confidence_score < self.min_confidence:
            self.logger.warning(
                f"Confidence {confidence_score} below minimum {self.min_confidence}, "
                f"using minimum position size"
            )
            confidence_score = self.min_confidence
        
        confidence_score = min(confidence_score, self.max_confidence)
        
        # Calculate base position percentage from confidence
        # Linear scaling from min_position_pct to max_position_pct
        # confidence 50 -> min_position_pct
        # confidence 100 -> max_position_pct
        confidence_factor = (confidence_score - self.min_confidence) / (
            self.max_confidence - self.min_confidence
        )
        
        position_pct = (
            self.min_position_pct + 
            confidence_factor * (self.max_position_pct - self.min_position_pct)
        )
        
        # Adjust for trend strength if provided
        if trend_strength is not None:
            trend_strength = max(0.0, min(1.0, trend_strength))
            # Increase position in strong trends
            trend_adjustment = 1.0 + (trend_strength * 0.3)  # Up to 30% boost
            position_pct *= trend_adjustment
            self.logger.debug(f"Trend adjustment: x{trend_adjustment:.2f}")
        
        # Adjust for volatility if provided
        if volatility is not None:
            volatility = max(0.0, min(1.0, volatility))
            # Reduce position in high volatility
            volatility_adjustment = 1.0 - (volatility * 0.3)  # Up to 30% reduction
            position_pct *= volatility_adjustment
            self.logger.debug(f"Volatility adjustment: x{volatility_adjustment:.2f}")
        
        # Cap position percentage
        position_pct = max(self.min_position_pct, min(self.max_position_pct, position_pct))
        
        # Calculate position size in USDT
        position_size_usdt = (balance * position_pct) / 100.0
        
        # Calculate quantity in base currency
        quantity = position_size_usdt / price
        
        self.logger.info(
            f"Position sizing: confidence={confidence_score:.1f}, "
            f"position_pct={position_pct:.2f}%, "
            f"size=${position_size_usdt:.2f} ({quantity:.8f} units)"
        )
        
        return quantity, position_size_usdt
    
    def get_position_info(
        self,
        confidence_score: float,
        balance: float
    ) -> dict:
        """
        Get position sizing information for display/logging
        
        Args:
            confidence_score: Signal confidence (0-100)
            balance: Available balance
        
        Returns:
            Dictionary with position sizing details
        """
        confidence_score = max(self.min_confidence, min(self.max_confidence, confidence_score))
        
        confidence_factor = (confidence_score - self.min_confidence) / (
            self.max_confidence - self.min_confidence
        )
        
        position_pct = (
            self.min_position_pct + 
            confidence_factor * (self.max_position_pct - self.min_position_pct)
        )
        
        return {
            'confidence': confidence_score,
            'position_percentage': position_pct,
            'position_size_range': f"{self.min_position_pct}% - {self.max_position_pct}%",
            'estimated_position_usd': (balance * position_pct) / 100.0
        }
