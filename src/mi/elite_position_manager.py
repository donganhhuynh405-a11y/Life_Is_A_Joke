"""
Elite Position Manager
Advanced stop-loss and take-profit management

Based on strategies from top performing trading bots
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ElitePositionManager:
    """
    Advanced position management with trailing stops, partial profits, etc.
    """

    def __init__(self, config: Dict):
        """Initialize elite position manager"""
        self.config = config

        # Stop-loss configuration
        self.use_trailing_stop = config.get('USE_TRAILING_STOP', True)
        self.trailing_stop_activation_pct = config.get('TRAILING_STOP_ACTIVATION_PCT', 2.0)
        self.trailing_stop_distance_pct = config.get('TRAILING_STOP_DISTANCE_PCT', 1.0)

        # Take-profit configuration
        self.use_partial_tp = config.get('USE_PARTIAL_TP', True)

        # Parse partial TP levels from config
        partial_tp_str = config.get('PARTIAL_TP_LEVELS', '1.5:0.33,3.0:0.33')
        self.partial_tp_levels = []
        if isinstance(partial_tp_str, str):
            try:
                for level_str in partial_tp_str.split(','):
                    pct, close_pct = level_str.split(':')
                    self.partial_tp_levels.append({
                        'pct': float(pct),
                        'close_pct': float(close_pct)
                    })
            except Exception as e:
                logger.warning(f"Failed to parse PARTIAL_TP_LEVELS: {e}, using defaults")
                self.partial_tp_levels = [
                    {'pct': 1.5, 'close_pct': 0.33},
                    {'pct': 3.0, 'close_pct': 0.33},
                ]
        else:
            self.partial_tp_levels = config.get('PARTIAL_TP_LEVELS', [
                {'pct': 1.5, 'close_pct': 0.33},
                {'pct': 3.0, 'close_pct': 0.33},
            ])

        # Breakeven stop
        self.move_to_breakeven_pct = config.get('MOVE_TO_BREAKEVEN_PCT', 1.0)

        logger.info("🎯 Elite Position Manager initialized")
        logger.info(f"  Trailing Stop: {self.use_trailing_stop}")
        logger.info(f"  Partial TP: {self.use_partial_tp} (Levels: {self.partial_tp_levels})")

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        use_atr_stop: bool = True
    ) -> float:
        """
        Calculate optimal stop-loss price

        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: 'LONG' or 'SHORT'
            use_atr_stop: Use ATR-based stop

        Returns:
            Stop-loss price
        """
        if use_atr_stop and atr > 0:
            # ATR-based stop (2x ATR)
            stop_distance = 2 * atr
        else:
            # Fixed percentage stop (2%)
            stop_distance = entry_price * 0.02

        if direction == 'LONG':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        return stop_loss

    def update_trailing_stop(
        self,
        position: Dict,
        current_price: float
    ) -> Optional[float]:
        """
        Update trailing stop if price moved favorably

        Args:
            position: Position dict with entry_price, direction, current_stop
            current_price: Current market price

        Returns:
            New stop price or None if no update
        """
        if not self.use_trailing_stop:
            return None

        entry_price = position['entry_price']
        direction = position['direction']
        current_stop = position.get('stop_loss', 0)

        # Calculate profit percentage
        if direction == 'LONG':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100

        # Activate trailing stop if profit threshold reached
        if profit_pct < self.trailing_stop_activation_pct:
            return None

        # Calculate new trailing stop
        trailing_distance = current_price * (self.trailing_stop_distance_pct / 100)

        if direction == 'LONG':
            new_stop = current_price - trailing_distance
            # Only move stop up, never down
            if new_stop > current_stop:
                logger.info(f"📈 Trailing stop updated: ${current_stop:.4f} -> ${new_stop:.4f}")
                return new_stop
        else:
            new_stop = current_price + trailing_distance
            # Only move stop down, never up
            if current_stop == 0 or new_stop < current_stop:
                logger.info(f"📉 Trailing stop updated: ${current_stop:.4f} -> ${new_stop:.4f}")
                return new_stop

        return None

    def check_partial_take_profit(
        self,
        position: Dict,
        current_price: float
    ) -> Optional[Dict]:
        """
        Check if should take partial profits

        Args:
            position: Position information
            current_price: Current price

        Returns:
            Dict with partial close info or None
        """
        if not self.use_partial_tp:
            return None

        entry_price = position['entry_price']
        direction = position['direction']
        taken_levels = position.get('partial_tp_taken', [])

        # Calculate profit percentage
        if direction == 'LONG':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100

        # Check each TP level
        for level in self.partial_tp_levels:
            level_pct = level['pct']
            close_pct = level['close_pct']

            if profit_pct >= level_pct and level_pct not in taken_levels:
                logger.info(f"💰 Partial TP triggered at {level_pct}% profit, "
                            f"closing {close_pct * 100}% of position")
                return {
                    'level_pct': level_pct,
                    'close_percentage': close_pct,
                    'reason': f'Partial TP at {level_pct}% profit'
                }

        return None

    def should_move_to_breakeven(
        self,
        position: Dict,
        current_price: float
    ) -> bool:
        """
        Check if should move stop to breakeven

        Args:
            position: Position information
            current_price: Current price

        Returns:
            True if should move to breakeven
        """
        entry_price = position['entry_price']
        direction = position['direction']
        current_stop = position.get('stop_loss', 0)

        # Calculate profit percentage
        if direction == 'LONG':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            at_breakeven = current_stop >= entry_price
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
            at_breakeven = current_stop <= entry_price

        # Move to breakeven if profit threshold reached and not already there
        if profit_pct >= self.move_to_breakeven_pct and not at_breakeven:
            logger.info(f"🎯 Moving stop to breakeven (profit: {profit_pct:.2f}%)")
            return True

        return False
