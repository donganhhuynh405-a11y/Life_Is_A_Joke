"""
Simple Trend Following Strategy
A basic trend-following strategy using moving averages
"""

from typing import List, Dict
from strategies.base_strategy import BaseStrategy


class SimpleTrendStrategy(BaseStrategy):
    """
    Simple trend following strategy
    
    Uses short and long moving averages to detect trends:
    - BUY when short MA crosses above long MA (bullish)
    - SELL when short MA crosses below long MA (bearish)
    """
    
    def __init__(self, config, client, database, risk_manager):
        """Initialize simple trend strategy"""
        super().__init__(config, client, database, risk_manager)
        self.name = "SimpleTrend"
        self.short_period = 10
        self.long_period = 30
        self.last_signal = {}
    
    def analyze(self) -> List[Dict]:
        """
        Analyze market and generate signals
        
        Returns:
            List of trading signals
        """
        signals = []
        symbol = self.config.default_symbol
        
        try:
            # Get klines data (1 hour candles)
            klines = self.get_klines(symbol, interval='1h', limit=self.long_period + 10)
            
            if len(klines) < self.long_period:
                self.logger.warning(f"Not enough data for analysis: {len(klines)} candles")
                return signals
            
            # Calculate moving averages
            closes = [float(k[4]) for k in klines]  # Close prices
            
            short_ma = sum(closes[-self.short_period:]) / self.short_period
            long_ma = sum(closes[-self.long_period:]) / self.long_period
            
            # Previous MAs for crossover detection
            prev_short_ma = sum(closes[-(self.short_period+1):-1]) / self.short_period
            prev_long_ma = sum(closes[-(self.long_period+1):-1]) / self.long_period
            
            current_price = self.get_current_price(symbol)
            
            self.logger.debug(f"{symbol} - Price: {current_price}, Short MA: {short_ma:.2f}, Long MA: {long_ma:.2f}")
            
            # Check for open positions
            open_positions = self.db.get_open_positions()
            has_position = any(p['symbol'] == symbol for p in open_positions)
            
            # Bullish crossover - BUY signal
            if not has_position and prev_short_ma <= prev_long_ma and short_ma > long_ma:
                self.logger.info(f"Bullish crossover detected for {symbol}")
                signals.append({
                    'action': 'BUY',
                    'symbol': symbol,
                    'price': current_price,
                    'reason': f'MA crossover (Short: {short_ma:.2f} > Long: {long_ma:.2f})'
                })
                self.last_signal[symbol] = 'BUY'
            
            # Bearish crossover - SELL/CLOSE signal
            elif has_position and prev_short_ma >= prev_long_ma and short_ma < long_ma:
                self.logger.info(f"Bearish crossover detected for {symbol}")
                
                # Find the position to close
                position = next((p for p in open_positions if p['symbol'] == symbol), None)
                if position:
                    signals.append({
                        'action': 'CLOSE',
                        'symbol': symbol,
                        'price': current_price,
                        'position_id': position['id'],
                        'reason': f'MA crossover (Short: {short_ma:.2f} < Long: {long_ma:.2f})'
                    })
                    self.last_signal[symbol] = 'SELL'
            
            # Check stop loss and take profit for open positions
            for position in open_positions:
                if position['symbol'] == symbol:
                    self._check_exit_conditions(position, current_price, signals)
            
        except Exception as e:
            self.logger.error(f"Error in strategy analysis: {str(e)}", exc_info=True)
        
        return signals
    
    def _check_exit_conditions(self, position: Dict, current_price: float, signals: List[Dict]):
        """Check if position should be closed based on stop loss or take profit"""
        symbol = position['symbol']
        
        # Check stop loss
        if position['stop_loss'] and current_price <= position['stop_loss']:
            self.logger.info(f"Stop loss triggered for {symbol} at {current_price}")
            signals.append({
                'action': 'CLOSE',
                'symbol': symbol,
                'price': current_price,
                'position_id': position['id'],
                'reason': f'Stop loss hit ({current_price} <= {position["stop_loss"]})'
            })
        
        # Check take profit
        elif position['take_profit'] and current_price >= position['take_profit']:
            self.logger.info(f"Take profit triggered for {symbol} at {current_price}")
            signals.append({
                'action': 'CLOSE',
                'symbol': symbol,
                'price': current_price,
                'position_id': position['id'],
                'reason': f'Take profit hit ({current_price} >= {position["take_profit"]})'
            })
