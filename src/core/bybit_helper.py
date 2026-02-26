"""
Bybit Exchange Compatibility Helper
Provides Bybit-specific functionality and helpers
"""

import logging
from typing import Dict, List, Optional
from core.exchange_adapter import ExchangeAdapter


class BybitHelper:
    """
    Bybit-specific helper functions and compatibility layer
    
    Provides additional functionality specific to Bybit exchange
    """
    
    def __init__(self, exchange_adapter: ExchangeAdapter):
        """
        Initialize Bybit helper
        
        Args:
            exchange_adapter: ExchangeAdapter instance configured for Bybit
        """
        if exchange_adapter.exchange_id != 'bybit':
            raise ValueError("BybitHelper requires Bybit exchange adapter")
        
        self.exchange = exchange_adapter.exchange
        self.adapter = exchange_adapter
        self.logger = logging.getLogger(__name__)
    
    def get_account_type(self) -> str:
        """
        Get current account type (SPOT, CONTRACT, etc.)
        
        Returns:
            Account type string
        """
        try:
            # Bybit has different account types
            # This returns the type configured in options
            return self.exchange.options.get('defaultType', 'spot')
        except Exception as e:
            self.logger.error(f"Failed to get account type: {e}")
            return 'spot'
    
    def get_spot_balance(self, currency: Optional[str] = None) -> Dict:
        """
        Get spot wallet balance
        
        Args:
            currency: Specific currency (e.g., 'BTC', 'USDT') or None for all
            
        Returns:
            Balance dictionary
        """
        try:
            balance = self.exchange.fetch_balance({'type': 'spot'})
            
            if currency:
                return {
                    'currency': currency,
                    'free': balance['free'].get(currency, 0),
                    'used': balance['used'].get(currency, 0),
                    'total': balance['total'].get(currency, 0)
                }
            
            return balance
        except Exception as e:
            self.logger.error(f"Failed to get spot balance: {e}")
            raise
    
    def get_contract_balance(self, currency: Optional[str] = None) -> Dict:
        """
        Get derivatives/contract wallet balance
        
        Args:
            currency: Specific currency or None for all
            
        Returns:
            Balance dictionary
        """
        try:
            balance = self.exchange.fetch_balance({'type': 'contract'})
            
            if currency:
                return {
                    'currency': currency,
                    'free': balance['free'].get(currency, 0),
                    'used': balance['used'].get(currency, 0),
                    'total': balance['total'].get(currency, 0)
                }
            
            return balance
        except Exception as e:
            self.logger.error(f"Failed to get contract balance: {e}")
            raise
    
    def get_trading_fees(self, symbol: str) -> Dict:
        """
        Get trading fees for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with maker and taker fees
        """
        try:
            market = self.exchange.market(symbol)
            return {
                'symbol': symbol,
                'maker': market.get('maker', 0.001),  # Default 0.1%
                'taker': market.get('taker', 0.001),  # Default 0.1%
            }
        except Exception as e:
            self.logger.error(f"Failed to get trading fees: {e}")
            return {
                'symbol': symbol,
                'maker': 0.001,
                'taker': 0.001
            }
    
    def get_minimum_order_size(self, symbol: str) -> Dict:
        """
        Get minimum order size limits for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with min/max order limits
        """
        try:
            market = self.exchange.market(symbol)
            limits = market.get('limits', {})
            
            return {
                'symbol': symbol,
                'min_amount': limits.get('amount', {}).get('min', 0),
                'max_amount': limits.get('amount', {}).get('max', float('inf')),
                'min_cost': limits.get('cost', {}).get('min', 0),
                'max_cost': limits.get('cost', {}).get('max', float('inf')),
                'min_price': limits.get('price', {}).get('min', 0),
                'max_price': limits.get('price', {}).get('max', float('inf')),
            }
        except Exception as e:
            self.logger.error(f"Failed to get order limits: {e}")
            return {
                'symbol': symbol,
                'min_amount': 0,
                'max_amount': float('inf'),
                'min_cost': 0,
                'max_cost': float('inf'),
            }
    
    def validate_order_size(self, symbol: str, amount: float, price: float) -> tuple:
        """
        Validate if order size meets exchange requirements
        
        Args:
            symbol: Trading pair symbol
            amount: Order amount
            price: Order price
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            limits = self.get_minimum_order_size(symbol)
            
            # Check minimum amount
            if amount < limits['min_amount']:
                return False, f"Amount {amount} below minimum {limits['min_amount']}"
            
            # Check maximum amount
            if amount > limits['max_amount']:
                return False, f"Amount {amount} above maximum {limits['max_amount']}"
            
            # Check minimum cost (amount * price)
            cost = amount * price
            if cost < limits['min_cost']:
                return False, f"Order cost {cost} below minimum {limits['min_cost']}"
            
            # Check maximum cost
            if cost > limits['max_cost']:
                return False, f"Order cost {cost} above maximum {limits['max_cost']}"
            
            return True, "Order size valid"
            
        except Exception as e:
            self.logger.error(f"Failed to validate order size: {e}")
            return False, str(e)
    
    def get_available_pairs(self, base_currency: Optional[str] = None) -> List[str]:
        """
        Get list of available trading pairs
        
        Args:
            base_currency: Filter by base currency (e.g., 'BTC')
            
        Returns:
            List of trading pair symbols
        """
        try:
            markets = self.exchange.markets
            pairs = list(markets.keys())
            
            if base_currency:
                pairs = [p for p in pairs if p.startswith(base_currency + '/')]
            
            return sorted(pairs)
        except Exception as e:
            self.logger.error(f"Failed to get available pairs: {e}")
            return []
    
    def get_ticker_24h(self, symbol: str) -> Dict:
        """
        Get 24-hour ticker statistics
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with 24h statistics
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'last_price': ticker.get('last'),
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'high_24h': ticker.get('high'),
                'low_24h': ticker.get('low'),
                'volume_24h': ticker.get('baseVolume'),
                'quote_volume_24h': ticker.get('quoteVolume'),
                'change_24h': ticker.get('change'),
                'percentage_24h': ticker.get('percentage'),
            }
        except Exception as e:
            self.logger.error(f"Failed to get 24h ticker: {e}")
            raise
    
    def switch_to_spot(self):
        """Switch account type to SPOT trading"""
        try:
            self.exchange.options['defaultType'] = 'spot'
            self.logger.info("Switched to SPOT trading")
        except Exception as e:
            self.logger.error(f"Failed to switch to SPOT: {e}")
            raise
    
    def switch_to_contract(self):
        """Switch account type to CONTRACT/derivatives trading"""
        try:
            self.exchange.options['defaultType'] = 'contract'
            self.logger.info("Switched to CONTRACT trading")
        except Exception as e:
            self.logger.error(f"Failed to switch to CONTRACT: {e}")
            raise
    
    def is_testnet(self) -> bool:
        """
        Check if connected to testnet
        
        Returns:
            True if testnet, False if production
        """
        return self.exchange.options.get('sandbox', False)
    
    def get_server_time(self) -> int:
        """
        Get Bybit server time
        
        Returns:
            Server timestamp in milliseconds
        """
        try:
            return self.exchange.fetch_time()
        except Exception as e:
            self.logger.error(f"Failed to get server time: {e}")
            raise
    
    def create_spot_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """
        Create a spot market order (helper with validation)
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount
            
        Returns:
            Order response
        """
        try:
            # Ensure we're in spot mode
            original_type = self.exchange.options.get('defaultType')
            self.switch_to_spot()
            
            # Get current price for validation
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            
            # Validate order size
            is_valid, message = self.validate_order_size(symbol, amount, price)
            if not is_valid:
                raise ValueError(f"Invalid order size: {message}")
            
            # Create order
            order = self.exchange.create_market_order(symbol, side, amount)
            
            # Restore original type
            self.exchange.options['defaultType'] = original_type
            
            self.logger.info(f"Created spot market {side} order: {amount} {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to create spot market order: {e}")
            raise
    
    def create_spot_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        """
        Create a spot limit order (helper with validation)
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount
            price: Limit price
            
        Returns:
            Order response
        """
        try:
            # Ensure we're in spot mode
            original_type = self.exchange.options.get('defaultType')
            self.switch_to_spot()
            
            # Validate order size
            is_valid, message = self.validate_order_size(symbol, amount, price)
            if not is_valid:
                raise ValueError(f"Invalid order size: {message}")
            
            # Create order
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            
            # Restore original type
            self.exchange.options['defaultType'] = original_type
            
            self.logger.info(f"Created spot limit {side} order: {amount} {symbol} @ {price}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to create spot limit order: {e}")
            raise
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all open orders
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of open orders
        """
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol
            
        Returns:
            Cancellation response
        """
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Cancelled order {order_id} for {symbol}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            raise
    
    def cancel_all_orders(self, symbol: Optional[str] = None):
        """
        Cancel all open orders
        
        Args:
            symbol: Cancel only for this symbol, or None for all
        """
        try:
            open_orders = self.get_open_orders(symbol)
            
            for order in open_orders:
                try:
                    self.cancel_order(order['id'], order['symbol'])
                except Exception as e:
                    self.logger.error(f"Failed to cancel order {order['id']}: {e}")
            
            self.logger.info(f"Cancelled {len(open_orders)} orders")
            
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {e}")
            raise
