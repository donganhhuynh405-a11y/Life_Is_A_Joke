"""
Base Strategy Class
Abstract base class for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import logging


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""

    def __init__(self, config, client, database, risk_manager):
        """
        Initialize strategy

        Args:
            config: Configuration object
            client: Binance client
            database: Database instance
            risk_manager: Risk manager instance
        """
        self.config = config
        self.client = client
        self.db = database
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enabled = True
        self.name = self.__class__.__name__

    @abstractmethod
    def analyze(self) -> List[Dict]:
        """
        Analyze market and generate trading signals

        Returns:
            List of signal dictionaries containing action, symbol, price, etc.
        """

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])

    def get_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> List:
        """Get historical candlestick data"""
        return self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
