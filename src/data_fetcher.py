"""
Market data fetcher using CCXT
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
import cachetools

logger = logging.getLogger(__name__)

class DataFetcher:
    """Fetches and caches market data from exchanges"""
    
    def __init__(self, config):
        self.config = config
        self.exchanges = {}
        self.cache = cachetools.TTLCache(maxsize=100, ttl=300)  # 5 min cache
        self.initialized = False
        
    async def initialize(self):
        """Initialize exchange connections"""
        if self.initialized:
            return
        
        logger.info("Initializing exchange connections...")
        
        exchange_configs = {
            'binance': {
                'apiKey': self.config.secrets.get('binance_api_key', ''),
                'secret': self.config.secrets.get('binance_api_secret', ''),
                'enableRateLimit': True,
                'options': {
                    'defaultType': self.config.exchanges.binance.type,
                    'adjustForTimeDifference': True,
                }
            }
        }
        
        # Create exchange instances
        for name, config in exchange_configs.items():
            if self.config.exchanges.get(name, {}).get('enabled', False):
                try:
                    exchange_class = getattr(ccxt, name)
                    
                    # Testnet for paper trading
                    if self.config.environment in ['paper', 'test']:
                        if name == 'binance':
                            config['urls'] = {
                                'api': {
                                    'public': 'https://testnet.binance.vision/api',
                                    'private': 'https://testnet.binance.vision/api',
                                }
                            }
                    
                    self.exchanges[name] = exchange_class(config)
                    
                    # Test connection
                    await self.exchanges[name].fetch_time()
                    logger.info(f"✓ {name} connection established")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize {name}: {e}")
        
        self.initialized = True
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                         limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a symbol"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Fetch data
                ohlcv = await exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Calculate additional features
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                
                # Cache result
                self.cache[cache_key] = df
                
                logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
                return df
                
            except Exception as e:
                logger.warning(f"{exchange_name} failed for {symbol}: {e}")
                continue
        
        logger.error(f"All exchanges failed for {symbol}")
        return None
    
    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch current ticker data"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                ticker = await exchange.fetch_ticker(symbol)
                return {
                    'symbol': symbol,
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'last': ticker['last'],
                    'volume': ticker['baseVolume'],
                    'timestamp': datetime.now()
                }
            except Exception as e:
                logger.warning(f"{exchange_name} ticker failed: {e}")
                continue
        return None
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Fetch order book"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                orderbook = await exchange.fetch_order_book(symbol, limit)
                return {
                    'symbol': symbol,
                    'bids': orderbook['bids'][:limit],
                    'asks': orderbook['asks'][:limit],
                    'timestamp': datetime.now()
                }
            except Exception as e:
                logger.warning(f"{exchange_name} orderbook failed: {e}")
                continue
        return None
    
    async def fetch_all_symbols(self) -> List[Dict]:
        """Fetch data for all configured symbols"""
        symbols = self.config.trading.symbols
        timeframe = self.config.trading.timeframes.primary
        
        tasks = []
        for symbol in symbols:
            tasks.append(self.fetch_ohlcv(symbol, timeframe))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        market_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
            elif result is not None:
                market_data[symbol] = result
        
        return market_data
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is available and has sufficient liquidity"""
        try:
            # Check ticker
            ticker = await self.fetch_ticker(symbol)
            if not ticker:
                return False
            
            # Check volume
            if ticker['volume'] < self.config.trading.filters.min_24h_volume:
                logger.info(f"{symbol} volume too low: {ticker['volume']}")
                return False
            
            # Check order book spread
            orderbook = await self.fetch_order_book(symbol, 5)
            if orderbook:
                best_bid = orderbook['bids'][0][0]
                best_ask = orderbook['asks'][0][0]
                spread = (best_ask - best_bid) / best_bid
                
                if spread > 0.001:  # More than 0.1% spread
                    logger.info(f"{symbol} spread too high: {spread:.4f}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Symbol validation failed for {symbol}: {e}")
            return False
    
    async def shutdown(self):
        """Close exchange connections"""
        logger.info("Closing exchange connections...")
        for name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                logger.info(f"✓ {name} connection closed")
            except Exception as e:
                logger.error(f"Error closing {name}: {e}")
        
        self.exchanges.clear()
        self.initialized = False
