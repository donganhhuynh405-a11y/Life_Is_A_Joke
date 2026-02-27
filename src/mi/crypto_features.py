"""
Advanced Feature Engineering for Crypto Trading
Crypto-specific features that give edge over traditional models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OnChainMetrics:
    """On-chain data metrics"""
    whale_movements: float  # Large wallet activity
    exchange_inflows: float  # BTC moving to exchanges (bearish)
    exchange_outflows: float  # BTC leaving exchanges (bullish)
    active_addresses: float  # Network activity
    transaction_count: float  # Usage metric
    gas_fees: float  # Network congestion
    stablecoin_supply: float  # Dry powder indicator
    long_term_holder_supply: float  # HODLer metric


@dataclass
class MarketMicrostructure:
    """Market microstructure features"""
    order_book_imbalance: float  # Buy vs sell pressure
    bid_ask_spread: float  # Liquidity indicator
    depth_imbalance: float  # Cumulative depth at levels
    trade_flow_imbalance: float  # Aggressive buy vs sell
    volume_profile: Dict[float, float]  # Price-volume distribution
    liquidity_score: float  # Overall liquidity
    market_impact: float  # Expected price impact per unit


@dataclass
class CrossExchangeData:
    """Cross-exchange arbitrage signals"""
    price_differences: Dict[str, float]  # Price gaps between exchanges
    volume_differences: Dict[str, float]  # Volume distribution
    funding_rates: Dict[str, float]  # Perpetual contract funding
    basis_spreads: Dict[str, float]  # Spot vs futures
    triangular_arbitrage: float  # Multi-hop arbitrage opportunities


class AdvancedFeatureEngineer:
    """
    Creates advanced features for crypto trading
    Goes far beyond basic OHLCV
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.feature_history = {}

    def extract_all_features(
        self,
        ohlcv: pd.DataFrame,
        onchain: Optional[OnChainMetrics] = None,
        orderbook: Optional[Dict] = None,
        cross_exchange: Optional[CrossExchangeData] = None
    ) -> pd.DataFrame:
        """
        Extract all available features

        Args:
            ohlcv: Basic price data
            onchain: On-chain metrics
            orderbook: Order book snapshot
            cross_exchange: Cross-exchange data

        Returns:
            DataFrame with all features
        """
        features = pd.DataFrame(index=ohlcv.index)

        # Basic price features
        features = pd.concat([features, self.price_features(ohlcv)], axis=1)

        # Volume features
        features = pd.concat([features, self.volume_features(ohlcv)], axis=1)

        # Volatility features
        features = pd.concat([features, self.volatility_features(ohlcv)], axis=1)

        # Momentum features
        features = pd.concat([features, self.momentum_features(ohlcv)], axis=1)

        # Pattern recognition
        features = pd.concat([features, self.pattern_features(ohlcv)], axis=1)

        # Market regime
        features = pd.concat([features, self.regime_features(ohlcv)], axis=1)

        # Time-based features
        features = pd.concat([features, self.time_features(ohlcv)], axis=1)

        # On-chain features (if available)
        if onchain:
            features = pd.concat([features, self.onchain_features(onchain)], axis=1)

        # Microstructure features (if orderbook available)
        if orderbook:
            features = pd.concat([features, self.microstructure_features(orderbook)], axis=1)

        # Cross-exchange features
        if cross_exchange:
            features = pd.concat([features, self.cross_exchange_features(cross_exchange)], axis=1)

        logger.info(f"Extracted {len(features.columns)} features")
        return features

    def price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced price-based features"""
        features = pd.DataFrame(index=df.index)

        # Returns at multiple horizons
        for period in [1, 5, 15, 30, 60]:
            features[f'return_{period}'] = df['close'].pct_change(period)
            features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        # Price levels
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']

        # Distance from moving averages
        for period in [7, 25, 99, 200]:
            ma = df['close'].rolling(period).mean()
            features[f'distance_ma{period}'] = (df['close'] - ma) / ma

        # Support and resistance
        features['distance_52w_high'] = (df['close'] - df['close'].rolling(252).max()) / df['close']
        features['distance_52w_low'] = (df['close'] - df['close'].rolling(252).min()) / df['close']

        # Fibonacci levels
        high = df['high'].rolling(100).max()
        low = df['low'].rolling(100).min()
        diff = high - low

        for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
            fib_level = low + diff * level
            features[f'fib_{int(level * 1000)}'] = (df['close'] - fib_level) / df['close']

        return features

    def volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced volume analysis"""
        features = pd.DataFrame(index=df.index)

        # Volume ratios
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_std_ratio'] = df['volume'] / df['volume'].rolling(20).std()

        # On-Balance Volume (OBV)
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        features['obv'] = obv
        features['obv_ma_ratio'] = obv / obv.rolling(20).mean()

        # Volume-weighted prices
        features['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        features['vwap_distance'] = (df['close'] - features['vwap']) / features['vwap']

        # Accumulation/Distribution
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        features['acc_dist'] = (clv * df['volume']).cumsum()

        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(
            typical_price > typical_price.shift(1),
            0).rolling(14).sum()
        negative_flow = money_flow.where(
            typical_price < typical_price.shift(1),
            0).rolling(14).sum()

        features['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1)))

        # Volume profile (simplified)
        features['volume_concentration'] = df['volume'].rolling(
            20).std() / df['volume'].rolling(20).mean()

        return features

    def volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive volatility features"""
        features = pd.DataFrame(index=df.index)

        # Historical volatility at multiple periods
        for period in [5, 10, 20, 50]:
            returns = df['close'].pct_change()
            features[f'vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)

        # Parkinson volatility (uses high-low)
        hl_ratio = np.log(df['high'] / df['low'])
        features['parkinson_vol'] = (hl_ratio ** 2 / (4 * np.log(2))).rolling(20).mean()

        # Garman-Klass volatility (more efficient)
        features['gk_vol'] = np.sqrt(
            0.5 * (np.log(df['high'] / df['low']) ** 2) -
            (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)
        ).rolling(20).mean()

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean()
        features['atr_pct'] = features['atr'] / df['close']

        # Volatility regimes
        vol = df['close'].pct_change().rolling(20).std()
        features['vol_regime'] = pd.qcut(
            vol.rank(
                method='first'),
            q=5,
            labels=False,
            duplicates='drop')

        # Volatility of volatility
        features['vol_of_vol'] = vol.rolling(20).std()

        return features

    def momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced momentum indicators"""
        features = pd.DataFrame(index=df.index)

        # RSI at multiple periods
        for period in [7, 14, 28]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / loss.replace(0, 1)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD variations
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()

            features[f'macd_{fast}_{slow}'] = macd
            features[f'macd_signal_{fast}_{slow}'] = signal_line
            features[f'macd_hist_{fast}_{slow}'] = macd - signal_line

        # Stochastic oscillator
        for period in [14, 28]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            features[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = df['close'].pct_change(period) * 100

        # Commodity Channel Index (CCI)
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        features['cci'] = (tp - sma) / (0.015 * mad)

        # Williams %R
        high_14 = df['high'].rolling(14).max()
        low_14 = df['low'].rolling(14).min()
        features['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

        # Awesome Oscillator
        median_price = (df['high'] + df['low']) / 2
        features['awesome_oscillator'] = median_price.rolling(
            5).mean() - median_price.rolling(34).mean()

        return features

    def pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick pattern recognition"""
        features = pd.DataFrame(index=df.index)

        # Body and shadow sizes
        body = abs(df['close'] - df['open'])
        full_range = df['high'] - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']

        features['body_ratio'] = body / full_range.replace(0, 1)
        features['upper_shadow_ratio'] = upper_shadow / full_range.replace(0, 1)
        features['lower_shadow_ratio'] = lower_shadow / full_range.replace(0, 1)

        # Doji detection
        features['is_doji'] = (body / full_range < 0.1).astype(int)

        # Hammer/Hanging man
        is_hammer = (
            (lower_shadow > 2 * body) &
            (upper_shadow < 0.3 * body)
        )
        features['is_hammer'] = is_hammer.astype(int)

        # Shooting star
        is_shooting_star = (
            (upper_shadow > 2 * body) &
            (lower_shadow < 0.3 * body)
        )
        features['is_shooting_star'] = is_shooting_star.astype(int)

        # Engulfing patterns
        is_bullish_engulfing = (
            (df['close'] > df['open']) &  # Current bullish
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous bearish
            (df['open'] < df['close'].shift(1)) &  # Current opens below previous close
            (df['close'] > df['open'].shift(1))  # Current closes above previous open
        )
        features['is_bullish_engulfing'] = is_bullish_engulfing.astype(int)

        is_bearish_engulfing = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        )
        features['is_bearish_engulfing'] = is_bearish_engulfing.astype(int)

        # Consecutive candles
        is_bullish = (df['close'] > df['open']).astype(int)
        is_bearish = (df['close'] < df['open']).astype(int)

        features['consecutive_bullish'] = is_bullish.groupby(
            (is_bullish != is_bullish.shift()).cumsum()).cumsum()
        features['consecutive_bearish'] = is_bearish.groupby(
            (is_bearish != is_bearish.shift()).cumsum()).cumsum()

        return features

    def regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market regime classification"""
        features = pd.DataFrame(index=df.index)

        # Trend strength (ADX)
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)

        plus_di = 100 * plus_dm.rolling(14).mean() / tr.rolling(14).mean()
        minus_di = 100 * minus_dm.rolling(14).mean() / tr.rolling(14).mean()

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        features['adx'] = dx.rolling(14).mean()

        # Trend direction
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        features['trend_direction'] = np.sign(ema_20 - ema_50)

        # Ranging vs trending
        features['is_ranging'] = (features['adx'] < 25).astype(int)
        features['is_trending'] = (features['adx'] > 40).astype(int)

        # Bull vs bear market
        sma_200 = df['close'].rolling(200).mean()
        features['bull_market'] = (df['close'] > sma_200).astype(int)

        # Volatility regime
        vol = df['close'].pct_change().rolling(20).std()
        vol_ma = vol.rolling(50).mean()
        features['high_volatility'] = (vol > vol_ma * 1.5).astype(int)
        features['low_volatility'] = (vol < vol_ma * 0.7).astype(int)

        return features

    def time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features (seasonality)"""
        features = pd.DataFrame(index=df.index)

        # Hour of day (for intraday)
        if hasattr(df.index, 'hour'):
            features['hour'] = df.index.hour
            features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

        # Day of week
        if hasattr(df.index, 'dayofweek'):
            features['day_of_week'] = df.index.dayofweek
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # Day of month
        if hasattr(df.index, 'day'):
            features['day_of_month'] = df.index.day
            features['day_sin'] = np.sin(2 * np.pi * df.index.day / 31)
            features['day_cos'] = np.cos(2 * np.pi * df.index.day / 31)

        # Month
        if hasattr(df.index, 'month'):
            features['month'] = df.index.month
            features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

        # Quarter
        if hasattr(df.index, 'quarter'):
            features['quarter'] = df.index.quarter

        return features

    def onchain_features(self, onchain: OnChainMetrics) -> pd.DataFrame:
        """On-chain metrics to features"""
        features = pd.DataFrame({
            'whale_activity': [onchain.whale_movements],
            'exchange_netflow': [onchain.exchange_outflows - onchain.exchange_inflows],
            'network_activity': [onchain.active_addresses],
            'transaction_volume': [onchain.transaction_count],
            'gas_pressure': [onchain.gas_fees],
            'stablecoin_supply': [onchain.stablecoin_supply],
            'hodl_ratio': [onchain.long_term_holder_supply]
        })
        return features

    def microstructure_features(self, orderbook: Dict) -> pd.DataFrame:
        """Order book microstructure features"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])

        if not bids or not asks:
            return pd.DataFrame()

        # Calculate imbalance at different levels
        def calculate_imbalance(depth=10):
            bid_vol = sum(b[1] for b in bids[:depth])
            ask_vol = sum(a[1] for a in asks[:depth])
            return (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

        features = pd.DataFrame({
            'ob_imbalance_5': [calculate_imbalance(5)],
            'ob_imbalance_10': [calculate_imbalance(10)],
            'ob_imbalance_20': [calculate_imbalance(20)],
            'bid_ask_spread': [(asks[0][0] - bids[0][0]) / bids[0][0]],
            'depth_ratio': [sum(b[1] for b in bids[:10]) / sum(a[1] for a in asks[:10])]
        })

        return features

    def cross_exchange_features(self, cross_data: CrossExchangeData) -> pd.DataFrame:
        """Cross-exchange arbitrage signals"""
        features = pd.DataFrame({
            'avg_price_diff': [np.mean(list(cross_data.price_differences.values()))],
            'max_price_diff': [np.max(list(cross_data.price_differences.values()))],
            'avg_funding_rate': [np.mean(list(cross_data.funding_rates.values()))],
            'basis_spread': [np.mean(list(cross_data.basis_spreads.values()))],
            'triangular_arb': [cross_data.triangular_arbitrage]
        })

        return features


def engineer_target_variables(df: pd.DataFrame, horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
    """
    Create multiple target variables for multi-horizon prediction
    """
    targets = pd.DataFrame(index=df.index)

    for horizon in horizons:
        # Price direction (classification)
        future_return = df['close'].shift(-horizon) / df['close'] - 1
        targets[f'direction_{horizon}'] = np.sign(future_return)

        # Return magnitude (regression)
        targets[f'return_{horizon}'] = future_return

        # Volatility (realized future volatility)
        future_returns = df['close'].pct_change().shift(-horizon).rolling(horizon).std()
        targets[f'volatility_{horizon}'] = future_returns

        # Optimal position size (based on actual outcome)
        sharpe = future_return / future_returns.replace(0, 1)
        targets[f'optimal_size_{horizon}'] = np.clip(sharpe / 2, 0, 1)  # Kelly-like sizing

    return targets
