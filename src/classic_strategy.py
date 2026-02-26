"""
Классическая стратегия торговли на основе индикаторов
Аналог стратегий из OctoBot с защитой от флета
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ClassicTradingStrategy:
    """
    Гибридная стратегия: MACD + RSI + трендовая фильтрация
    Правила:
    - MACD выше сигнальной линии + гистограмма растет = сигнал на покупку
    - RSI < 30 (перепроданность) для подтверждения входа
    - EMA(20) > EMA(50) для трендовой фильтрации (только лонг в аптренде)
    - Фильтр флета: ATR(14) должен быть > 1% от цены
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicators_cache = {}
        
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Экспоненциальная скользящая средняя"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD, сигнальная линия и гистограмма"""
        ema12 = self.calculate_ema(prices, 12)
        ema26 = self.calculate_ema(prices, 26)
        macd_line = ema12 - ema26
        signal_line = self.calculate_ema(macd_line, 9)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Индекс относительной силы"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range - волатильность"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2.0):
        """Полосы Боллинджера"""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = sma + (rolling_std * std)
        lower_band = sma - (rolling_std * std)
        return upper_band, sma, lower_band
    
    def analyze_market(self, df: pd.DataFrame) -> Dict:
        """
        Основной анализ рынка
        
        Возвращает:
        {
            'signal': 'BUY'/'SELL'/'HOLD',
            'confidence': 0.0-1.0,
            'indicators': {значения индикаторов},
            'conditions': {условия для логирования}
        }
        """
        if len(df) < 50:  # Минимум данных
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'indicators': {},
                'conditions': {'error': 'Insufficient data'}
            }
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Рассчитываем индикаторы
        ema20 = self.calculate_ema(close, 20).iloc[-1]
        ema50 = self.calculate_ema(close, 50).iloc[-1]
        macd_line, signal_line, histogram = self.calculate_macd(close)
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        macd_histogram = histogram.iloc[-1]
        macd_prev_histogram = histogram.iloc[-2]
        
        rsi = self.calculate_rsi(close)
        current_rsi = rsi.iloc[-1]
        
        atr = self.calculate_atr(high, low, close)
        current_atr = atr.iloc[-1]
        atr_percent = (current_atr / close.iloc[-1]) * 100
        
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(close)
        price_position = (close.iloc[-1] - lower_bb.iloc[-1]) / (upper_bb.iloc[-1] - lower_bb.iloc[-1])
        
        # Условия для BUY
        buy_conditions = {
            'trend_up': ema20 > ema50,  # Восходящий тренд
            'macd_bullish': current_macd > current_signal,  # MACD выше сигнальной
            'macd_rising': macd_histogram > macd_prev_histogram,  # Гистограмма растет
            'rsi_oversold': current_rsi < 35,  # Зона перепроданности
            'high_volatility': atr_percent > 0.5,  # Достаточная волатильность
            'price_lower_band': close.iloc[-1] < middle_bb.iloc[-1]  # Цена ниже средней линии Боллинджера
        }
        
        # Условия для SELL
        sell_conditions = {
            'trend_down': ema20 < ema50,  # Нисходящий тренд
            'macd_bearish': current_macd < current_signal,  # MACD ниже сигнальной
            'macd_falling': macd_histogram < macd_prev_histogram,  # Гистограмма падает
            'rsi_overbought': current_rsi > 65,  # Зона перекупленности
            'price_upper_band': close.iloc[-1] > middle_bb.iloc[-1]  # Цена выше средней линии Боллинджера
        }
        
        # Подсчет баллов
        buy_score = sum(buy_conditions.values())
        sell_score = sum(sell_conditions.values())
        
        # Определение сигнала
        signal = 'HOLD'
        confidence = 0.0
        
        if buy_score >= 4 and buy_conditions['high_volatility']:
            signal = 'BUY'
            confidence = min(0.9, buy_score / 6.0)
            # Увеличиваем уверенность при сильной перепроданности
            if current_rsi < 30:
                confidence = min(0.95, confidence + 0.15)
                
        elif sell_score >= 4 and not buy_conditions['trend_up']:
            signal = 'SELL'
            confidence = min(0.9, sell_score / 6.0)
            # Увеличиваем уверенность при сильной перекупленности
            if current_rsi > 70:
                confidence = min(0.95, confidence + 0.15)
        
        return {
            'signal': signal,
            'confidence': round(confidence, 2),
            'indicators': {
                'ema20': float(ema20),
                'ema50': float(ema50),
                'macd': float(current_macd),
                'macd_signal': float(current_signal),
                'macd_histogram': float(macd_histogram),
                'rsi': float(current_rsi),
                'atr_percent': float(atr_percent),
                'bb_position': float(price_position)
            },
            'conditions': {
                'buy': buy_conditions,
                'sell': sell_conditions,
                'buy_score': buy_score,
                'sell_score': sell_score
            }
        }
    
    def calculate_position_size(self, balance: float, risk_per_trade: float = 0.02,
                              stop_loss_pct: float = 0.03, price: float = None) -> Dict:
        """
        Расчет размера позиции по методу Келли с ограничением
        
        Args:
            balance: доступный баланс
            risk_per_trade: риск на сделку (2% по умолчанию)
            stop_loss_pct: стоп-лосс в процентах
            price: текущая цена актива
            
        Returns:
            {'size': количество, 'risk_amount': сумма риска}
        """
        if price is None or price <= 0:
            return {'size': 0, 'risk_amount': 0}
        
        # Максимальный риск на сделку
        max_risk_amount = balance * risk_per_trade
        
        # Количество с учетом стоп-лосса
        position_size = max_risk_amount / (price * stop_loss_pct)
        
        # Ограничение: не более 20% баланса на одну позицию
        max_position_value = balance * 0.2
        max_size = max_position_value / price
        
        position_size = min(position_size, max_size)
        
        return {
            'size': round(position_size, 6),
            'risk_amount': round(max_risk_amount, 2),
            'value': round(position_size * price, 2)
        }
