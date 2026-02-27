"""
Trend Analyzer Module
Analyzes market trends and provides trend strength indicators
"""

import logging
import numpy as np
from typing import Dict, Tuple
from datetime import datetime


class TrendAnalyzer:
    """
    Analyzes market trends using multiple timeframes and indicators
    Provides trend direction, strength, and confidence metrics
    """

    TREND_BULLISH = "BULLISH"
    TREND_BEARISH = "BEARISH"
    TREND_SIDEWAYS = "SIDEWAYS"

    def __init__(self):
        """Initialize trend analyzer"""
        self.logger = logging.getLogger(__name__)

        # Trend strength thresholds
        self.strong_trend_threshold = 0.7  # 70% strength
        self.weak_trend_threshold = 0.3    # 30% strength

        # Cache for recent trend analysis
        self._trend_cache = {}
        self._cache_timeout = 300  # 5 minutes

    def analyze_trend(
        self,
        klines_data: list,
        symbol: str = None,
        timeframe: str = "1h"
    ) -> Dict:
        """
        Analyze trend from klines data

        Args:
            klines_data: List of klines (OHLCV data)
            symbol: Symbol being analyzed (for caching)
            timeframe: Timeframe of the data

        Returns:
            Dictionary with trend analysis:
            {
                'trend': 'BULLISH'|'BEARISH'|'SIDEWAYS',
                'strength': float (0-1),
                'confidence': float (0-100),
                'ema_trend': 'UP'|'DOWN'|'FLAT',
                'price_trend': 'UP'|'DOWN'|'FLAT',
                'volume_trend': 'INCREASING'|'DECREASING'|'STABLE',
                'adx': float (0-100),
                'trend_score': float (0-100),
                'description': str,
                'recommendation': str
            }
        """
        try:
            # Check cache
            cache_key = f"{symbol}_{timeframe}" if symbol else timeframe
            if cache_key in self._trend_cache:
                cached_time, cached_result = self._trend_cache[cache_key]
                if (datetime.now().timestamp() - cached_time) < self._cache_timeout:
                    return cached_result

            # Extract data from klines
            closes = np.array([float(k[4]) for k in klines_data])  # Close prices
            highs = np.array([float(k[2]) for k in klines_data])   # High prices
            lows = np.array([float(k[3]) for k in klines_data])    # Low prices
            volumes = np.array([float(k[5]) for k in klines_data])  # Volumes

            if len(closes) < 50:
                self.logger.warning(f"Insufficient data for trend analysis: {len(closes)} candles")
                return self._get_default_trend()

            # Calculate multiple trend indicators
            ema_trend_result = self._analyze_ema_trend(closes)
            price_action_result = self._analyze_price_action(closes, highs, lows)
            volume_result = self._analyze_volume_trend(volumes)
            adx_value = self._calculate_adx(highs, lows, closes)

            # Combine indicators to determine overall trend
            trend_direction, trend_strength = self._determine_overall_trend(
                ema_trend_result,
                price_action_result,
                adx_value
            )

            # Calculate confidence based on indicator agreement
            confidence = self._calculate_confidence(
                ema_trend_result,
                price_action_result,
                adx_value
            )

            # Calculate overall trend score (0-100)
            trend_score = self._calculate_trend_score(
                trend_direction,
                trend_strength,
                confidence,
                adx_value
            )

            # Generate description and recommendation
            description = self._generate_description(
                trend_direction,
                trend_strength,
                adx_value
            )

            recommendation = self._generate_recommendation(
                trend_direction,
                trend_strength,
                trend_score
            )

            result = {
                'trend': trend_direction,
                'strength': trend_strength,
                'confidence': confidence,
                'ema_trend': ema_trend_result['direction'],
                'price_trend': price_action_result['direction'],
                'volume_trend': volume_result['trend'],
                'adx': adx_value,
                'trend_score': trend_score,
                'description': description,
                'recommendation': recommendation,
                'timestamp': datetime.now().isoformat()
            }

            # Cache result
            self._trend_cache[cache_key] = (datetime.now().timestamp(), result)

            return result

        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}", exc_info=True)
            return self._get_default_trend()

    def _analyze_ema_trend(self, closes: np.ndarray) -> Dict:
        """Analyze trend using EMAs"""
        # Calculate EMAs
        ema_9 = self._calculate_ema(closes, 9)
        ema_21 = self._calculate_ema(closes, 21)
        ema_50 = self._calculate_ema(closes, 50)

        current_price = closes[-1]

        # Determine EMA trend
        if ema_9 > ema_21 > ema_50 and current_price > ema_9:
            direction = "UP"
            strength = min(1.0, (ema_9 - ema_50) / ema_50 * 10)  # Normalized
        elif ema_9 < ema_21 < ema_50 and current_price < ema_9:
            direction = "DOWN"
            strength = min(1.0, (ema_50 - ema_9) / ema_50 * 10)  # Normalized
        else:
            direction = "FLAT"
            strength = 0.3

        return {
            'direction': direction,
            'strength': strength,
            'ema_9': ema_9,
            'ema_21': ema_21,
            'ema_50': ema_50
        }

    def _analyze_price_action(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Dict:
        """Analyze price action patterns"""
        # Check for higher highs and higher lows (uptrend)
        # Check for higher highs and higher lows (uptrend) via linear regression
        _ = highs[-20:]  # noqa: F841
        _ = lows[-20:]   # noqa: F841

        # Simple trend detection using linear regression
        x = np.arange(len(closes[-20:]))
        y = closes[-20:]

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        avg_price = np.mean(y)

        # Normalize slope
        normalized_slope = slope / avg_price if avg_price > 0 else 0

        if normalized_slope > 0.001:  # Upward slope
            direction = "UP"
            strength = min(1.0, abs(normalized_slope) * 100)
        elif normalized_slope < -0.001:  # Downward slope
            direction = "DOWN"
            strength = min(1.0, abs(normalized_slope) * 100)
        else:
            direction = "FLAT"
            strength = 0.2

        return {
            'direction': direction,
            'strength': strength,
            'slope': normalized_slope
        }

    def _analyze_volume_trend(self, volumes: np.ndarray) -> Dict:
        """Analyze volume trends"""
        recent_volume = np.mean(volumes[-10:])
        older_volume = np.mean(volumes[-30:-10])

        if recent_volume > older_volume * 1.2:
            trend = "INCREASING"
        elif recent_volume < older_volume * 0.8:
            trend = "DECREASING"
        else:
            trend = "STABLE"

        return {
            'trend': trend,
            'recent_avg': recent_volume,
            'older_avg': older_volume
        }

    def _calculate_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """
        Calculate Average Directional Index (ADX)
        ADX measures trend strength regardless of direction
        """
        try:
            # Calculate True Range
            high_low = highs - lows
            high_close = np.abs(highs - np.roll(closes, 1))
            low_close = np.abs(lows - np.roll(closes, 1))

            true_range = np.maximum(high_low, np.maximum(high_close, low_close))

            # Calculate Directional Movement
            up_move = highs - np.roll(highs, 1)
            down_move = np.roll(lows, 1) - lows

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            # Smooth the values
            atr = self._smooth(true_range, period)

            # Ensure atr is valid (not NaN, not zero)
            atr = np.nan_to_num(atr, nan=1.0, posinf=1.0, neginf=1.0)
            atr = np.where(atr == 0, 1.0, atr)  # Replace zeros with 1.0

            # Now safe to divide
            plus_di = 100 * self._smooth(plus_dm, period) / atr
            minus_di = 100 * self._smooth(minus_dm, period) / atr

            # Calculate DX and ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = self._smooth(dx, period)

            return float(adx[-1]) if len(adx) > 0 else 25.0

        except Exception as e:
            self.logger.warning(f"Error calculating ADX: {str(e)}")
            return 25.0  # Neutral default

    def _smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Apply exponential smoothing"""
        alpha = 2 / (period + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]

        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]

        return smoothed

    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA for given period"""
        if len(data) < period:
            return float(np.mean(data))

        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])  # Start with SMA

        for price in data[period:]:
            ema = (price - ema) * multiplier + ema

        return float(ema)

    def _determine_overall_trend(
        self,
        ema_result: Dict,
        price_result: Dict,
        adx: float
    ) -> Tuple[str, float]:
        """Determine overall trend from multiple indicators"""
        # Count bullish and bearish signals
        bullish_votes = 0
        bearish_votes = 0

        if ema_result['direction'] == "UP":
            bullish_votes += 1
        elif ema_result['direction'] == "DOWN":
            bearish_votes += 1

        if price_result['direction'] == "UP":
            bullish_votes += 1
        elif price_result['direction'] == "DOWN":
            bearish_votes += 1

        # Determine trend direction
        if bullish_votes > bearish_votes:
            trend = self.TREND_BULLISH
        elif bearish_votes > bullish_votes:
            trend = self.TREND_BEARISH
        else:
            trend = self.TREND_SIDEWAYS

        # Calculate strength (combination of indicator strengths and ADX)
        avg_indicator_strength = (ema_result['strength'] + price_result['strength']) / 2
        adx_factor = min(1.0, adx / 50)  # Normalize ADX (50+ is strong trend)

        overall_strength = (avg_indicator_strength * 0.6 + adx_factor * 0.4)

        return trend, overall_strength

    def _calculate_confidence(
        self,
        ema_result: Dict,
        price_result: Dict,
        adx: float
    ) -> float:
        """Calculate confidence in trend analysis"""
        # Base confidence from indicator agreement
        if ema_result['direction'] == price_result['direction']:
            confidence = 80.0
        else:
            confidence = 50.0

        # Adjust based on ADX (trend strength)
        if adx > 40:
            confidence = min(100, confidence + 15)
        elif adx > 25:
            confidence = min(100, confidence + 5)
        elif adx < 15:
            confidence = max(30, confidence - 15)

        return confidence

    def _calculate_trend_score(
        self,
        trend: str,
        strength: float,
        confidence: float,
        adx: float
    ) -> float:
        """Calculate overall trend score (0-100)"""
        # Base score from confidence
        score = confidence * 0.4

        # Add strength component
        score += strength * 100 * 0.3

        # Add ADX component (trend quality)
        score += min(100, adx * 2) * 0.3

        # Penalize sideways trends
        if trend == self.TREND_SIDEWAYS:
            score *= 0.6

        return min(100, max(0, score))

    def _generate_description(
        self,
        trend: str,
        strength: float,
        adx: float
    ) -> str:
        """Generate human-readable trend description"""
        # Describe strength
        if strength > self.strong_trend_threshold:
            strength_desc = "сильный"
        elif strength > self.weak_trend_threshold:
            strength_desc = "умеренный"
        else:
            strength_desc = "слабый"

        # Describe trend quality (ADX)
        if adx > 40:
            quality = "чёткий"
        elif adx > 25:
            quality = "развивающийся"
        else:
            quality = "неопределённый"

        if trend == self.TREND_BULLISH:
            return f"{strength_desc.capitalize()} {quality} восходящий тренд (ADX: {adx:.1f})"
        elif trend == self.TREND_BEARISH:
            return f"{strength_desc.capitalize()} {quality} нисходящий тренд (ADX: {adx:.1f})"
        else:
            return f"Боковой тренд, направление неопределённое (ADX: {adx:.1f})"

    def _generate_recommendation(
        self,
        trend: str,
        strength: float,
        score: float
    ) -> str:
        """Generate trading recommendation based on trend"""
        if trend == self.TREND_BULLISH:
            if strength > self.strong_trend_threshold and score > 70:
                return "Отличные условия для покупки (BUY). Сильный восходящий тренд."
            elif strength > self.weak_trend_threshold and score > 60:
                return "Хорошие условия для покупки (BUY). Развивающийся тренд."
            else:
                return "Возможность покупки (BUY), но с осторожностью. Слабый тренд."

        elif trend == self.TREND_BEARISH:
            if strength > self.strong_trend_threshold and score > 70:
                return "Избегать покупок. Сильный нисходящий тренд. Рассмотреть SHORT."
            elif strength > self.weak_trend_threshold and score > 60:
                return "Осторожно с покупками. Развивающийся нисходящий тренд."
            else:
                return "Слабый нисходящий тренд. Ждать подтверждения разворота."

        else:
            return "Боковой рынок. Ждать прорыва или использовать range-стратегию."

    def _get_default_trend(self) -> Dict:
        """Return default trend when analysis fails"""
        return {
            'trend': self.TREND_SIDEWAYS,
            'strength': 0.3,
            'confidence': 50.0,
            'ema_trend': 'FLAT',
            'price_trend': 'FLAT',
            'volume_trend': 'STABLE',
            'adx': 20.0,
            'trend_score': 40.0,
            'description': 'Недостаточно данных для анализа тренда',
            'recommendation': 'Ждать дополнительных данных',
            'timestamp': datetime.now().isoformat()
        }

    def get_trend_summary(self, trends: Dict[str, Dict]) -> str:
        """
        Generate summary of trends across multiple symbols

        Args:
            trends: Dictionary mapping symbol to trend analysis

        Returns:
            Human-readable summary string
        """
        if not trends:
            return "Нет данных о трендах"

        bullish_count = sum(1 for t in trends.values() if t['trend'] == self.TREND_BULLISH)
        bearish_count = sum(1 for t in trends.values() if t['trend'] == self.TREND_BEARISH)
        sideways_count = sum(1 for t in trends.values() if t['trend'] == self.TREND_SIDEWAYS)

        avg_score = np.mean([t['trend_score'] for t in trends.values()])
        avg_strength = np.mean([t['strength'] for t in trends.values()])

        summary = f"Анализ рынка ({len(trends)} символов):\n"
        summary += f"  Восходящих трендов: {bullish_count}\n"
        summary += f"  Нисходящих трендов: {bearish_count}\n"
        summary += f"  Боковых трендов: {sideways_count}\n"
        summary += f"  Средний балл тренда: {avg_score:.1f}/100\n"
        summary += f"  Средняя сила: {avg_strength * 100:.1f}%\n"

        return summary
