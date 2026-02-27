"""Advanced feature engineering using only pandas and numpy.

No external ML libraries (xgboost, lightgbm, shap, etc.) are used here,
ensuring this module is always importable.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute a rich set of technical and statistical features from OHLCV data.

    All computations rely solely on ``pandas`` and ``numpy``.
    """

    def __init__(
        self,
        close_col: str = "close",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        volume_col: str = "volume",
    ) -> None:
        """Initialise column name mapping.

        Args:
            close_col: Name of the closing-price column.
            open_col: Name of the opening-price column.
            high_col: Name of the high-price column.
            low_col: Name of the low-price column.
            volume_col: Name of the volume column.
        """
        self.close_col = close_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col

    # ------------------------------------------------------------------
    # Individual feature groups
    # ------------------------------------------------------------------

    def add_returns(self, df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Add log-return features for multiple look-back periods.

        Args:
            df: Input OHLCV DataFrame.
            periods: List of look-back periods (in bars).

        Returns:
            DataFrame with new return columns appended.
        """
        df = df.copy()
        for p in periods:
            df[f"log_return_{p}"] = np.log(df[self.close_col] / df[self.close_col].shift(p))
        return df

    def add_moving_averages(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20, 50, 100, 200],
    ) -> pd.DataFrame:
        """Add simple and exponential moving averages.

        Args:
            df: Input DataFrame.
            windows: Rolling window sizes.

        Returns:
            DataFrame with SMA and EMA columns appended.
        """
        df = df.copy()
        for w in windows:
            df[f"sma_{w}"] = df[self.close_col].rolling(w).mean()
            df[f"ema_{w}"] = df[self.close_col].ewm(span=w, adjust=False).mean()
        return df

    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """Add Bollinger Band features.

        Args:
            df: Input DataFrame.
            window: Rolling window for the moving average.
            num_std: Width of the bands in standard deviations.

        Returns:
            DataFrame with BB_upper, BB_lower, BB_width, BB_pct appended.
        """
        df = df.copy()
        mid = df[self.close_col].rolling(window).mean()
        std = df[self.close_col].rolling(window).std()
        df["BB_upper"] = mid + num_std * std
        df["BB_lower"] = mid - num_std * std
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / mid.replace(0, np.nan)
        df["BB_pct"] = (df[self.close_col] - df["BB_lower"]) / \
            (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
        return df

    def add_rsi(self, df: pd.DataFrame, windows: List[int] = [7, 14, 21]) -> pd.DataFrame:
        """Add Relative Strength Index for multiple periods.

        Args:
            df: Input DataFrame.
            windows: RSI periods.

        Returns:
            DataFrame with RSI columns appended.
        """
        df = df.copy()
        delta = df[self.close_col].diff()
        for w in windows:
            gain = delta.clip(lower=0).rolling(w).mean()
            loss = (-delta.clip(upper=0)).rolling(w).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f"rsi_{w}"] = 100 - (100 / (1 + rs))
        return df

    def add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Add MACD, MACD signal, and MACD histogram.

        Args:
            df: Input DataFrame.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal EMA period.

        Returns:
            DataFrame with MACD, MACD_signal, MACD_hist appended.
        """
        df = df.copy()
        ema_fast = df[self.close_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[self.close_col].ewm(span=slow, adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
        return df

    def add_atr(self, df: pd.DataFrame, windows: List[int] = [7, 14, 21]) -> pd.DataFrame:
        """Add Average True Range (ATR) for multiple periods.

        Args:
            df: Input DataFrame (must have high, low, close columns).
            windows: ATR periods.

        Returns:
            DataFrame with ATR columns appended.
        """
        df = df.copy()
        prev_close = df[self.close_col].shift(1)
        tr = pd.concat(
            [
                df[self.high_col] - df[self.low_col],
                (df[self.high_col] - prev_close).abs(),
                (df[self.low_col] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        for w in windows:
            df[f"atr_{w}"] = tr.rolling(w).mean()
        return df

    def add_stochastic(
        self,
        df: pd.DataFrame,
        k_window: int = 14,
        d_window: int = 3,
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator (%K and %D).

        Args:
            df: Input DataFrame.
            k_window: Look-back window for %K.
            d_window: Smoothing window for %D.

        Returns:
            DataFrame with stoch_k and stoch_d appended.
        """
        df = df.copy()
        low_min = df[self.low_col].rolling(k_window).min()
        high_max = df[self.high_col].rolling(k_window).max()
        df["stoch_k"] = 100 * (df[self.close_col] - low_min) / \
            (high_max - low_min).replace(0, np.nan)
        df["stoch_d"] = df["stoch_k"].rolling(d_window).mean()
        return df

    def add_volume_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [
            5,
            10,
            20]) -> pd.DataFrame:
        """Add volume-based features.

        Args:
            df: Input DataFrame.
            windows: Look-back windows for rolling statistics.

        Returns:
            DataFrame with volume ratio and OBV columns appended.
        """
        df = df.copy()
        for w in windows:
            vol_ma = df[self.volume_col].rolling(w).mean()
            df[f"vol_ratio_{w}"] = df[self.volume_col] / vol_ma.replace(0, np.nan)

        # On-Balance Volume
        direction = np.sign(df[self.close_col].diff()).fillna(0)
        df["obv"] = (df[self.volume_col] * direction).cumsum()
        return df

    def add_statistical_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [10, 20, 50],
    ) -> pd.DataFrame:
        """Add rolling statistical moments and z-scores.

        Args:
            df: Input DataFrame.
            windows: Rolling window sizes.

        Returns:
            DataFrame with skewness, kurtosis, and z-score columns.
        """
        df = df.copy()
        for w in windows:
            rolling = df[self.close_col].rolling(w)
            df[f"skew_{w}"] = rolling.skew()
            df[f"kurt_{w}"] = rolling.kurt()
            mean = rolling.mean()
            std = rolling.std()
            df[f"zscore_{w}"] = (df[self.close_col] - mean) / std.replace(0, np.nan)
        return df

    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 10],
    ) -> pd.DataFrame:
        """Add lagged versions of specified columns.

        Args:
            df: Input DataFrame.
            columns: Column names to lag.
            lags: Lag periods.

        Returns:
            DataFrame with lagged columns appended.
        """
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                logger.warning("Column '%s' not found; skipping lag features.", col)
                continue
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    def add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple candlestick pattern indicators.

        Args:
            df: Input DataFrame with open, high, low, close.

        Returns:
            DataFrame with pattern columns appended.
        """
        df = df.copy()
        body = df[self.close_col] - df[self.open_col]
        wick_up = df[self.high_col] - df[[self.close_col, self.open_col]].max(axis=1)
        wick_down = df[[self.close_col, self.open_col]].min(axis=1) - df[self.low_col]
        range_ = df[self.high_col] - df[self.low_col]

        df["candle_body"] = body
        df["candle_body_pct"] = body / range_.replace(0, np.nan)
        df["upper_wick_pct"] = wick_up / range_.replace(0, np.nan)
        df["lower_wick_pct"] = wick_down / range_.replace(0, np.nan)
        df["is_doji"] = (body.abs() <= 0.1 * range_).astype(int)
        df["is_bullish"] = (body > 0).astype(int)
        return df

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def build_all_features(
        self,
        df: pd.DataFrame,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """Apply all feature groups in sequence.

        Args:
            df: Raw OHLCV DataFrame.
            drop_na: Whether to drop rows with any NaN values after
                feature construction.

        Returns:
            Feature-enriched DataFrame.
        """
        df = self.add_returns(df)
        df = self.add_moving_averages(df)
        df = self.add_bollinger_bands(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_atr(df)
        df = self.add_stochastic(df)
        df = self.add_volume_features(df)
        df = self.add_statistical_features(df)
        df = self.add_candlestick_patterns(df)

        # Lag key features
        lag_cols = ["log_return_1", "rsi_14", "MACD", "atr_14"]
        existing_lag_cols = [c for c in lag_cols if c in df.columns]
        df = self.add_lag_features(df, existing_lag_cols)

        if drop_na:
            initial_len = len(df)
            df = df.dropna()
            logger.info("Dropped %d NaN rows; %d remain.", initial_len - len(df), len(df))

        return df

    def get_feature_columns(self, df: pd.DataFrame,
                            exclude: Optional[List[str]] = None) -> List[str]:
        """Return the list of engineered feature column names.

        Args:
            df: Feature-enriched DataFrame.
            exclude: Columns to exclude from the returned list.

        Returns:
            List of feature column names.
        """
        base_cols = {self.close_col, self.open_col, self.high_col, self.low_col, self.volume_col}
        exclude_set = (set(exclude) if exclude else set()) | base_cols
        return [c for c in df.columns if c not in exclude_set]
