"""
Market-Specific ML Model Trainer

Обучает персональную ML модель для каждого торгуемого символа
на полной исторической глубине данных этого рынка.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Метрики качества модели"""
    symbol: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_samples: int
    test_samples: int
    training_date: str
    model_version: str = "1.0"
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MarketSpecificTrainer:
    """
    Тренер ML моделей специфичных для каждого рынка
    """
    
    def __init__(self, models_dir: str = "/var/lib/trading-bot/models"):
        """
        Args:
            models_dir: Директория для сохранения обученных моделей
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Минимальное количество данных для обучения
        self.min_training_samples = 1000
        
        # Параметры фичей
        self.lookback_period = 60  # Смотреть на последние 60 свечей
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'volume_sma'
        ]
    
    def _get_model_path(self, symbol: str) -> Path:
        """Путь к файлу модели"""
        symbol_dir = self.models_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / "model.pkl"
    
    def _get_metrics_path(self, symbol: str) -> Path:
        """Путь к файлу метрик"""
        symbol_dir = self.models_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / "metrics.json"
    
    def _get_scaler_path(self, symbol: str) -> Path:
        """Путь к файлу scaler"""
        symbol_dir = self.models_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / "scaler.pkl"
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать технические индикаторы для фичей
        
        Args:
            df: DataFrame с OHLCV данными
            
        Returns:
            DataFrame с добавленными индикаторами
        """
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_middle'] = sma_20
        df['bb_lower'] = sma_20 - (std_20 * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Заполнить NaN
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        
        return df
    
    def prepare_features_and_labels(
        self, 
        df: pd.DataFrame, 
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовить фичи и метки для обучения
        
        Args:
            df: DataFrame с данными и индикаторами
            prediction_horizon: Горизонт предсказания (количество свечей вперед)
            
        Returns:
            (X, y) - фичи и метки
        """
        # Расчет будущей доходности
        df['future_return'] = df['close'].shift(-prediction_horizon) / df['close'] - 1
        
        # Метки: 1 если цена вырастет >0.5%, 0 если упадет >0.5%, иначе нет сигнала
        df['label'] = 0
        df.loc[df['future_return'] > 0.005, 'label'] = 1  # UP
        df.loc[df['future_return'] < -0.005, 'label'] = -1  # DOWN
        
        # Удалить строки с NaN
        df_clean = df.dropna()
        
        if len(df_clean) < self.min_training_samples:
            raise ValueError(f"Insufficient data: {len(df_clean)} < {self.min_training_samples}")
        
        # Создать sequences (скользящее окно)
        X_list = []
        y_list = []
        
        for i in range(self.lookback_period, len(df_clean)):
            # Последние lookback_period свечей
            window = df_clean.iloc[i-self.lookback_period:i]
            
            # Извлечь фичи
            features = window[self.feature_columns].values.flatten()
            X_list.append(features)
            
            # Метка
            label = df_clean.iloc[i]['label']
            y_list.append(label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"✅ Prepared {len(X)} samples with {X.shape[1]} features each")
        
        return X, y
    
    def train_model(
        self, 
        symbol: str, 
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Optional[ModelMetrics]:
        """
        Обучить модель для конкретного символа
        
        Args:
            symbol: Торговый символ
            df: DataFrame с историческими данными
            test_size: Размер тестовой выборки (0.0 - 1.0)
            
        Returns:
            ModelMetrics с результатами обучения
        """
        logger.info(f"🎓 Training model for {symbol} on {len(df)} candles")
        
        try:
            # Рассчитать индикаторы
            df = self.calculate_technical_indicators(df)
            
            # Подготовить данные
            X, y = self.prepare_features_and_labels(df)
            
            # Разделить на train/test
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Нормализация
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Обучить модель (Random Forest для надежности)
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            
            logger.info("🔄 Training Random Forest...")
            model.fit(X_train_scaled, y_train)
            
            # Предсказание
            y_pred = model.predict(X_test_scaled)
            
            # Метрики
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Для multi-class нужно указать average
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = ModelMetrics(
                symbol=symbol,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                train_samples=len(X_train),
                test_samples=len(X_test),
                training_date=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Model trained successfully!")
            logger.info(f"   Accuracy: {accuracy:.4f}")
            logger.info(f"   Precision: {precision:.4f}")
            logger.info(f"   Recall: {recall:.4f}")
            logger.info(f"   F1 Score: {f1:.4f}")
            
            # Сохранить модель
            model_path = self._get_model_path(symbol)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Сохранить scaler
            scaler_path = self._get_scaler_path(symbol)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Сохранить метрики
            metrics_path = self._get_metrics_path(symbol)
            with open(metrics_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            
            logger.info(f"💾 Model saved to {model_path}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to train model for {symbol}: {e}", exc_info=True)
            return None
    
    def load_model(self, symbol: str) -> Optional[Tuple]:
        """
        Загрузить обученную модель
        
        Args:
            symbol: Торговый символ
            
        Returns:
            (model, scaler, metrics) или None
        """
        model_path = self._get_model_path(symbol)
        scaler_path = self._get_scaler_path(symbol)
        metrics_path = self._get_metrics_path(symbol)
        
        if not model_path.exists():
            logger.warning(f"No trained model found for {symbol}")
            return None
        
        try:
            # Загрузить модель
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Загрузить scaler
            scaler = None
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            
            # Загрузить метрики
            metrics = None
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            
            logger.info(f"✅ Loaded model for {symbol} (accuracy: {metrics.get('accuracy', 'N/A') if metrics else 'N/A'})")
            
            return model, scaler, metrics
            
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            return None
    
    def predict(
        self, 
        symbol: str, 
        recent_data: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Сделать предсказание для символа
        
        Args:
            symbol: Торговый символ
            recent_data: DataFrame с последними данными (минимум lookback_period свечей)
            
        Returns:
            Dict с предсказанием и уверенностью
        """
        model_data = self.load_model(symbol)
        if model_data is None:
            return None
        
        model, scaler, metrics = model_data
        
        try:
            # Рассчитать индикаторы
            df = self.calculate_technical_indicators(recent_data)
            
            # Взять последнее окно
            if len(df) < self.lookback_period:
                logger.warning(f"Insufficient data for prediction: {len(df)} < {self.lookback_period}")
                return None
            
            window = df.iloc[-self.lookback_period:]
            features = window[self.feature_columns].values.flatten().reshape(1, -1)
            
            # Нормализация
            if scaler:
                features = scaler.transform(features)
            
            # Предсказание
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Маппинг классов
            classes = model.classes_
            class_idx = np.where(classes == prediction)[0][0]
            confidence = probabilities[class_idx]
            
            # Интерпретация
            signal_map = {
                1: 'BUY',
                -1: 'SELL',
                0: 'HOLD'
            }
            
            result = {
                'signal': signal_map.get(prediction, 'HOLD'),
                'confidence': float(confidence),
                'prediction': int(prediction),
                'probabilities': {
                    signal_map.get(cls, f'class_{cls}'): float(prob)
                    for cls, prob in zip(classes, probabilities)
                },
                'model_accuracy': metrics.get('accuracy', 0.0) if metrics else 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}", exc_info=True)
            return None
    
    def get_model_info(self, symbol: str) -> Optional[Dict]:
        """Получить информацию о модели символа"""
        metrics_path = self._get_metrics_path(symbol)
        
        if not metrics_path.exists():
            return None
        
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read model info: {e}")
            return None
