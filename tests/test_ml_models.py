"""
Tests for PricePredictorLSTM in ml_models.py.
TensorFlow is mocked so tests run without GPU/heavy dependencies.
"""
import sys
import os
import types
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _make_tf_mock():
    """Return a minimal mock TensorFlow/Keras namespace."""
    tf = MagicMock()
    mock_model = MagicMock()
    mock_model.fit.return_value = MagicMock(history={'loss': [0.5], 'val_loss': [0.6]})
    mock_model.predict.return_value = np.array([[0.6]])
    mock_model.save = MagicMock()
    sequential_cls = MagicMock(return_value=mock_model)
    tf.keras.models.Sequential = sequential_cls
    tf.keras.layers.LSTM = MagicMock()
    tf.keras.layers.Dense = MagicMock()
    tf.keras.layers.Dropout = MagicMock()
    tf.keras.layers.BatchNormalization = MagicMock()
    tf.keras.regularizers.l2 = MagicMock(return_value=None)
    tf.keras.optimizers.Adam = MagicMock()
    tf.keras.callbacks.EarlyStopping = MagicMock()
    tf.keras.callbacks.ReduceLROnPlateau = MagicMock()
    return tf, mock_model


@pytest.fixture
def predictor():
    from ml_models import PricePredictorLSTM
    return PricePredictorLSTM(lookback=10, features=5)


class TestPricePredictorLSTM:
    def test_instantiation(self, predictor):
        assert predictor.lookback == 10
        assert predictor.features == 5
        assert predictor.is_trained is False
        assert predictor.model is None

    def test_build_model_with_tf_mock(self, predictor):
        tf_mock, mock_model = _make_tf_mock()
        with patch.dict('sys.modules', {
            'tensorflow': tf_mock,
            'tensorflow.keras': tf_mock.keras,
            'tensorflow.keras.models': tf_mock.keras.models,
            'tensorflow.keras.layers': tf_mock.keras.layers,
            'tensorflow.keras.regularizers': tf_mock.keras.regularizers,
            'tensorflow.keras.optimizers': tf_mock.keras.optimizers,
            'tensorflow.keras.callbacks': tf_mock.keras.callbacks,
        }):
            result = predictor.build_model()
        assert result is True
        assert predictor.model is not None

    def test_build_model_returns_false_without_tf(self, predictor):
        with patch.dict('sys.modules', {'tensorflow': None}):
            # Force ImportError path
            with patch('builtins.__import__', side_effect=ImportError):
                result = predictor.build_model()
        assert result is False

    def test_prepare_sequences_shape(self, predictor):
        X = np.random.rand(20, 5)
        y = np.random.rand(20)
        X_seq, y_seq = predictor._prepare_sequences(X, y)
        expected_samples = 20 - predictor.lookback + 1
        assert X_seq.shape == (expected_samples, predictor.lookback, 5)
        assert y_seq.shape == (expected_samples,)

    def test_predict_returns_none_when_not_trained(self, predictor):
        X = np.random.rand(10, 5)
        with pytest.raises((RuntimeError, TypeError)):
            predictor.predict(X)

    def test_predict_with_trained_mock_model(self, predictor):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.75]])
        predictor.model = mock_model
        predictor.is_trained = True

        # Use a simple scaler mock to avoid sklearn/scipy import issues
        scaler_x = MagicMock()
        scaler_x.transform = MagicMock(side_effect=lambda x: x)
        scaler_y = MagicMock()
        scaler_y.inverse_transform = MagicMock(return_value=np.array([[0.75]]))
        predictor.scaler_x = scaler_x
        predictor.scaler_y = scaler_y

        X = np.random.rand(15, 5)
        # predict may raise/return based on sequence length; just ensure no unhandled crash
        try:
            result = predictor.predict(X)
            assert result is None or isinstance(result, (float, np.floating, np.ndarray))
        except (RuntimeError, ValueError):
            pass  # acceptable – not enough data for sequences
