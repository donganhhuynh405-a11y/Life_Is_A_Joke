"""Transformer-based time series models for market prediction.

Imports for torch, transformers, etc. are done lazily inside functions
so the module can be imported even when those libraries are absent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TimeSeriesTransformer:
    """Transformer model for financial time-series forecasting.

    The model is built on top of PyTorch and the Hugging Face
    ``transformers`` library, both of which are imported lazily.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        device: Optional[str] = None,
    ) -> None:
        """Initialise the transformer configuration.

        Args:
            d_model: Embedding dimension for the transformer.
            nhead: Number of attention heads.
            num_encoder_layers: Depth of the encoder stack.
            num_decoder_layers: Depth of the decoder stack.
            dim_feedforward: Width of the feed-forward sublayer.
            dropout: Dropout probability.
            sequence_length: Number of time-steps fed as input.
            prediction_horizon: Number of future steps to predict.
            device: ``"cpu"`` or ``"cuda"``; auto-detected when *None*.
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self._device = device
        self._model: Any = None
        self._scaler: Any = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_device(self) -> Any:
        """Return the torch device, resolving *auto* when needed."""
        try:
            import torch  # noqa: PLC0415

            if self._device is not None:
                return torch.device(self._device)
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            logger.warning("PyTorch is not installed; device cannot be determined.")
            return None

    def _build_model(self, input_size: int) -> Any:
        """Construct the PyTorch transformer module.

        Args:
            input_size: Number of input features per time-step.

        Returns:
            An ``nn.Module`` instance.
        """
        try:
            import torch
            import torch.nn as nn

            class _TransformerModel(nn.Module):
                def __init__(
                    self,
                    input_size: int,
                    d_model: int,
                    nhead: int,
                    num_encoder_layers: int,
                    num_decoder_layers: int,
                    dim_feedforward: int,
                    dropout: float,
                    prediction_horizon: int,
                ) -> None:
                    super().__init__()
                    self.input_proj = nn.Linear(input_size, d_model)
                    self.transformer = nn.Transformer(
                        d_model=d_model,
                        nhead=nhead,
                        num_encoder_layers=num_encoder_layers,
                        num_decoder_layers=num_decoder_layers,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        batch_first=True,
                    )
                    self.output_proj = nn.Linear(d_model, prediction_horizon)

                def forward(self, src: Any, tgt: Any) -> Any:
                    src = self.input_proj(src)
                    tgt = self.input_proj(tgt)
                    out = self.transformer(src, tgt)
                    return self.output_proj(out[:, -1, :])

            return _TransformerModel(
                input_size=input_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                num_decoder_layers=self.num_decoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                prediction_horizon=self.prediction_horizon,
            )
        except ImportError:
            logger.error("PyTorch is required to build the transformer model.")
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """Train the transformer on time-series data.

        Args:
            X: Input array of shape ``(n_samples, sequence_length, n_features)``.
            y: Target array of shape ``(n_samples, prediction_horizon)``.
            epochs: Training epochs.
            batch_size: Mini-batch size.
            learning_rate: Initial learning rate for Adam.
            validation_split: Fraction of data held out for validation.

        Returns:
            Dictionary with ``"train_loss"`` and ``"val_loss"`` lists.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
        except ImportError as exc:
            logger.error("Missing dependency for transformer training: %s", exc)
            raise

        device = self._get_device()
        self._scaler = StandardScaler()
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self._scaler.fit_transform(X_flat).reshape(n_samples, seq_len, n_features)

        split = int(n_samples * (1 - validation_split))
        X_train, X_val = X_scaled[:split], X_scaled[split:]
        y_train, y_val = y[:split], y[split:]

        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)

        self._model = self._build_model(n_features).to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        for epoch in range(epochs):
            self._model.train()
            epoch_loss = 0.0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                pred = self._model(xb, xb[:, -1:, :])
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            history["train_loss"].append(epoch_loss / len(train_dl))

            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val_t, X_val_t[:, -1:, :])
                val_loss = criterion(val_pred, y_val_t).item()
            history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d – train_loss: %.6f  val_loss: %.6f",
                    epoch + 1,
                    epochs,
                    history["train_loss"][-1],
                    val_loss,
                )
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for the given input sequences.

        Args:
            X: Array of shape ``(n_samples, sequence_length, n_features)``.

        Returns:
            Predictions of shape ``(n_samples, prediction_horizon)``.
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        try:
            import torch
        except ImportError:
            logger.error("PyTorch is required for inference.")
            raise

        device = self._get_device()
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self._scaler.transform(X_flat).reshape(n_samples, seq_len, n_features)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self._model.eval()
        with torch.no_grad():
            preds = self._model(X_t, X_t[:, -1:, :])
        return preds.cpu().numpy()

    def save(self, path: str) -> None:
        """Persist the model weights to *path*.

        Args:
            path: File path for the saved ``.pt`` checkpoint.
        """
        if self._model is None:
            raise RuntimeError("No model to save.")
        try:
            import torch
        except ImportError:
            logger.error("PyTorch is required to save the model.")
            raise
        torch.save(self._model.state_dict(), path)
        logger.info("Transformer model saved to %s", path)

    def load(self, path: str, input_size: int) -> None:
        """Load model weights from *path*.

        Args:
            path: File path of the saved ``.pt`` checkpoint.
            input_size: Number of input features (must match the saved model).
        """
        try:
            import torch
        except ImportError:
            logger.error("PyTorch is required to load the model.")
            raise
        device = self._get_device()
        self._model = self._build_model(input_size).to(device)
        self._model.load_state_dict(torch.load(path, map_location=device))
        self._model.eval()
        logger.info("Transformer model loaded from %s", path)


class TemporalConvNet:
    """Temporal Convolutional Network (TCN) for sequence modelling.

    Wraps a dilated causal CNN stack implemented in PyTorch.
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        prediction_horizon: int = 1,
    ) -> None:
        """Initialise the TCN configuration.

        Args:
            input_channels: Number of input feature channels.
            hidden_channels: Channels in each residual block.
            num_levels: Number of dilated conv levels.
            kernel_size: Convolutional kernel size.
            dropout: Dropout probability.
            prediction_horizon: Steps ahead to predict.
        """
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.prediction_horizon = prediction_horizon
        self._model: Any = None

    def _build_model(self) -> Any:
        """Construct the TCN torch module."""
        try:
            import torch.nn as nn

            class _ResidualBlock(nn.Module):
                def __init__(self, in_ch: int, out_ch: int, dilation: int, kernel_size: int, dropout: float) -> None:
                    super().__init__()
                    pad = (kernel_size - 1) * dilation
                    self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
                    self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(dropout)
                    self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

                def forward(self, x: Any) -> Any:
                    out = self.relu(self.conv1(x)[:, :, : x.size(2)])
                    out = self.dropout(out)
                    out = self.relu(self.conv2(out)[:, :, : x.size(2)])
                    out = self.dropout(out)
                    res = x if self.downsample is None else self.downsample(x)
                    return self.relu(out + res)

            class _TCNModel(nn.Module):
                def __init__(
                    self,
                    input_channels: int,
                    hidden_channels: int,
                    num_levels: int,
                    kernel_size: int,
                    dropout: float,
                    prediction_horizon: int,
                ) -> None:
                    super().__init__()
                    layers = []
                    in_ch = input_channels
                    for i in range(num_levels):
                        dilation = 2 ** i
                        layers.append(_ResidualBlock(in_ch, hidden_channels, dilation, kernel_size, dropout))
                        in_ch = hidden_channels
                    self.network = nn.Sequential(*layers)
                    self.fc = nn.Linear(hidden_channels, prediction_horizon)

                def forward(self, x: Any) -> Any:
                    # x: (batch, channels, seq_len)
                    out = self.network(x)
                    return self.fc(out[:, :, -1])

            return _TCNModel(
                self.input_channels,
                self.hidden_channels,
                self.num_levels,
                self.kernel_size,
                self.dropout,
                self.prediction_horizon,
            )
        except ImportError:
            logger.error("PyTorch is required to build the TCN model.")
            raise

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32, learning_rate: float = 1e-3) -> Dict[str, List[float]]:
        """Train the TCN.

        Args:
            X: Input of shape ``(n_samples, channels, seq_len)``.
            y: Targets of shape ``(n_samples, prediction_horizon)``.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Adam learning rate.

        Returns:
            Training history dict.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            logger.error("Missing dependency: %s", exc)
            raise

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._build_model().to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.FloatTensor(y).to(device)
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        history: Dict[str, List[float]] = {"train_loss": []}
        for epoch in range(epochs):
            self._model.train()
            ep_loss = 0.0
            for xb, yb in dl:
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            history["train_loss"].append(ep_loss / len(dl))
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the trained TCN.

        Args:
            X: Array of shape ``(n_samples, channels, seq_len)``.

        Returns:
            Predictions of shape ``(n_samples, prediction_horizon)``.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        try:
            import torch
        except ImportError:
            raise
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()
        with torch.no_grad():
            return self._model(torch.FloatTensor(X).to(device)).cpu().numpy()
