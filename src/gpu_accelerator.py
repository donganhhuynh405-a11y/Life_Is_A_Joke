"""GPU acceleration utilities with lazy PyTorch import."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> Any:
    """
    Return the best available compute device.

    Parameters
    ----------
    prefer_gpu : bool
        If True, return a CUDA or MPS device when available.

    Returns
    -------
    torch.device or str
        A PyTorch device object, or ``"cpu"`` string if torch is unavailable.
    """
    try:
        import torch  # lazy import

        if prefer_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("GPU device selected: %s", torch.cuda.get_device_name(0))
                return device
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Apple MPS device selected.")
                return device

        return torch.device("cpu")
    except ImportError:
        logger.warning("torch not available; falling back to CPU-only mode.")
        return "cpu"


def device_info() -> Dict[str, Any]:
    """
    Return a dictionary of hardware and CUDA/MPS information.

    Returns
    -------
    dict
        Keys: ``torch_available``, ``cuda_available``, ``cuda_device_count``,
        ``cuda_device_name``, ``mps_available``, ``cpu_threads``.
    """
    info: Dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "mps_available": False,
        "cpu_threads": os.cpu_count(),
    }
    try:
        import torch
        info["torch_available"] = True
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------

def to_tensor(
    array: np.ndarray,
    device: Any = None,
    dtype: Optional[str] = "float32",
) -> Any:
    """
    Convert a NumPy array to a PyTorch tensor on the specified device.

    Parameters
    ----------
    array : np.ndarray
        Input array.
    device : optional
        Target device (from :func:`get_device`).  Defaults to CPU.
    dtype : str, optional
        NumPy/torch dtype string, e.g. ``"float32"``, ``"int64"``.

    Returns
    -------
    torch.Tensor or np.ndarray
        Falls back to the original array if torch is unavailable.
    """
    try:
        import torch

        np_dtype = np.dtype(dtype) if dtype else None
        arr = array.astype(np_dtype) if np_dtype else array
        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        if device is not None and not isinstance(device, str):
            tensor = tensor.to(device)
        return tensor
    except ImportError:
        return array


def from_tensor(tensor: Any) -> np.ndarray:
    """
    Convert a PyTorch tensor back to a NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray

    Returns
    -------
    np.ndarray
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    try:
        return tensor.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(tensor)


# ---------------------------------------------------------------------------
# Batch GPU inference helper
# ---------------------------------------------------------------------------

class GPUBatchInference:
    """
    Helper for batched GPU inference with automatic chunking.

    Splits large input arrays into chunks that fit in GPU memory and
    concatenates the results.

    Parameters
    ----------
    model : torch.nn.Module or callable
        Model to run inference on.
    chunk_size : int
        Maximum number of samples per GPU batch.
    device : optional
        Target device; defaults to best available.
    mixed_precision : bool
        Use ``torch.autocast`` for faster inference on supported hardware.
    """

    def __init__(
        self,
        model: Any,
        chunk_size: int = 512,
        device: Optional[Any] = None,
        mixed_precision: bool = True,
    ) -> None:
        self.model = model
        self.chunk_size = chunk_size
        self.device = device or get_device()
        self.mixed_precision = mixed_precision

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run batched inference on a NumPy array of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (n_samples, ...).

        Returns
        -------
        np.ndarray
            Model outputs, shape (n_samples, ...).
        """
        try:
            import torch

            self.model.eval()
            all_outputs: List[np.ndarray] = []

            with torch.no_grad():
                for start in range(0, len(inputs), self.chunk_size):
                    chunk = inputs[start : start + self.chunk_size]
                    tensor = to_tensor(chunk, device=self.device)

                    if self.mixed_precision and self.device != "cpu":
                        with torch.autocast(device_type=str(self.device).split(":")[0]):
                            output = self.model(tensor)
                    else:
                        output = self.model(tensor)

                    all_outputs.append(from_tensor(output))

            return np.concatenate(all_outputs, axis=0)

        except ImportError:
            logger.warning("torch not available; running inference on CPU via numpy.")
            return self._numpy_fallback(inputs)

    def _numpy_fallback(self, inputs: np.ndarray) -> np.ndarray:
        """Call the model directly if it accepts numpy arrays."""
        try:
            return self.model(inputs)
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference fallback failed: %s", exc)
            return np.zeros(len(inputs))


# ---------------------------------------------------------------------------
# Memory utilities
# ---------------------------------------------------------------------------

def clear_gpu_cache() -> None:
    """Release unused GPU memory back to the OS."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared.")
    except ImportError:
        pass


def gpu_memory_stats() -> Dict[str, float]:
    """
    Return current GPU memory usage in megabytes.

    Returns
    -------
    dict
        Keys: ``allocated_mb``, ``reserved_mb``, ``free_mb``.
        Returns zeros if CUDA is unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "free_mb": total - reserved,
            }
    except ImportError:
        pass
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0}
