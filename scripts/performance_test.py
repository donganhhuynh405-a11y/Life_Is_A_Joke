#!/usr/bin/env python3
"""
performance_test.py - Performance testing and benchmarking script.

Measures latency, throughput, and resource usage of key bot components.

Usage:
    python scripts/performance_test.py [--component all|data|ml|risk|sentiment]
"""

import argparse
import asyncio
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("performance_test")


class PerformanceTimer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def benchmark(func: Callable, iterations: int = 100, warmup: int = 5,
              label: str = "") -> dict[str, float]:
    """Benchmark a callable and return statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    gc.collect()
    timings = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)

    timings.sort()
    n = len(timings)
    total = sum(timings)

    return {
        "label": label or getattr(func, "__name__", "benchmark"),
        "iterations": n,
        "mean_ms": (total / n) * 1000,
        "min_ms": timings[0] * 1000,
        "max_ms": timings[-1] * 1000,
        "p50_ms": timings[n // 2] * 1000,
        "p95_ms": timings[int(n * 0.95)] * 1000,
        "p99_ms": timings[int(n * 0.99)] * 1000,
        "throughput_rps": n / total,
    }


def print_results(results: list[dict]) -> None:
    """Print benchmark results in a formatted table."""
    if not results:
        return

    print(f"\n{'Component':<35} {'Mean (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'RPS':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['label']:<35} "
            f"{r['mean_ms']:>10.3f} "
            f"{r['p95_ms']:>10.3f} "
            f"{r['p99_ms']:>10.3f} "
            f"{r['throughput_rps']:>10.1f}"
        )
    print()


def benchmark_sentiment(n_iterations: int = 200) -> list[dict]:
    """Benchmark sentiment analysis."""
    print("Benchmarking: Sentiment Analysis")

    from src.sentiment import SentimentAnalyzer

    cfg = {}
    analyzer = SentimentAnalyzer(cfg)

    texts_small = ["Bitcoin is going to the moon! Very bullish!"]
    texts_large = texts_small * 20

    results = []
    results.append(benchmark(
        lambda: analyzer.analyze_texts(texts_small),
        iterations=n_iterations,
        label="SentimentAnalyzer.analyze (1 text)",
    ))
    results.append(benchmark(
        lambda: analyzer.analyze_texts(texts_large),
        iterations=n_iterations // 4,
        label="SentimentAnalyzer.analyze (20 texts)",
    ))
    results.append(benchmark(
        lambda: analyzer.analyze_single_text("BTC crash incoming, sell everything!"),
        iterations=n_iterations,
        label="SentimentAnalyzer.analyze_single",
    ))

    return results


def benchmark_risk_manager(n_iterations: int = 500) -> list[dict]:
    """Benchmark risk management calculations."""
    print("Benchmarking: Risk Manager")

    # Mock the WALLogger to avoid file I/O
    with patch("src.risk_manager.WALLogger", MagicMock):
        from src.risk_manager import RiskManager

        cfg = {}
        rm = RiskManager(cfg)

    results = []
    results.append(benchmark(
        lambda: rm.compute_position_size(10000, 0.05, 0.6),
        iterations=n_iterations,
        label="RiskManager.compute_position_size",
    ))

    return results


def benchmark_feature_engineering(n_iterations: int = 50) -> list[dict]:
    """Benchmark feature engineering."""
    print("Benchmarking: Feature Engineering")
    try:
        import numpy as np
        import pandas as pd
        from src.ml_advanced.feature_engineering import FeatureEngineer

        engineer = FeatureEngineer()

        # Create mock OHLCV data
        dates = pd.date_range("2023-01-01", periods=500, freq="1h")
        np.random.seed(42)
        prices = 30000 + np.cumsum(np.random.randn(500) * 100)
        df = pd.DataFrame({
            "open": prices * (1 + np.random.randn(500) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(500)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(500)) * 0.002),
            "close": prices,
            "volume": np.random.uniform(100, 1000, 500),
        }, index=dates)

        results = []
        results.append(benchmark(
            lambda: engineer.create_features(df),
            iterations=n_iterations,
            label="FeatureEngineer.create_features (500 rows)",
        ))
        return results
    except Exception as e:
        logger.warning("Feature engineering benchmark skipped: %s", e)
        return []


def benchmark_portfolio_optimizer(n_iterations: int = 20) -> list[dict]:
    """Benchmark portfolio optimization."""
    print("Benchmarking: Portfolio Optimizer")
    try:
        import numpy as np
        import pandas as pd
        from src.portfolio.optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()

        # Create mock returns data
        np.random.seed(42)
        n_assets = 5
        n_periods = 252
        symbols = [f"ASSET{i}" for i in range(n_assets)]
        returns = pd.DataFrame(
            np.random.randn(n_periods, n_assets) * 0.02,
            columns=symbols,
        )

        results = []
        results.append(benchmark(
            lambda: optimizer.optimize_max_sharpe(returns),
            iterations=n_iterations,
            label="PortfolioOptimizer.max_sharpe (5 assets)",
        ))
        results.append(benchmark(
            lambda: optimizer.optimize_min_volatility(returns),
            iterations=n_iterations,
            label="PortfolioOptimizer.min_vol (5 assets)",
        ))
        return results
    except Exception as e:
        logger.warning("Portfolio optimizer benchmark skipped: %s", e)
        return []


def benchmark_cache_strategies(n_iterations: int = 1000) -> list[dict]:
    """Benchmark cache strategy performance."""
    print("Benchmarking: Cache Strategies")
    try:
        from src.cache.strategies import LRUStrategy, TTLStrategy

        lru = LRUStrategy(maxsize=128)
        ttl = TTLStrategy(maxsize=128, ttl=300)

        # Populate
        for i in range(50):
            lru.set(f"key:{i}", f"value:{i}")
            ttl.set(f"key:{i}", f"value:{i}")

        results = []
        results.append(benchmark(
            lambda: lru.get("key:25"),
            iterations=n_iterations,
            label="LRUStrategy.get (hit)",
        ))
        results.append(benchmark(
            lambda: lru.set("key:new", "new_value"),
            iterations=n_iterations,
            label="LRUStrategy.set",
        ))
        results.append(benchmark(
            lambda: ttl.get("key:25"),
            iterations=n_iterations,
            label="TTLStrategy.get (hit)",
        ))
        return results
    except Exception as e:
        logger.warning("Cache benchmark skipped: %s", e)
        return []


COMPONENT_MAP = {
    "sentiment": benchmark_sentiment,
    "risk": benchmark_risk_manager,
    "features": benchmark_feature_engineering,
    "portfolio": benchmark_portfolio_optimizer,
    "cache": benchmark_cache_strategies,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance testing")
    parser.add_argument(
        "--component",
        choices=["all"] + list(COMPONENT_MAP.keys()),
        default="all",
        help="Component to benchmark (default: all)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override default iteration count",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  Trading Bot Performance Benchmarks")
    print("=" * 80)

    all_results = []

    components = COMPONENT_MAP if args.component == "all" else {
        args.component: COMPONENT_MAP[args.component]
    }

    for name, bench_fn in components.items():
        print(f"\n[{name.upper()}]")
        try:
            kwargs = {}
            if args.iterations:
                kwargs["n_iterations"] = args.iterations
            results = bench_fn(**kwargs)
            all_results.extend(results)
            print_results(results)
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("=" * 80)
    print("  Summary")
    print("=" * 80)
    print_results(all_results)

    # Flag slow operations
    slow = [r for r in all_results if r.get("p95_ms", 0) > 100]
    if slow:
        print("⚠ Slow operations (P95 > 100ms):")
        for r in slow:
            print(f"  - {r['label']}: {r['p95_ms']:.1f}ms")
    else:
        print("✓ All operations within acceptable latency thresholds")
    print()


if __name__ == "__main__":
    main()
