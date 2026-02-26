# Performance Optimization Guide

## Overview

This guide covers profiling, tuning, and scaling the trading bot for high-throughput production use.

---

## Performance Baselines

Typical performance on a 2-core / 4 GB VPS:

| Operation | Typical Latency |
|-----------|----------------|
| OHLCV fetch (1 symbol, 100 candles) | 150–400 ms |
| Technical indicator computation | < 5 ms |
| ML ensemble inference (CPU) | 50–200 ms |
| BERT sentiment inference (CPU) | 800–2000 ms |
| BERT sentiment inference (GPU) | 50–150 ms |
| Order placement | 100–500 ms |
| Full signal cycle (no BERT) | 300–700 ms |
| Full signal cycle (with BERT) | 1–3 s |

---

## Profiling

### Application Profiling

```bash
# Profile main loop for 60 seconds
python -m cProfile -o profile.out src/main.py &
sleep 60
kill %1

# Analyze
python -m pstats profile.out
# In pstats: sort cumtime; stats 20
```

### Memory Profiling

```bash
pip install memory-profiler
python -m memory_profiler src/main.py
```

### Line-by-line Profiling

```python
from line_profiler import LineProfiler

lp = LineProfiler()
lp_wrapper = lp(your_function)
lp_wrapper()
lp.print_stats()
```

---

## Data Fetching Optimizations

### Enable Caching

```yaml
data_fetcher:
  cache_enabled: true
  cache_ttl_seconds: 60     # Cache OHLCV for 1 minute
  cache_backend: "memory"   # "memory" or "redis"
```

### Redis Cache (High Volume)

```yaml
data_fetcher:
  cache_backend: "redis"
  redis_url: "redis://localhost:6379/0"
  cache_ttl_seconds: 30
```

### Batch Fetching

Fetch multiple symbols in parallel:

```yaml
data_fetcher:
  async_enabled: true
  max_concurrent_requests: 5   # Respect exchange rate limits
```

### WebSocket Instead of Polling

For the lowest latency, use WebSocket data streams instead of REST polling:

```yaml
data_fetcher:
  mode: "websocket"   # instead of "polling"
  poll_interval_seconds: 5  # fallback only
```

---

## ML Inference Optimizations

### Model Caching

Models are loaded once at startup. Avoid reloading on each signal cycle.

### Reduce BERT Usage

BERT is the main bottleneck. Options:

1. **Run less frequently:**
   ```yaml
   sentiment:
     interval_minutes: 15   # Only re-run sentiment every 15 min
   ```

2. **Use lightweight alternative:**
   ```yaml
   sentiment:
     engine: "basic"   # VADER-based, ~10 ms per call
   ```

3. **GPU acceleration:**
   ```yaml
   sentiment:
     device: "cuda"   # or "mps" on Apple Silicon
   ```

### Async ML Inference via Celery

Offload ML to worker processes:

```bash
# Start Celery workers
celery -A src.celery_app worker \
  --concurrency=4 \
  --queue=ml_inference \
  --loglevel=info
```

```yaml
ml:
  async_inference: true
  celery_queue: "ml_inference"
```

### Model Quantization

Reduce BERT memory and CPU usage with quantization:

```python
from transformers import pipeline
import torch

model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    torch_dtype=torch.float16,  # half precision
)
```

---

## Database Optimizations

### SQLite Tuning (Small Deployments)

```python
# In database.py — applied at connection time
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.execute("PRAGMA cache_size=10000")
conn.execute("PRAGMA temp_store=MEMORY")
```

### PostgreSQL (Recommended for Production)

```yaml
database:
  type: postgresql
  pool_size: 10
  max_overflow: 20
  pool_pre_ping: true
```

Key indexes to maintain:

```sql
CREATE INDEX idx_trades_symbol_time ON trades(symbol, executed_at DESC);
CREATE INDEX idx_signals_symbol ON signals(symbol, created_at DESC);
```

---

## System-Level Optimizations

### CPU Affinity

Pin the bot process to specific CPU cores:

```bash
taskset -c 0,1 python src/main.py
```

### Process Priority

```bash
nice -n -10 python src/main.py   # Higher CPU priority (requires root)
```

### Huge Pages (Linux)

For large ML model loading:

```bash
echo 512 > /proc/sys/vm/nr_hugepages
```

---

## Horizontal Scaling

### Multiple Symbol Workers

Split symbol workload across multiple processes:

```yaml
# config-btc.yaml
trading:
  symbols: ["BTC/USDT"]

# config-eth.yaml
trading:
  symbols: ["ETH/USDT", "SOL/USDT"]
```

```bash
python src/main.py --config config-btc.yaml &
python src/main.py --config config-eth.yaml &
```

### Kubernetes Scaling

```bash
kubectl scale deployment trading-bot --replicas=3
```

Each pod handles a subset of symbols; shared PostgreSQL coordinates state.

---

## Monitoring Performance

### Key Metrics to Track

| Metric | Target |
|--------|--------|
| Signal cycle time | < 2 s |
| Order latency (place → ack) | < 1 s |
| Memory usage | < 3 GB |
| CPU usage (idle) | < 20% |
| Cache hit rate | > 80% |

### Prometheus Queries

```promql
# Signal cycle duration (p95)
histogram_quantile(0.95, signal_cycle_duration_seconds_bucket)

# Order fill rate
rate(orders_filled_total[5m]) / rate(orders_placed_total[5m])

# ML inference duration
histogram_quantile(0.99, ml_inference_duration_seconds_bucket)
```

---

## Benchmarking

Run the built-in benchmark suite:

```bash
python scripts/benchmark.py \
  --symbol BTC/USDT \
  --candles 1000 \
  --iterations 100 \
  --strategy enhanced_multi_indicator
```

This measures signal generation throughput and reports p50/p95/p99 latencies.
