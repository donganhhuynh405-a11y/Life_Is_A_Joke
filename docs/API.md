# API Documentation

## Overview

The trading bot exposes a REST API (default port `8080`) for monitoring, control, and integration. Authentication uses Bearer tokens configured in `config.yaml`.

Base URL: `http://localhost:8080/api/v1`

---

## Authentication

All endpoints require an `Authorization` header:

```
Authorization: Bearer <API_TOKEN>
```

The token is set via the `API_TOKEN` environment variable.

---

## Endpoints

### Health & Status

#### `GET /health`
Returns service health status.

**Response**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "environment": "paper"
}
```

#### `GET /status`
Returns full bot status including active positions and P&L.

**Response**
```json
{
  "running": true,
  "environment": "paper",
  "active_positions": 2,
  "total_trades_today": 14,
  "pnl_today_pct": 1.23,
  "portfolio_value_usdt": 10250.00,
  "last_signal": {
    "symbol": "BTC/USDT",
    "action": "BUY",
    "confidence": 0.82,
    "timestamp": "2024-01-15T12:00:00Z"
  }
}
```

---

### Positions

#### `GET /positions`
List all open positions.

**Response**
```json
{
  "positions": [
    {
      "id": "pos_001",
      "symbol": "BTC/USDT",
      "side": "long",
      "size": 0.01,
      "entry_price": 42000.00,
      "current_price": 43500.00,
      "unrealized_pnl": 15.00,
      "unrealized_pnl_pct": 3.57,
      "stop_loss": 41000.00,
      "take_profit": 45000.00,
      "opened_at": "2024-01-15T10:00:00Z"
    }
  ]
}
```

#### `DELETE /positions/{id}`
Close a specific position at market price.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| `id` | string | Position ID |

**Response** `200 OK`
```json
{"message": "Position pos_001 closed", "fill_price": 43450.00}
```

---

### Signals

#### `GET /signals`
Returns the latest trading signals.

**Query Parameters**
| Name | Default | Description |
|------|---------|-------------|
| `symbol` | all | Filter by symbol (e.g. `BTC/USDT`) |
| `limit` | 10 | Number of records to return |

**Response**
```json
{
  "signals": [
    {
      "symbol": "BTC/USDT",
      "action": "BUY",
      "confidence": 0.87,
      "strategy": "enhanced_multi_indicator",
      "ml_prediction": "UP",
      "sentiment_score": 0.65,
      "timestamp": "2024-01-15T12:00:00Z"
    }
  ]
}
```

---

### Trades

#### `GET /trades`
Historical trade log.

**Query Parameters**
| Name | Default | Description |
|------|---------|-------------|
| `symbol` | all | Filter by symbol |
| `from` | 7d ago | ISO 8601 start datetime |
| `to` | now | ISO 8601 end datetime |
| `limit` | 50 | Max records |

**Response**
```json
{
  "trades": [
    {
      "id": "trade_123",
      "symbol": "ETH/USDT",
      "side": "buy",
      "quantity": 0.5,
      "price": 2200.00,
      "fee": 1.10,
      "pnl": 45.00,
      "pnl_pct": 4.09,
      "strategy": "classic_macd_rsi",
      "executed_at": "2024-01-14T09:30:00Z"
    }
  ],
  "total": 1,
  "summary": {
    "total_pnl": 45.00,
    "win_rate": 0.63,
    "total_trades": 1
  }
}
```

---

### Bot Control

#### `POST /bot/start`
Start the trading bot.

**Response** `200 OK`
```json
{"message": "Bot started", "pid": 12345}
```

#### `POST /bot/stop`
Gracefully stop the bot (closes no positions).

**Response** `200 OK`
```json
{"message": "Bot stopped"}
```

#### `POST /bot/pause`
Pause new signal processing (keeps positions open).

---

### Configuration

#### `GET /config`
Returns sanitized active configuration (secrets redacted).

#### `POST /config/reload`
Hot-reloads `config.yaml` without restarting.

**Response** `200 OK`
```json
{"message": "Configuration reloaded", "symbols": ["BTC/USDT", "ETH/USDT"]}
```

---

### Metrics

#### `GET /metrics`
Prometheus-format metrics endpoint.

```
# HELP trades_total Total number of trades executed
# TYPE trades_total counter
trades_total{symbol="BTC/USDT",side="buy"} 42
trades_total{symbol="ETH/USDT",side="sell"} 31
...
```

---

## Error Responses

All errors follow the format:

```json
{
  "error": "description of the error",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-15T12:00:00Z"
}
```

| HTTP Code | Meaning |
|-----------|---------|
| `400` | Bad request / invalid parameters |
| `401` | Missing or invalid API token |
| `403` | Forbidden (insufficient permissions) |
| `404` | Resource not found |
| `429` | Rate limit exceeded |
| `500` | Internal server error |

---

## WebSocket

Connect to `ws://localhost:8080/ws/feed` for real-time updates.

**Subscribe message:**
```json
{"subscribe": ["signals", "trades", "positions"]}
```

**Event message:**
```json
{
  "event": "signal",
  "data": {
    "symbol": "BTC/USDT",
    "action": "SELL",
    "confidence": 0.79,
    "timestamp": "2024-01-15T12:05:00Z"
  }
}
```
