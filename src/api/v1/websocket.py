"""
src/api/v1/websocket.py - WebSocket endpoints for real-time data streaming.

Provides real-time:
  - Price ticker updates
  - Trading signals
  - Portfolio updates
  - Trade execution notifications
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Set

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
    from fastapi.websockets import WebSocketState
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

logger = logging.getLogger(__name__)

ws_router = APIRouter() if HAS_FASTAPI else None


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        # Maps topic -> set of websocket connections
        self._connections: Dict[str, Set] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: Any, topic: str) -> None:
        """Accept a new WebSocket connection for a topic."""
        await websocket.accept()
        async with self._lock:
            if topic not in self._connections:
                self._connections[topic] = set()
            self._connections[topic].add(websocket)
        logger.info("WS client connected: topic=%s total=%d",
                    topic, len(self._connections.get(topic, set())))

    async def disconnect(self, websocket: Any, topic: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if topic in self._connections:
                self._connections[topic].discard(websocket)
                if not self._connections[topic]:
                    del self._connections[topic]
        logger.info("WS client disconnected: topic=%s", topic)

    async def broadcast(self, topic: str, data: Dict[str, Any]) -> None:
        """Broadcast a message to all clients subscribed to a topic."""
        if topic not in self._connections:
            return

        message = json.dumps({**data, "timestamp": datetime.utcnow().isoformat()})
        dead: Set = set()

        for ws in list(self._connections.get(topic, set())):
            try:
                if HAS_FASTAPI and ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(message)
            except Exception:
                dead.add(ws)

        # Cleanup dead connections
        if dead:
            async with self._lock:
                if topic in self._connections:
                    self._connections[topic] -= dead

    async def broadcast_all(self, data: Dict[str, Any]) -> None:
        """Broadcast to all connected clients across all topics."""
        for topic in list(self._connections.keys()):
            await self.broadcast(topic, data)

    @property
    def total_connections(self) -> int:
        return sum(len(v) for v in self._connections.values())


# Global connection manager
manager = ConnectionManager()


if HAS_FASTAPI and ws_router is not None:

    @ws_router.websocket("/ws/ticker/{symbol}")
    async def ticker_websocket(websocket: WebSocket, symbol: str) -> None:
        """
        WebSocket endpoint for real-time price ticker updates.

        Messages format:
        ```json
        {
          "type": "ticker",
          "symbol": "BTC/USDT",
          "bid": 45000.0,
          "ask": 45001.0,
          "last": 45000.5,
          "volume": 1234.5,
          "timestamp": "2024-01-01T00:00:00"
        }
        ```
        """
        topic = f"ticker:{symbol}"
        await manager.connect(websocket, topic)
        try:
            while True:
                # Send heartbeat / wait for client messages
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    if data == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                    }))
        except WebSocketDisconnect:
            pass
        finally:
            await manager.disconnect(websocket, topic)

    @ws_router.websocket("/ws/signals")
    async def signals_websocket(websocket: WebSocket) -> None:
        """
        WebSocket endpoint for real-time trading signals.

        Messages format:
        ```json
        {
          "type": "signal",
          "symbol": "BTC/USDT",
          "action": "BUY",
          "confidence": 0.85,
          "strategy": "EnhancedMultiIndicator",
          "timestamp": "2024-01-01T00:00:00"
        }
        ```
        """
        topic = "signals"
        await manager.connect(websocket, topic)
        try:
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    if data == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                    }))
        except WebSocketDisconnect:
            pass
        finally:
            await manager.disconnect(websocket, topic)

    @ws_router.websocket("/ws/portfolio")
    async def portfolio_websocket(websocket: WebSocket) -> None:
        """
        WebSocket endpoint for real-time portfolio updates.

        Messages format:
        ```json
        {
          "type": "portfolio_update",
          "total_value": 12500.0,
          "unrealized_pnl": 250.0,
          "positions_count": 3,
          "timestamp": "2024-01-01T00:00:00"
        }
        ```
        """
        topic = "portfolio"
        await manager.connect(websocket, topic)
        try:
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    if data == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                    }))
        except WebSocketDisconnect:
            pass
        finally:
            await manager.disconnect(websocket, topic)

    @ws_router.websocket("/ws/trades")
    async def trades_websocket(websocket: WebSocket) -> None:
        """
        WebSocket endpoint for real-time trade execution notifications.

        Messages format:
        ```json
        {
          "type": "trade",
          "symbol": "BTC/USDT",
          "side": "buy",
          "amount": 0.01,
          "price": 45000.0,
          "order_id": "ABC123",
          "timestamp": "2024-01-01T00:00:00"
        }
        ```
        """
        topic = "trades"
        await manager.connect(websocket, topic)
        try:
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    if data == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                    }))
        except WebSocketDisconnect:
            pass
        finally:
            await manager.disconnect(websocket, topic)


async def notify_signal(signal: Dict[str, Any]) -> None:
    """Broadcast a trading signal to all signal subscribers."""
    await manager.broadcast("signals", {"type": "signal", **signal})


async def notify_trade(trade: Dict[str, Any]) -> None:
    """Broadcast a trade execution to all trade subscribers."""
    await manager.broadcast("trades", {"type": "trade", **trade})


async def notify_portfolio_update(update: Dict[str, Any]) -> None:
    """Broadcast a portfolio update to all portfolio subscribers."""
    await manager.broadcast("portfolio", {"type": "portfolio_update", **update})


async def notify_ticker(symbol: str, ticker: Dict[str, Any]) -> None:
    """Broadcast ticker data to symbol-specific subscribers."""
    await manager.broadcast(f"ticker:{symbol}", {"type": "ticker", "symbol": symbol, **ticker})
