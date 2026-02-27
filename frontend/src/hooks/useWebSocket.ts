/**
 * useWebSocket.ts - React hook for WebSocket connections with auto-reconnect.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

const WS_BASE = (import.meta.env.VITE_WS_URL ?? 'ws://localhost:8001') + '/api/v1/ws';

export type WsStatus = 'connecting' | 'open' | 'closed' | 'error';

export interface UseWebSocketOptions {
  /** Called when a message is received */
  onMessage?: (data: unknown) => void;
  /** Auto reconnect on disconnect (default: true) */
  autoReconnect?: boolean;
  /** Delay between reconnect attempts in ms (default: 3000) */
  reconnectDelay?: number;
  /** Max reconnect attempts (default: 10) */
  maxRetries?: number;
}

export interface UseWebSocketReturn {
  status: WsStatus;
  send: (data: string | object) => void;
  disconnect: () => void;
  reconnect: () => void;
}

export function useWebSocket(
  path: string,
  options: UseWebSocketOptions = {},
): UseWebSocketReturn {
  const {
    onMessage,
    autoReconnect = true,
    reconnectDelay = 3000,
    maxRetries = 10,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const retryCountRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [status, setStatus] = useState<WsStatus>('connecting');

  const connect = useCallback(() => {
    const url = `${WS_BASE}/${path}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    setStatus('connecting');

    ws.onopen = () => {
      setStatus('open');
      retryCountRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage?.(data);
      } catch {
        onMessage?.(event.data);
      }
    };

    ws.onerror = () => {
      setStatus('error');
    };

    ws.onclose = () => {
      setStatus('closed');
      if (autoReconnect && retryCountRef.current < maxRetries) {
        retryCountRef.current += 1;
        reconnectTimerRef.current = setTimeout(connect, reconnectDelay);
      }
    };
  }, [path, onMessage, autoReconnect, reconnectDelay, maxRetries]);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }
    wsRef.current?.close();
    wsRef.current = null;
    setStatus('closed');
  }, []);

  const reconnect = useCallback(() => {
    disconnect();
    retryCountRef.current = 0;
    connect();
  }, [connect, disconnect]);

  const send = useCallback((data: string | object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const payload = typeof data === 'string' ? data : JSON.stringify(data);
      wsRef.current.send(payload);
    } else {
      console.warn('[useWebSocket] Cannot send – socket not open');
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { status, send, disconnect, reconnect };
}

export default useWebSocket;
