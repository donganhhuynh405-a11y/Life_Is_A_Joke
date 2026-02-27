/**
 * API client for trading bot backend.
 * Wraps axios with base URL, auth headers, and typed response helpers.
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8001/api/v1';

export const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 15_000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor – attach auth token if present
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor – normalise errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.error ??
      error.response?.data?.detail ??
      error.message ??
      'Unknown error';
    console.error('[API Error]', message);
    return Promise.reject(new Error(message));
  },
);

// ---------------------------------------------------------------------------
// Typed API helpers
// ---------------------------------------------------------------------------

export interface HealthStatus {
  status: string;
  version: string;
  environment: string;
  timestamp: string;
  services: Record<string, string>;
}

export interface TickerData {
  symbol: string;
  bid: number | null;
  ask: number | null;
  last: number | null;
  volume: number | null;
  timestamp: string;
}

export interface TradingSignal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD' | 'CLOSE';
  confidence: number;
  price: number | null;
  reason: string | null;
  strategy: string | null;
  timestamp: string;
}

export interface PortfolioSummary {
  total_value_usdt: number;
  cash_usdt: number;
  unrealized_pnl: number;
  realized_pnl: number;
  total_pnl: number;
  positions_count: number;
}

export interface StrategyInfo {
  name: string;
  status: 'active' | 'paused' | 'stopped';
  description: string | null;
  parameters: Record<string, unknown> | null;
  total_signals: number;
}

export interface BacktestRequest {
  strategy: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
}

export interface BacktestResult {
  strategy: string;
  symbol: string;
  total_return: number;
  sharpe_ratio: number | null;
  max_drawdown: number | null;
  total_trades: number;
  win_rate: number | null;
}

export interface SentimentResult {
  score: number;
  sentiment: string;
  fomo: boolean;
  fud: boolean;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export const api = {
  getHealth: () => apiClient.get<HealthStatus>('/health').then((r) => r.data),

  getTicker: (symbol: string) =>
    apiClient.get<TickerData>(`/market/${encodeURIComponent(symbol)}/ticker`).then((r) => r.data),

  getSignals: (params?: { symbol?: string; strategy?: string; limit?: number }) =>
    apiClient
      .get<{ signals: TradingSignal[]; count: number }>('/signals', { params })
      .then((r) => r.data),

  getPortfolio: () =>
    apiClient
      .get<{ summary: PortfolioSummary; allocations: unknown[] }>('/portfolio')
      .then((r) => r.data),

  getStrategies: () =>
    apiClient
      .get<{ strategies: StrategyInfo[]; count: number }>('/strategies')
      .then((r) => r.data),

  updateStrategy: (name: string, body: { status?: string; parameters?: Record<string, unknown> }) =>
    apiClient.patch(`/strategies/${encodeURIComponent(name)}`, body).then((r) => r.data),

  runBacktest: (request: BacktestRequest) =>
    apiClient.post<BacktestResult>('/backtest', request).then((r) => r.data),

  analyzeSentiment: (texts: string[]) =>
    apiClient
      .post<{ results: SentimentResult; text_count: number }>('/sentiment/analyze', { texts })
      .then((r) => r.data),

  getPerformance: (period = '7d', strategy?: string) =>
    apiClient.get('/performance', { params: { period, strategy } }).then((r) => r.data),

  getOHLCV: (symbol: string, timeframe = '1h', limit = 100) =>
    apiClient
      .get(`/market/${encodeURIComponent(symbol)}/ohlcv`, { params: { timeframe, limit } })
      .then((r) => r.data),
};

export default api;
