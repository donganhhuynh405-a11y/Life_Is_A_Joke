/**
 * RealTimeChart.tsx - Real-time price chart using Chart.js.
 */

import React, { useEffect, useRef, useState } from 'react';
import { Wifi, WifiOff } from 'lucide-react';
import api from '../api/client';
import { useWebSocket } from '../hooks/useWebSocket';

interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface RealTimeChartProps {
  symbol?: string;
  timeframe?: string;
}

const SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'];
const TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d'];

export default function RealTimeChart({
  symbol: initialSymbol = 'BTC/USDT',
  timeframe: initialTimeframe = '1h',
}: RealTimeChartProps) {
  const [symbol, setSymbol] = useState(initialSymbol);
  const [timeframe, setTimeframe] = useState(initialTimeframe);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [lastPrice, setLastPrice] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<unknown>(null);

  // WebSocket for real-time price updates
  const { status: wsStatus } = useWebSocket(
    `ticker/${encodeURIComponent(symbol)}`,
    {
      onMessage: (data: unknown) => {
        const d = data as Record<string, unknown>;
        if (d.type === 'ticker' && typeof d.last === 'number') {
          setLastPrice(d.last);
        }
      },
    },
  );

  async function loadOHLCV() {
    setLoading(true);
    try {
      const data = await api.getOHLCV(symbol, timeframe, 100);
      setCandles(data.candles ?? []);
    } catch {
      setCandles([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadOHLCV(); }, [symbol, timeframe]);

  // Render simple line chart with Chart.js if available
  useEffect(() => {
    if (!chartRef.current || candles.length === 0) return;

    const loadChart = async () => {
      try {
        // Dynamic import to avoid build-time dependency
        const { Chart, registerables } = await import('chart.js');
        Chart.register(...registerables);

        if (chartInstanceRef.current) {
          (chartInstanceRef.current as { destroy: () => void }).destroy();
        }

        const ctx = chartRef.current!.getContext('2d');
        if (!ctx) return;

        chartInstanceRef.current = new Chart(ctx, {
          type: 'line',
          data: {
            labels: candles.map((c) =>
              new Date(c.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            ),
            datasets: [
              {
                label: `${symbol} Close`,
                data: candles.map((c) => c.close),
                borderColor: 'rgb(99, 102, 241)',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { ticks: { maxTicksLimit: 10 } },
              y: { ticks: { callback: (v) => `$${Number(v).toLocaleString()}` } },
            },
          },
        });
      } catch {
        // Chart.js not available - skip
      }
    };

    loadChart();

    return () => {
      if (chartInstanceRef.current) {
        (chartInstanceRef.current as { destroy: () => void }).destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [candles, symbol]);

  const isConnected = wsStatus === 'open';

  return (
    <div className="realtime-chart">
      <div className="page-header">
        <h2>Real-Time Charts</h2>
        <div className="chart-controls">
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {SYMBOLS.map((s) => <option key={s}>{s}</option>)}
          </select>
          <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
            {TIMEFRAMES.map((t) => <option key={t}>{t}</option>)}
          </select>
          <button className="btn btn-secondary" onClick={loadOHLCV} disabled={loading}>
            Reload
          </button>
        </div>
      </div>

      {/* Header bar */}
      <div className="chart-header-bar">
        <div className="chart-title">
          <h3>{symbol}</h3>
          {lastPrice !== null && (
            <span className="live-price">${lastPrice.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
          )}
        </div>
        <div className="ws-status">
          {isConnected ? (
            <><Wifi size={14} className="text-success" /> Live</>
          ) : (
            <><WifiOff size={14} className="text-muted" /> {wsStatus}</>
          )}
        </div>
      </div>

      {/* Chart canvas */}
      <div className="chart-container">
        {loading ? (
          <div className="loading"><div className="spinner" /><span>Loading chart data...</span></div>
        ) : candles.length === 0 ? (
          <div className="chart-placeholder">
            <p className="text-muted">No chart data available from API.</p>
            <p className="text-muted text-sm">Connect to a live exchange to see price data.</p>
          </div>
        ) : (
          <canvas ref={chartRef} style={{ width: '100%', height: '400px' }} />
        )}
      </div>
    </div>
  );
}
