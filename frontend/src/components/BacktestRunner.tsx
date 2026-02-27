/**
 * BacktestRunner.tsx - Backtesting interface component.
 */

import React, { useState } from 'react';
import { Play, Loader } from 'lucide-react';
import api, { BacktestRequest, BacktestResult } from '../api/client';

const STRATEGIES = ['SimpleTrend', 'EnhancedMultiIndicator'];
const SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'];
const TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d'];

function MetricRow({ label, value, format = 'text' }: {
  label: string;
  value: number | null | undefined;
  format?: 'text' | 'percent' | 'currency' | 'ratio';
}) {
  if (value === null || value === undefined) return null;
  let display = String(value);
  if (format === 'percent') display = `${(value * 100).toFixed(2)}%`;
  else if (format === 'currency') display = `$${value.toFixed(2)}`;
  else if (format === 'ratio') display = value.toFixed(3);

  return (
    <div className="metric-row">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{display}</span>
    </div>
  );
}

export default function BacktestRunner() {
  const [form, setForm] = useState<BacktestRequest>({
    strategy: STRATEGIES[0],
    symbol: SYMBOLS[0],
    timeframe: '1h',
    start_date: '2023-01-01T00:00:00',
    end_date: '2024-01-01T00:00:00',
    initial_capital: 10000,
  });
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleChange(field: keyof BacktestRequest, value: string | number) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function handleRun() {
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.runBacktest(form);
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="backtest-runner">
      <div className="page-header">
        <h2>Backtest Runner</h2>
      </div>

      <div className="backtest-layout">
        {/* Form */}
        <div className="card backtest-form">
          <h3>Configuration</h3>

          <label className="form-field">
            <span>Strategy</span>
            <select value={form.strategy} onChange={(e) => handleChange('strategy', e.target.value)}>
              {STRATEGIES.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>

          <label className="form-field">
            <span>Symbol</span>
            <select value={form.symbol} onChange={(e) => handleChange('symbol', e.target.value)}>
              {SYMBOLS.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>

          <label className="form-field">
            <span>Timeframe</span>
            <select value={form.timeframe} onChange={(e) => handleChange('timeframe', e.target.value)}>
              {TIMEFRAMES.map((t) => <option key={t}>{t}</option>)}
            </select>
          </label>

          <label className="form-field">
            <span>Start Date</span>
            <input
              type="datetime-local"
              value={form.start_date.slice(0, 16)}
              onChange={(e) => handleChange('start_date', e.target.value + ':00')}
            />
          </label>

          <label className="form-field">
            <span>End Date</span>
            <input
              type="datetime-local"
              value={form.end_date.slice(0, 16)}
              onChange={(e) => handleChange('end_date', e.target.value + ':00')}
            />
          </label>

          <label className="form-field">
            <span>Initial Capital (USDT)</span>
            <input
              type="number"
              min={100}
              step={100}
              value={form.initial_capital}
              onChange={(e) => handleChange('initial_capital', Number(e.target.value))}
            />
          </label>

          <button
            className="btn btn-primary"
            onClick={handleRun}
            disabled={running}
          >
            {running ? (
              <><Loader size={16} className="spin" /> Running...</>
            ) : (
              <><Play size={16} /> Run Backtest</>
            )}
          </button>
        </div>

        {/* Results */}
        <div className="card backtest-results">
          <h3>Results</h3>
          {error && <div className="alert alert-danger">{error}</div>}
          {!result && !error && !running && (
            <p className="text-muted">Configure and run a backtest to see results.</p>
          )}
          {running && (
            <div className="loading"><div className="spinner" /><span>Running backtest...</span></div>
          )}
          {result && (
            <div className="metrics">
              <div className="metric-row">
                <span className="metric-label">Strategy</span>
                <span className="metric-value">{result.strategy}</span>
              </div>
              <div className="metric-row">
                <span className="metric-label">Symbol</span>
                <span className="metric-value">{result.symbol}</span>
              </div>
              <MetricRow label="Total Return" value={result.total_return} format="percent" />
              <MetricRow label="Sharpe Ratio" value={result.sharpe_ratio} format="ratio" />
              <MetricRow label="Max Drawdown" value={result.max_drawdown} format="percent" />
              <MetricRow label="Total Trades" value={result.total_trades} />
              <MetricRow label="Win Rate" value={result.win_rate} format="percent" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
