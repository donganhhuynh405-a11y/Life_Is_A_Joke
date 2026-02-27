/**
 * Dashboard.tsx - Main trading dashboard component.
 *
 * Shows real-time overview: portfolio value, P&L, active signals, health status.
 */

import React, { useEffect, useState } from 'react';
import { TrendingUp, TrendingDown, Activity, AlertCircle, CheckCircle } from 'lucide-react';
import api, { HealthStatus, PortfolioSummary, TradingSignal } from '../api/client';

function StatCard({
  label,
  value,
  subtext,
  positive,
}: {
  label: string;
  value: string;
  subtext?: string;
  positive?: boolean;
}) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className={`stat-value ${positive === true ? 'positive' : positive === false ? 'negative' : ''}`}>
        {value}
      </span>
      {subtext && <span className="stat-sub">{subtext}</span>}
    </div>
  );
}

function ServiceBadge({ name, status }: { name: string; status: string }) {
  const ok = status === 'ok';
  return (
    <div className="service-badge">
      {ok ? (
        <CheckCircle size={14} className="text-success" />
      ) : (
        <AlertCircle size={14} className="text-warning" />
      )}
      <span>{name}</span>
      <span className={`badge ${ok ? 'badge-success' : 'badge-warning'}`}>{status}</span>
    </div>
  );
}

function SignalRow({ signal }: { signal: TradingSignal }) {
  const isBuy = signal.action === 'BUY';
  const isSell = signal.action === 'SELL';
  return (
    <div className="signal-row">
      {isBuy ? (
        <TrendingUp size={16} className="text-success" />
      ) : isSell ? (
        <TrendingDown size={16} className="text-danger" />
      ) : (
        <Activity size={16} className="text-muted" />
      )}
      <span className="signal-symbol">{signal.symbol}</span>
      <span className={`badge ${isBuy ? 'badge-success' : isSell ? 'badge-danger' : 'badge-neutral'}`}>
        {signal.action}
      </span>
      <span className="signal-confidence">{(signal.confidence * 100).toFixed(0)}%</span>
      <span className="signal-strategy text-muted">{signal.strategy ?? '—'}</span>
    </div>
  );
}

export default function Dashboard() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [h, p, s] = await Promise.allSettled([
        api.getHealth(),
        api.getPortfolio(),
        api.getSignals({ limit: 10 }),
      ]);

      if (h.status === 'fulfilled') setHealth(h.value);
      if (p.status === 'fulfilled') setPortfolio(p.value.summary);
      if (s.status === 'fulfilled') setSignals(s.value.signals);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const interval = setInterval(load, 30_000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !health) {
    return <div className="loading"><div className="spinner" /><span>Loading dashboard...</span></div>;
  }

  const pnl = portfolio?.total_pnl ?? 0;
  const pnlPositive = pnl >= 0;

  return (
    <div className="dashboard">
      <div className="page-header">
        <h2>Dashboard</h2>
        <button className="btn btn-secondary" onClick={load} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="alert alert-warning">
          <AlertCircle size={16} />
          <span>API unavailable – showing cached data. {error}</span>
        </div>
      )}

      {/* Stats grid */}
      <div className="stats-grid">
        <StatCard
          label="Portfolio Value"
          value={`$${(portfolio?.total_value_usdt ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}`}
          subtext={`${portfolio?.positions_count ?? 0} open positions`}
        />
        <StatCard
          label="Total P&L"
          value={`${pnlPositive ? '+' : ''}$${pnl.toFixed(2)}`}
          subtext="All time"
          positive={pnlPositive}
        />
        <StatCard
          label="Unrealized P&L"
          value={`$${(portfolio?.unrealized_pnl ?? 0).toFixed(2)}`}
          positive={(portfolio?.unrealized_pnl ?? 0) >= 0}
        />
        <StatCard
          label="Cash"
          value={`$${(portfolio?.cash_usdt ?? 0).toFixed(2)}`}
        />
      </div>

      <div className="dashboard-grid">
        {/* Services health */}
        <div className="card">
          <h3>Service Health</h3>
          <div className="services-list">
            <ServiceBadge name="API" status={health?.status ?? 'unknown'} />
            {health?.services &&
              Object.entries(health.services).map(([name, status]) => (
                <ServiceBadge key={name} name={name} status={status} />
              ))}
          </div>
          <div className="text-muted text-sm mt-2">
            Env: {health?.environment ?? '—'} · v{health?.version ?? '?'}
          </div>
        </div>

        {/* Recent signals */}
        <div className="card">
          <h3>Recent Signals</h3>
          {signals.length === 0 ? (
            <p className="text-muted">No signals yet.</p>
          ) : (
            <div className="signals-list">
              {signals.map((s, i) => (
                <SignalRow key={i} signal={s} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
