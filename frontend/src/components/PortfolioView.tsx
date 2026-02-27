/**
 * PortfolioView.tsx - Portfolio visualization component.
 */

import React, { useEffect, useState } from 'react';
import { TrendingUp, TrendingDown, Briefcase } from 'lucide-react';
import api from '../api/client';

interface AssetAllocation {
  symbol: string;
  amount: number;
  value_usdt: number;
  weight: number;
  unrealized_pnl: number;
}

interface PortfolioSummary {
  total_value_usdt: number;
  cash_usdt: number;
  unrealized_pnl: number;
  realized_pnl: number;
  total_pnl: number;
  positions_count: number;
}

function AllocationBar({ weight }: { weight: number }) {
  return (
    <div className="allocation-bar-container">
      <div
        className="allocation-bar"
        style={{ width: `${(weight * 100).toFixed(1)}%` }}
      />
      <span>{(weight * 100).toFixed(1)}%</span>
    </div>
  );
}

function AssetRow({ asset }: { asset: AssetAllocation }) {
  const positive = asset.unrealized_pnl >= 0;
  return (
    <tr className="asset-row">
      <td><strong>{asset.symbol}</strong></td>
      <td>{asset.amount.toFixed(6)}</td>
      <td>${asset.value_usdt.toFixed(2)}</td>
      <td><AllocationBar weight={asset.weight} /></td>
      <td className={positive ? 'text-success' : 'text-danger'}>
        {positive ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
        {positive ? '+' : ''}${asset.unrealized_pnl.toFixed(2)}
      </td>
    </tr>
  );
}

export default function PortfolioView() {
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [allocations, setAllocations] = useState<AssetAllocation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getPortfolio();
      setSummary(data.summary);
      setAllocations(data.allocations as AssetAllocation[]);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const interval = setInterval(load, 15_000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !summary) {
    return <div className="loading"><div className="spinner" /><span>Loading portfolio...</span></div>;
  }

  const totalPnl = summary?.total_pnl ?? 0;
  const pnlPositive = totalPnl >= 0;

  return (
    <div className="portfolio-view">
      <div className="page-header">
        <h2><Briefcase size={20} /> Portfolio</h2>
        <button className="btn btn-secondary" onClick={load} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && <div className="alert alert-warning">{error}</div>}

      {/* Summary cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <span className="stat-label">Total Value</span>
          <span className="stat-value">
            ${(summary?.total_value_usdt ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Total P&L</span>
          <span className={`stat-value ${pnlPositive ? 'positive' : 'negative'}`}>
            {pnlPositive ? '+' : ''}${totalPnl.toFixed(2)}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Unrealized P&L</span>
          <span className={`stat-value ${(summary?.unrealized_pnl ?? 0) >= 0 ? 'positive' : 'negative'}`}>
            ${(summary?.unrealized_pnl ?? 0).toFixed(2)}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Cash Available</span>
          <span className="stat-value">${(summary?.cash_usdt ?? 0).toFixed(2)}</span>
        </div>
      </div>

      {/* Allocations table */}
      <div className="card mt-4">
        <h3>Asset Allocations ({summary?.positions_count ?? 0} positions)</h3>
        {allocations.length === 0 ? (
          <p className="text-muted">No open positions.</p>
        ) : (
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Amount</th>
                  <th>Value (USDT)</th>
                  <th>Allocation</th>
                  <th>Unrealized P&L</th>
                </tr>
              </thead>
              <tbody>
                {allocations.map((a) => (
                  <AssetRow key={a.symbol} asset={a} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
