/**
 * StrategyManager.tsx - Strategy management UI component.
 */

import React, { useEffect, useState } from 'react';
import { Play, Pause, Square, Settings2 } from 'lucide-react';
import api, { StrategyInfo } from '../api/client';

const STATUS_COLORS: Record<string, string> = {
  active: 'badge-success',
  paused: 'badge-warning',
  stopped: 'badge-danger',
};

const STATUS_ICONS: Record<string, React.ReactNode> = {
  active: <Play size={14} />,
  paused: <Pause size={14} />,
  stopped: <Square size={14} />,
};

function StrategyCard({ strategy, onUpdate }: {
  strategy: StrategyInfo;
  onUpdate: (name: string, status: string) => void;
}) {
  const [updating, setUpdating] = useState(false);

  async function setStatus(newStatus: string) {
    setUpdating(true);
    try {
      await onUpdate(strategy.name, newStatus);
    } finally {
      setUpdating(false);
    }
  }

  return (
    <div className="card strategy-card">
      <div className="strategy-header">
        <div className="strategy-title">
          <h4>{strategy.name}</h4>
          <span className={`badge ${STATUS_COLORS[strategy.status] ?? 'badge-neutral'}`}>
            {STATUS_ICONS[strategy.status]}
            {strategy.status}
          </span>
        </div>
        <div className="strategy-actions">
          {strategy.status !== 'active' && (
            <button
              className="btn btn-sm btn-success"
              onClick={() => setStatus('active')}
              disabled={updating}
              title="Activate"
            >
              <Play size={14} /> Activate
            </button>
          )}
          {strategy.status === 'active' && (
            <button
              className="btn btn-sm btn-warning"
              onClick={() => setStatus('paused')}
              disabled={updating}
              title="Pause"
            >
              <Pause size={14} /> Pause
            </button>
          )}
          {strategy.status !== 'stopped' && (
            <button
              className="btn btn-sm btn-danger"
              onClick={() => setStatus('stopped')}
              disabled={updating}
              title="Stop"
            >
              <Square size={14} /> Stop
            </button>
          )}
        </div>
      </div>
      {strategy.description && (
        <p className="strategy-description text-muted">{strategy.description}</p>
      )}
      <div className="strategy-meta">
        <span>Total signals: <strong>{strategy.total_signals}</strong></span>
        {strategy.parameters && (
          <details>
            <summary><Settings2 size={14} /> Parameters</summary>
            <pre className="code-block">{JSON.stringify(strategy.parameters, null, 2)}</pre>
          </details>
        )}
      </div>
    </div>
  );
}

export default function StrategyManager() {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<string | null>(null);

  async function loadStrategies() {
    setLoading(true);
    try {
      const data = await api.getStrategies();
      setStrategies(data.strategies);
    } catch (e) {
      setMessage(`Failed to load strategies: ${e}`);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpdate(name: string, status: string) {
    try {
      await api.updateStrategy(name, { status });
      setMessage(`Strategy "${name}" updated to ${status}`);
      await loadStrategies();
    } catch (e) {
      setMessage(`Failed to update strategy: ${e}`);
    }
  }

  useEffect(() => { loadStrategies(); }, []);

  if (loading) {
    return <div className="loading"><div className="spinner" /><span>Loading strategies...</span></div>;
  }

  return (
    <div className="strategy-manager">
      <div className="page-header">
        <h2>Strategy Manager</h2>
        <button className="btn btn-secondary" onClick={loadStrategies}>Refresh</button>
      </div>

      {message && (
        <div className="alert alert-info">
          {message}
          <button className="btn-close" onClick={() => setMessage(null)}>×</button>
        </div>
      )}

      <div className="strategies-grid">
        {strategies.map((s) => (
          <StrategyCard key={s.name} strategy={s} onUpdate={handleUpdate} />
        ))}
        {strategies.length === 0 && (
          <p className="text-muted">No strategies configured.</p>
        )}
      </div>
    </div>
  );
}
