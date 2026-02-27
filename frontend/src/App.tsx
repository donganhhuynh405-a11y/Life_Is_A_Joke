import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Activity,
  BarChart2,
  TrendingUp,
  LineChart,
} from 'lucide-react';

// Lazy-loaded pages
const Dashboard = lazy(() => import('./components/Dashboard'));
const StrategyManager = lazy(() => import('./components/StrategyManager'));
const BacktestRunner = lazy(() => import('./components/BacktestRunner'));
const PortfolioView = lazy(() => import('./components/PortfolioView'));
const RealTimeChart = lazy(() => import('./components/RealTimeChart'));

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/portfolio', label: 'Portfolio', icon: BarChart2 },
  { to: '/strategies', label: 'Strategies', icon: Activity },
  { to: '/backtest', label: 'Backtest', icon: TrendingUp },
  { to: '/charts', label: 'Charts', icon: LineChart },
];

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>🤖 Trading Bot</h1>
        <span className="badge badge-paper">Paper</span>
      </div>
      <nav>
        {navItems.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `nav-item ${isActive ? 'nav-item--active' : ''}`
            }
          >
            <Icon size={18} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">
        <span className="text-muted">v1.3.0</span>
      </div>
    </aside>
  );
}

function LoadingFallback() {
  return (
    <div className="loading">
      <div className="spinner" />
      <span>Loading...</span>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/portfolio" element={<PortfolioView />} />
              <Route path="/strategies" element={<StrategyManager />} />
              <Route path="/backtest" element={<BacktestRunner />} />
              <Route path="/charts" element={<RealTimeChart symbol="BTC/USDT" />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </BrowserRouter>
  );
}
