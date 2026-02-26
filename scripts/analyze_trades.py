#!/usr/bin/env python3
"""
Trading Analysis Script
Analyzes historical trades and provides insights
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml import TradeAnalyzer, PerformanceAnalyzer, SignalScorer


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title):
    """Print a section header"""
    print(f"\n{title}")
    print("-" * 80)


def format_pnl(value):
    """Format P&L value with color"""
    if value > 0:
        return f"${value:,.2f}"
    elif value < 0:
        return f"${value:,.2f}"
    else:
        return f"${value:,.2f}"


def main():
    """Main analysis function"""
    print_header("TRADING BOT PERFORMANCE ANALYSIS")
    
    # Initialize analyzers
    trade_analyzer = TradeAnalyzer()
    perf_analyzer = PerformanceAnalyzer()
    signal_scorer = SignalScorer()
    
    # Overall Performance (Last 30 days)
    print_section("📊 OVERALL PERFORMANCE (Last 30 Days)")
    overall = trade_analyzer.analyze_performance(days=30)
    
    print(f"Total Trades: {overall['total_trades']}")
    print(f"Profitable Trades: {overall['profitable_trades']} ({overall['win_rate']:.1f}% win rate)")
    print(f"Losing Trades: {overall['losing_trades']}")
    print(f"Total P&L: {format_pnl(overall['total_pnl'])}")
    print(f"Average Profit per Win: {format_pnl(overall['avg_profit'])}")
    print(f"Average Loss per Loss: {format_pnl(overall['avg_loss'])}")
    print(f"Best Trade: {format_pnl(overall['best_trade'])}")
    print(f"Worst Trade: {format_pnl(overall['worst_trade'])}")
    
    if overall['profit_factor'] != float('inf'):
        print(f"Profit Factor: {overall['profit_factor']:.2f}")
    
    # Advanced Performance Metrics
    print_section("📈 ADVANCED METRICS")
    advanced = perf_analyzer.get_performance_summary(days=30)
    
    print(f"Sharpe Ratio (Annualized): {advanced['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {format_pnl(advanced['max_drawdown'])} ({advanced['max_drawdown_pct']:.1f}%)")
    print(f"Max Winning Streak: {advanced['max_win_streak']} trades")
    print(f"Max Losing Streak: {advanced['max_loss_streak']} trades")
    print(f"Current Streak: {advanced['current_streak']} {advanced['current_streak_type']} trades")
    
    # Performance by Symbol
    print_section("💰 PERFORMANCE BY TRADING PAIR")
    by_symbol = trade_analyzer.analyze_by_symbol(days=30)
    
    if by_symbol:
        print(f"{'Symbol':<15} {'Trades':<8} {'Win Rate':<12} {'Total P&L':<15} {'Avg P&L':<12}")
        print("-" * 80)
        for symbol, stats in list(by_symbol.items())[:10]:  # Top 10
            print(f"{symbol:<15} {stats['total_trades']:<8} {stats['win_rate']:>6.1f}%     {format_pnl(stats['total_pnl']):<15} {format_pnl(stats['avg_pnl']):<12}")
    else:
        print("No trading data available")
    
    # Performance by Strategy
    print_section("🎯 PERFORMANCE BY STRATEGY")
    by_strategy = trade_analyzer.analyze_by_strategy(days=30)
    
    if by_strategy:
        for strategy, stats in by_strategy.items():
            print(f"\n{strategy}:")
            print(f"  Total Trades: {stats['total_trades']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Total P&L: {format_pnl(stats['total_pnl'])}")
            print(f"  Average P&L: {format_pnl(stats['avg_pnl'])}")
    else:
        print("No strategy data available")
    
    # Pattern Analysis
    print_section("🔍 PROFITABLE TRADE PATTERNS")
    patterns = trade_analyzer.find_common_patterns_in_profitable_trades()
    
    if patterns:
        print(f"Total Profitable Trades: {patterns['total_profitable']}")
        print(f"Average Holding Time: {patterns['avg_holding_time_hours']:.1f} hours")
        print(f"Long Trades: {patterns['long_trades']} (Avg: {format_pnl(patterns['avg_profit_long'])})")
        print(f"Short Trades: {patterns['short_trades']} (Avg: {format_pnl(patterns['avg_profit_short'])})")
    else:
        print("Not enough data for pattern analysis")
    
    # Best Performing Pairs
    print_section("⭐ TOP 5 BEST PERFORMING PAIRS")
    best_pairs = signal_scorer.get_best_performing_pairs(limit=5, days=30)
    
    if best_pairs:
        for pair in best_pairs:
            print(f"\n{pair['symbol']}:")
            print(f"  Trades: {pair['trades']}, Win Rate: {pair['win_rate']:.1f}%")
            print(f"  Total P&L: {format_pnl(pair['total_pnl'])}, Avg P&L: {format_pnl(pair['avg_pnl'])}")
    else:
        print("Not enough data")
    
    # Worst Performing Pairs
    print_section("⚠️  TOP 5 WORST PERFORMING PAIRS")
    worst_pairs = signal_scorer.get_worst_performing_pairs(limit=5, days=30)
    
    if worst_pairs:
        for pair in worst_pairs:
            print(f"\n{pair['symbol']}:")
            print(f"  Trades: {pair['trades']}, Win Rate: {pair['win_rate']:.1f}%")
            print(f"  Total P&L: {format_pnl(pair['total_pnl'])}, Avg P&L: {format_pnl(pair['avg_pnl'])}")
    else:
        print("Not enough data")
    
    # Recommendations
    print_section("💡 RECOMMENDATIONS")
    recommendations = trade_analyzer.get_recommendations()
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
