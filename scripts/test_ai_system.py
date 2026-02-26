#!/usr/bin/env python3
"""
AI System Diagnostic Tool
Tests all AI components to ensure they are working correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.ai_commentary import AICommentaryGenerator
from src.ml.adaptive_tactics import AdaptiveTacticsManager
from src.ml import TradeAnalyzer, PerformanceAnalyzer, SignalScorer
from src.core.config import Config
from src.core.database import Database


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_test(name, status, details=""):
    """Print test result"""
    emoji = "✅" if status else "❌"
    print(f"{emoji} {name}")
    if details:
        print(f"   {details}")


def test_ml_analyzers():
    """Test ML analyzer components"""
    print_header("TESTING ML ANALYZERS")
    
    config = Config()
    db_path = config.db_path
    
    # Test TradeAnalyzer
    try:
        analyzer = TradeAnalyzer(db_path=db_path)
        stats = analyzer.analyze_performance(days=7)
        print_test("TradeAnalyzer", True, 
                   f"Analyzed {stats.get('total_trades', 0)} trades, "
                   f"{stats.get('win_rate', 0):.1f}% win rate")
    except Exception as e:
        print_test("TradeAnalyzer", False, f"Error: {e}")
        return False
    
    # Test PerformanceAnalyzer
    try:
        perf_analyzer = PerformanceAnalyzer(db_path=db_path)
        summary = perf_analyzer.get_performance_summary(days=7)
        print_test("PerformanceAnalyzer", True,
                   f"Sharpe: {summary.get('sharpe_ratio', 0):.2f}, "
                   f"Drawdown: {summary.get('max_drawdown_pct', 0):.1f}%")
    except Exception as e:
        print_test("PerformanceAnalyzer", False, f"Error: {e}")
        return False
    
    # Test SignalScorer
    try:
        scorer = SignalScorer(db_path=db_path)
        score = scorer.score_signal('BTCUSDT', 'BUY', confidence=0.75)
        print_test("SignalScorer", True,
                   f"Score: {score.get('score', 0)}/100, "
                   f"Recommendation: {score.get('recommendation', 'N/A')}")
    except Exception as e:
        print_test("SignalScorer", False, f"Error: {e}")
        return False
    
    # Test get_symbol_stats
    try:
        symbol_stats = scorer.get_symbol_stats('BTCUSDT', days=30)
        print_test("SignalScorer.get_symbol_stats", True,
                   f"Trades: {symbol_stats.get('total_trades', 0)}, "
                   f"Win rate: {symbol_stats.get('win_rate', 0):.1f}%")
    except Exception as e:
        print_test("SignalScorer.get_symbol_stats", False, f"Error: {e}")
        return False
    
    # Test get_side_stats
    try:
        side_stats = scorer.get_side_stats('BTCUSDT', 'BUY', days=30)
        print_test("SignalScorer.get_side_stats", True,
                   f"Trades: {side_stats.get('trades', 0)}, "
                   f"Win rate: {side_stats.get('win_rate', 0):.1f}%")
    except Exception as e:
        print_test("SignalScorer.get_side_stats", False, f"Error: {e}")
        return False
    
    return True


def test_ai_commentary():
    """Test AI Commentary Generator"""
    print_header("TESTING AI COMMENTARY")
    
    try:
        ai = AICommentaryGenerator()
        print_test("AICommentaryGenerator initialization", True)
    except Exception as e:
        print_test("AICommentaryGenerator initialization", False, f"Error: {e}")
        return False
    
    # Test position open commentary
    try:
        commentary = ai.generate_position_open_commentary('BTCUSDT', 'BUY', 0.85)
        print_test("Position Open Commentary", True, 
                   f"Generated {len(commentary)} chars")
        print(f"\n   Sample output:\n   {commentary[:200]}...\n")
    except Exception as e:
        print_test("Position Open Commentary", False, f"Error: {e}")
        return False
    
    # Test position close commentary
    try:
        commentary = ai.generate_position_close_commentary('BTCUSDT', 'BUY', 25.50, 2.5)
        print_test("Position Close Commentary", True,
                   f"Generated {len(commentary)} chars")
        print(f"\n   Sample output:\n   {commentary[:200]}...\n")
    except Exception as e:
        print_test("Position Close Commentary", False, f"Error: {e}")
        return False
    
    # Test daily summary commentary
    try:
        commentary = ai.generate_daily_summary_commentary(50.00, 3)
        print_test("Daily Summary Commentary", True,
                   f"Generated {len(commentary)} chars")
        print(f"\n   Sample output:\n   {commentary[:200]}...\n")
    except Exception as e:
        print_test("Daily Summary Commentary", False, f"Error: {e}")
        return False
    
    # Test cache
    try:
        cache_info = ai._cache
        cache_size = len(cache_info)
        print_test("AI Commentary Cache", True,
                   f"{cache_size} cached entries")
    except Exception as e:
        print_test("AI Commentary Cache", False, f"Error: {e}")
    
    return True


def test_adaptive_tactics():
    """Test Adaptive Tactics Manager"""
    print_header("TESTING ADAPTIVE TACTICS")
    
    try:
        config = Config()
        database = Database(config)
        tactics = AdaptiveTacticsManager(config, database, logger=None)
        print_test("AdaptiveTacticsManager initialization", True)
    except Exception as e:
        print_test("AdaptiveTacticsManager initialization", False, f"Error: {e}")
        return False
    
    # Test analyze and adjust
    try:
        result = tactics.analyze_and_adjust()
        adjustments = result.get('adjustments', [])
        print_test("Analyze and Adjust", True,
                   f"Made {len(adjustments)} tactical adjustments")
        
        if adjustments:
            print("\n   Recent adjustments:")
            for adj in adjustments[:3]:
                print(f"   • {adj}")
    except Exception as e:
        print_test("Analyze and Adjust", False, f"Error: {e}")
        return False
    
    # Test get current tactics
    try:
        current = tactics.get_current_tactics()
        print_test("Get Current Tactics", True,
                   f"Position multiplier: {current.get('position_size_multiplier', 1.0):.2f}, "
                   f"Confidence threshold: {current.get('confidence_threshold', 0.5):.2f}")
        
        print("\n   Current Tactics:")
        print(f"   • Position size multiplier: {current.get('position_size_multiplier', 1.0):.2f}")
        print(f"   • Confidence threshold: {current.get('confidence_threshold', 0.5):.2f}")
        print(f"   • Max positions: {current.get('max_positions', 0)}")
        
        blocked = current.get('blocked_symbols', [])
        if blocked:
            print(f"   • Blocked symbols: {', '.join(blocked)}")
        else:
            print(f"   • Blocked symbols: None")
    except Exception as e:
        print_test("Get Current Tactics", False, f"Error: {e}")
        return False
    
    return True


def test_integration():
    """Test integration with bot components"""
    print_header("TESTING INTEGRATION")
    
    # Test database connection
    try:
        config = Config()
        db = Database(config)
        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM positions")
        count = cursor.fetchone()[0]
        print_test("Database Connection", True, f"{count} positions in database")
    except Exception as e:
        print_test("Database Connection", False, f"Error: {e}")
        return False
    
    # Test config loading
    try:
        config = Config()
        print_test("Config Loading", True,
                   f"Max positions: {config.max_open_positions}, "
                   f"Max daily trades: {config.max_daily_trades}")
    except Exception as e:
        print_test("Config Loading", False, f"Error: {e}")
        return False
    
    return True


def main():
    """Run all diagnostic tests"""
    print("\n")
    print("🤖" * 40)
    print("  AI SYSTEM DIAGNOSTICS")
    print("🤖" * 40)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_ml_analyzers()
    all_passed &= test_ai_commentary()
    all_passed &= test_adaptive_tactics()
    all_passed &= test_integration()
    
    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - AI System is fully operational!\n")
        print("The AI is working correctly and ready to:")
        print("  • Generate intelligent commentary on trades")
        print("  • Provide performance insights and recommendations")
        print("  • Automatically adapt trading tactics based on performance")
        print("  • Monitor and optimize strategy performance\n")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - AI System has issues\n")
        print("Please check the errors above and:")
        print("  1. Ensure the bot has been updated: sudo ./scripts/update_bot.sh")
        print("  2. Check logs: sudo journalctl -u trading-bot -n 100")
        print("  3. Verify database: ls -lh /var/lib/trading-bot/trading_bot.db")
        print("  4. Contact support if issues persist\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
