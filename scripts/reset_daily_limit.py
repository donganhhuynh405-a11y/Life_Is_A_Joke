#!/usr/bin/env python3
"""
Reset Daily Trade Limit Counter
Allows bot to continue trading by resetting today's trade count
"""

import sqlite3
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    # Load environment variables
    env_file = os.environ.get('CONFIG_DIR', '/etc/trading-bot') + '/.env'
    if not os.path.exists(env_file):
        env_file = str(Path(__file__).parent.parent / '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)
except ImportError:
    print("⚠️  python-dotenv not installed")

from src.core.config import Config


def reset_daily_trade_count():
    """Reset today's trade count to allow more trading"""
    
    print("================================================================================")
    print("  RESET DAILY TRADE LIMIT")
    print("================================================================================")
    print()
    
    config = Config()
    db_path = config.db_path
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    print(f"📁 Database: {db_path}")
    print(f"⚙️  Current max daily trades: {config.max_daily_trades}")
    print()
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get current count
        cursor.execute('''
            SELECT COUNT(*) as count 
            FROM trades 
            WHERE DATE(timestamp) = DATE('now')
        ''')
        current_count = cursor.fetchone()[0]
        
        print(f"📊 Current trades today: {current_count} / {config.max_daily_trades}")
        
        if current_count == 0:
            print("✅ No trades today - counter is already at 0")
            conn.close()
            return True
        
        print()
        print("⚠️  This will DELETE today's trade records from the database")
        print("    to reset the daily counter. This action cannot be undone!")
        print()
        
        # Confirm
        response = input("Are you sure you want to reset the counter? (yes/no): ").lower().strip()
        
        if response != 'yes':
            print("❌ Operation cancelled")
            conn.close()
            return False
        
        # Delete today's trades
        cursor.execute('''
            DELETE FROM trades 
            WHERE DATE(timestamp) = DATE('now')
        ''')
        
        deleted = cursor.rowcount
        conn.commit()
        
        print()
        print(f"✅ Successfully deleted {deleted} trade record(s)")
        print(f"✅ Daily trade counter reset to 0")
        print()
        print("📈 Bot can now execute new trades (up to the daily limit)")
        print()
        print("💡 Tip: Consider increasing MAX_DAILY_TRADES in .env if you")
        print("   frequently hit the limit:")
        print(f"   Current: MAX_DAILY_TRADES={config.max_daily_trades}")
        print("   Suggested: MAX_DAILY_TRADES=20 or MAX_DAILY_TRADES=50")
        print()
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error resetting counter: {e}")
        return False


def show_current_status():
    """Show current trading status without making changes"""
    
    print("================================================================================")
    print("  DAILY TRADING STATUS")
    print("================================================================================")
    print()
    
    config = Config()
    db_path = config.db_path
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get current count
        cursor.execute('''
            SELECT COUNT(*) as count 
            FROM trades 
            WHERE DATE(timestamp) = DATE('now')
        ''')
        current_count = cursor.fetchone()[0]
        
        # Get recent trades
        cursor.execute('''
            SELECT symbol, side, quantity, price, timestamp, strategy
            FROM trades 
            WHERE DATE(timestamp) = DATE('now')
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        recent_trades = cursor.fetchall()
        
        print(f"📊 Trades today: {current_count} / {config.max_daily_trades}")
        
        limit_status = "✅ AVAILABLE" if current_count < config.max_daily_trades else "🚫 LIMIT REACHED"
        remaining = max(0, config.max_daily_trades - current_count)
        
        print(f"🎯 Status: {limit_status}")
        print(f"📈 Remaining: {remaining} trades")
        print()
        
        if recent_trades:
            print("📋 Recent Trades Today:")
            print("--------------------------------------------------------------------------------")
            for trade in recent_trades:
                time = trade['timestamp'].split()[1] if ' ' in trade['timestamp'] else trade['timestamp']
                print(f"   {time} - {trade['side']} {trade['symbol']} x {trade['quantity']} @ ${trade['price']}")
            print()
        
        # Get daily P&L
        cursor.execute('''
            SELECT COALESCE(SUM(pnl), 0) as total
            FROM positions 
            WHERE status = 'closed'
            AND DATE(closed_at) = DATE('now', 'localtime')
            AND pnl IS NOT NULL
        ''')
        daily_pnl = cursor.fetchone()[0]
        
        if daily_pnl != 0:
            pnl_emoji = "💰" if daily_pnl > 0 else "📉"
            print(f"{pnl_emoji} Today's P&L: ${daily_pnl:+,.2f}")
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--status', '-s']:
            show_current_status()
            return
        elif sys.argv[1] in ['--help', '-h']:
            print("Usage:")
            print("  python3 scripts/reset_daily_limit.py          - Reset daily trade counter")
            print("  python3 scripts/reset_daily_limit.py --status - Show current status")
            print("  python3 scripts/reset_daily_limit.py --help   - Show this help")
            return
    
    # Default action: reset counter
    reset_daily_trade_count()


if __name__ == '__main__':
    main()
