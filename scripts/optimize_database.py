#!/usr/bin/env python3
"""
Database Optimization Script
Optimizes SQLite database with VACUUM, ANALYZE, and cleanup
"""

import os
import sys
import time
import sqlite3
import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path


class DatabaseOptimizer:
    """SQLite database optimization utility"""
    
    def __init__(self, db_path="/var/lib/trading-bot/trading_bot.db", backup=True):
        self.db_path = Path(db_path)
        self.backup = backup
        self.original_size = 0
        self.optimized_size = 0
        
        if not self.db_path.exists():
            print(f"❌ Database not found: {self.db_path}")
            sys.exit(1)
        
        self.original_size = self.db_path.stat().st_size
    
    def format_size(self, size):
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f}{unit}"
            size /= 1024.0
        return f"{size:.2f}TB"
    
    def create_backup(self):
        """Create database backup"""
        if not self.backup:
            return None
        
        print("\n💾 Creating database backup...")
        
        backup_dir = Path("/var/backups/trading-bot")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"trading_bot_backup_{timestamp}.db"
        
        try:
            shutil.copy2(self.db_path, backup_path)
            print(f"  ✓ Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"  ❌ Backup failed: {e}")
            return None
    
    def vacuum_database(self):
        """Run VACUUM to reclaim space"""
        print("\n🔧 Running VACUUM...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # VACUUM
            print("  Compacting database...")
            cursor.execute("VACUUM")
            
            conn.commit()
            conn.close()
            
            print("  ✓ VACUUM completed")
            return True
        except Exception as e:
            print(f"  ❌ VACUUM failed: {e}")
            return False
    
    def analyze_database(self):
        """Run ANALYZE to update statistics"""
        print("\n📊 Running ANALYZE...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ANALYZE
            print("  Updating statistics...")
            cursor.execute("ANALYZE")
            
            conn.commit()
            conn.close()
            
            print("  ✓ ANALYZE completed")
            return True
        except Exception as e:
            print(f"  ❌ ANALYZE failed: {e}")
            return False
    
    def cleanup_old_records(self, days=365):
        """Archive old trade records"""
        print(f"\n🗄️  Archiving trades older than {days} days...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            archived = 0
            
            # Archive old trades
            if 'trades' in tables:
                cutoff_date = datetime.now() - timedelta(days=days)
                cutoff_str = cutoff_date.strftime('%Y-%m-%d')
                
                cursor.execute("""
                    SELECT COUNT(*) FROM trades 
                    WHERE timestamp < ?
                """, (cutoff_str,))
                
                count = cursor.fetchone()[0]
                
                if count > 0:
                    print(f"  Found {count} old trade records")
                    
                    # Create archive table if not exists
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS trades_archive (
                            id INTEGER PRIMARY KEY,
                            symbol TEXT,
                            side TEXT,
                            order_id TEXT,
                            price REAL,
                            quantity REAL,
                            commission REAL,
                            profit_loss REAL,
                            strategy TEXT,
                            timestamp TEXT,
                            status TEXT
                        )
                    """)
                    
                    # Move to archive
                    cursor.execute("""
                        INSERT INTO trades_archive 
                        SELECT * FROM trades WHERE timestamp < ?
                    """, (cutoff_str,))
                    
                    # Delete from main table
                    cursor.execute("""
                        DELETE FROM trades WHERE timestamp < ?
                    """, (cutoff_str,))
                    
                    archived = count
                    print(f"  ✓ Archived {archived} trade records")
                else:
                    print("  No old trades to archive")
            
            conn.commit()
            conn.close()
            
            return archived
        except Exception as e:
            print(f"  ❌ Archive failed: {e}")
            return 0
    
    def reindex_database(self):
        """Rebuild database indexes"""
        print("\n🔄 Rebuilding indexes...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all indexes
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
            """)
            
            indexes = [row[0] for row in cursor.fetchall()]
            
            if indexes:
                print(f"  Found {len(indexes)} indexes")
                cursor.execute("REINDEX")
                print(f"  ✓ Reindexed {len(indexes)} indexes")
            else:
                print("  No custom indexes found")
            
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            print(f"  ❌ Reindex failed: {e}")
            return False
    
    def get_database_info(self):
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table sizes
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            print("\n📋 Database tables:")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  - {table}: {count} records")
            
            conn.close()
        except Exception as e:
            print(f"  ❌ Could not get database info: {e}")
    
    def run(self):
        """Run all optimization operations"""
        print("=" * 60)
        print("🔧 Trading Bot - Database Optimization")
        print("=" * 60)
        
        print(f"\nDatabase: {self.db_path}")
        print(f"Original size: {self.format_size(self.original_size)}")
        
        # Get info before optimization
        self.get_database_info()
        
        # Create backup
        if self.backup:
            self.create_backup()
        
        # Run optimizations
        self.cleanup_old_records()
        self.vacuum_database()
        self.analyze_database()
        self.reindex_database()
        
        # Get final size
        self.optimized_size = self.db_path.stat().st_size
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Optimization Summary")
        print("=" * 60)
        
        print(f"Original size: {self.format_size(self.original_size)}")
        print(f"Optimized size: {self.format_size(self.optimized_size)}")
        
        if self.original_size > self.optimized_size:
            saved = self.original_size - self.optimized_size
            percent = (saved / self.original_size) * 100
            print(f"Space saved: {self.format_size(saved)} ({percent:.1f}%)")
        
        print("\n✅ Optimization completed!")
        
        return self.optimized_size


def main():
    parser = argparse.ArgumentParser(description='Optimize trading bot database')
    parser.add_argument('--db-path', default='/var/lib/trading-bot/trading_bot.db',
                        help='Path to database file (default: /var/lib/trading-bot/trading_bot.db)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip creating backup before optimization')
    parser.add_argument('--archive-days', type=int, default=365,
                        help='Archive trades older than N days (default: 365)')
    
    args = parser.parse_args()
    
    # Confirmation
    print("⚠️  This will optimize the database (may take a few minutes)")
    if args.no_backup:
        print("⚠️  WARNING: No backup will be created!")
    
    # Run optimization
    optimizer = DatabaseOptimizer(
        db_path=args.db_path,
        backup=not args.no_backup
    )
    
    optimizer.run()
    
    sys.exit(0)


if __name__ == '__main__':
    main()
