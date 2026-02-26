#!/usr/bin/env python3
"""
Cleanup Old Data Script
Automatically removes old logs, data, and cache files
"""

import os
import sys
import time
import shutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path


class DataCleanup:
    """Automatic data cleanup utility"""
    
    def __init__(self, base_dir="/opt/trading-bot", dry_run=False, aggressive=False):
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.aggressive = aggressive
        self.total_freed = 0
        
        # Retention periods (days)
        if aggressive:
            self.retention = {
                'logs': 7,
                'historical_data': 30,
                'ml_models': 3,
                'news_cache': 1,
                'backups': 7,
                'temp_files': 0
            }
        else:
            self.retention = {
                'logs': 30,
                'historical_data': 90,
                'ml_models': 14,
                'news_cache': 7,
                'backups': 14,
                'temp_files': 1
            }
    
    def get_size(self, path):
        """Get size of file or directory"""
        if path.is_file():
            return path.stat().st_size
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except PermissionError:
            pass
        return total
    
    def format_size(self, size):
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f}{unit}"
            size /= 1024.0
        return f"{size:.2f}TB"
    
    def cleanup_old_files(self, directory, days, pattern='*'):
        """Remove files older than specified days"""
        if not directory.exists():
            return 0
        
        cutoff_time = time.time() - (days * 86400)
        freed = 0
        files_removed = 0
        
        try:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        size = file_path.stat().st_size
                        if self.dry_run:
                            print(f"  Would remove: {file_path} ({self.format_size(size)})")
                        else:
                            try:
                                file_path.unlink()
                                freed += size
                                files_removed += 1
                            except Exception as e:
                                print(f"  Error removing {file_path}: {e}")
        except Exception as e:
            print(f"  Error scanning {directory}: {e}")
        
        return freed, files_removed
    
    def cleanup_logs(self):
        """Clean up old log files"""
        print(f"\n📝 Cleaning logs older than {self.retention['logs']} days...")
        
        log_dirs = [
            Path("/var/log/trading-bot"),
            self.base_dir / "logs",
        ]
        
        total_freed = 0
        total_files = 0
        
        for log_dir in log_dirs:
            if log_dir.exists():
                result = self.cleanup_old_files(log_dir, self.retention['logs'], '*.log*')
                if isinstance(result, tuple):
                    freed, files = result
                else:
                    freed = result
                    files = 0
                total_freed += freed
                total_files += files
        
        if not self.dry_run:
            print(f"  ✓ Removed {total_files} log files, freed {self.format_size(total_freed)}")
        
        self.total_freed += total_freed
        return total_freed
    
    def cleanup_historical_data(self):
        """Clean up old historical ML data"""
        print(f"\n🧠 Cleaning historical data older than {self.retention['historical_data']} days...")
        
        data_dir = Path("/var/lib/trading-bot/historical_data")
        if not data_dir.exists():
            print("  No historical data directory found")
            return 0
        
        result = self.cleanup_old_files(data_dir, self.retention['historical_data'], '*.parquet')
        if isinstance(result, tuple):
            freed, files = result
        else:
            freed = result
            files = 0
        
        if not self.dry_run:
            print(f"  ✓ Removed {files} data files, freed {self.format_size(freed)}")
        
        self.total_freed += freed
        return freed
    
    def cleanup_ml_models(self):
        """Clean up old ML models"""
        print(f"\n🤖 Cleaning ML models older than {self.retention['ml_models']} days...")
        
        models_dir = Path("/var/lib/trading-bot/models")
        if not models_dir.exists():
            print("  No models directory found")
            return 0
        
        result = self.cleanup_old_files(models_dir, self.retention['ml_models'], '*.pkl')
        if isinstance(result, tuple):
            freed, files = result
        else:
            freed = result
            files = 0
        
        if not self.dry_run:
            print(f"  ✓ Removed {files} model files, freed {self.format_size(freed)}")
        
        self.total_freed += freed
        return freed
    
    def cleanup_news_cache(self):
        """Clean up old news cache"""
        print(f"\n📰 Cleaning news cache older than {self.retention['news_cache']} days...")
        
        # News is in database, but we can clean any cache files
        cache_dirs = [
            self.base_dir / "cache",
            Path("/tmp/trading-bot-cache")
        ]
        
        total_freed = 0
        total_files = 0
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                result = self.cleanup_old_files(cache_dir, self.retention['news_cache'])
                if isinstance(result, tuple):
                    freed, files = result
                else:
                    freed = result
                    files = 0
                total_freed += freed
                total_files += files
        
        if not self.dry_run and total_files > 0:
            print(f"  ✓ Removed {total_files} cache files, freed {self.format_size(total_freed)}")
        elif total_files == 0:
            print("  No cache files to clean")
        
        self.total_freed += total_freed
        return total_freed
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        print(f"\n🗑️  Cleaning temporary files...")
        
        temp_dirs = [
            self.base_dir / "tmp",
            Path("/tmp/trading-bot"),
        ]
        
        total_freed = 0
        total_files = 0
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                result = self.cleanup_old_files(temp_dir, self.retention['temp_files'])
                if isinstance(result, tuple):
                    freed, files = result
                else:
                    freed = result
                    files = 0
                total_freed += freed
                total_files += files
        
        if not self.dry_run and total_files > 0:
            print(f"  ✓ Removed {total_files} temp files, freed {self.format_size(total_freed)}")
        elif total_files == 0:
            print("  No temp files to clean")
        
        self.total_freed += total_freed
        return total_freed
    
    def cleanup_old_backups(self):
        """Clean up old backup files"""
        print(f"\n💾 Cleaning backups older than {self.retention['backups']} days...")
        
        backup_dir = Path("/var/backups/trading-bot")
        if not backup_dir.exists():
            print("  No backup directory found")
            return 0
        
        result = self.cleanup_old_files(backup_dir, self.retention['backups'], '*.tar.gz')
        if isinstance(result, tuple):
            freed, files = result
        else:
            freed = result
            files = 0
        
        if not self.dry_run:
            print(f"  ✓ Removed {files} backup files, freed {self.format_size(freed)}")
        
        self.total_freed += freed
        return freed
    
    def run(self):
        """Run all cleanup operations"""
        print("=" * 60)
        print("🔧 Trading Bot - Data Cleanup")
        print("=" * 60)
        
        if self.dry_run:
            print("\n⚠️  DRY RUN MODE - No files will be deleted\n")
        
        if self.aggressive:
            print("⚡ AGGRESSIVE MODE - Shorter retention periods\n")
        
        print(f"Retention periods:")
        for key, days in self.retention.items():
            print(f"  - {key}: {days} days")
        
        # Run all cleanup operations
        self.cleanup_logs()
        self.cleanup_historical_data()
        self.cleanup_ml_models()
        self.cleanup_news_cache()
        self.cleanup_temp_files()
        self.cleanup_old_backups()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Cleanup Summary")
        print("=" * 60)
        
        if self.dry_run:
            print(f"Would free: {self.format_size(self.total_freed)}")
        else:
            print(f"Total freed: {self.format_size(self.total_freed)}")
        
        print("\n✅ Cleanup completed!")
        
        return self.total_freed


def main():
    parser = argparse.ArgumentParser(description='Clean up old trading bot data')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--aggressive', action='store_true',
                        help='Use shorter retention periods for aggressive cleanup')
    parser.add_argument('--auto', action='store_true',
                        help='Run without confirmation (for automation)')
    parser.add_argument('--base-dir', default='/opt/trading-bot',
                        help='Base directory of trading bot (default: /opt/trading-bot)')
    
    args = parser.parse_args()
    
    # Confirmation
    if not args.dry_run and not args.auto:
        print("⚠️  This will permanently delete old data!")
        if args.aggressive:
            print("⚠️  AGGRESSIVE mode - very short retention periods!")
        response = input("\nContinue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return
    
    # Run cleanup
    cleanup = DataCleanup(
        base_dir=args.base_dir,
        dry_run=args.dry_run,
        aggressive=args.aggressive
    )
    
    freed = cleanup.run()
    
    # Exit code
    sys.exit(0)


if __name__ == '__main__':
    main()
