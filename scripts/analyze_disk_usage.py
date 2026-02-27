#!/usr/bin/env python3
"""
Disk Usage Analysis Script
Analyzes disk space usage of trading bot components
"""

import sys
import argparse
from pathlib import Path


class DiskAnalyzer:
    """Disk usage analyzer for trading bot"""

    def __init__(self, base_dir="/opt/trading-bot"):
        self.base_dir = Path(base_dir)
        self.total_size = 0
        self.component_sizes = {}

    def format_size(self, size):
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f}{unit}"
            size /= 1024.0
        return f"{size:.2f}TB"

    def get_directory_size(self, path):
        """Get total size of directory"""
        if not path.exists():
            return 0

        if path.is_file():
            return path.stat().st_size

        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except (PermissionError, OSError):
                        pass
        except (PermissionError, OSError):
            pass

        return total

    def analyze_logs(self):
        """Analyze log files"""
        print("\n📝 Analyzing logs...")

        log_dirs = [
            Path("/var/log/trading-bot"),
            self.base_dir / "logs"
        ]

        total_size = 0
        for log_dir in log_dirs:
            if log_dir.exists():
                size = self.get_directory_size(log_dir)
                total_size += size
                if size > 0:
                    print(f"  {log_dir}: {self.format_size(size)}")

        self.component_sizes['logs'] = total_size
        self.total_size += total_size

        if total_size > 1024 * 1024 * 1024:  # > 1GB
            print(f"  ⚠️  Logs are large ({self.format_size(total_size)})")
            print("  💡 Recommendation: Run cleanup_old_data.py")

    def analyze_database(self):
        """Analyze database files"""
        print("\n💾 Analyzing database...")

        db_dir = Path("/var/lib/trading-bot")

        if not db_dir.exists():
            print("  Database directory not found")
            return

        total_size = 0

        # Main database
        db_file = db_dir / "trading_bot.db"
        if db_file.exists():
            size = db_file.stat().st_size
            total_size += size
            print(f"  trading_bot.db: {self.format_size(size)}")

        # WAL files
        for wal_file in db_dir.glob("*.db-wal"):
            size = wal_file.stat().st_size
            total_size += size
            print(f"  {wal_file.name}: {self.format_size(size)}")

        # SHM files
        for shm_file in db_dir.glob("*.db-shm"):
            size = shm_file.stat().st_size
            total_size += size
            print(f"  {shm_file.name}: {self.format_size(size)}")

        self.component_sizes['database'] = total_size
        self.total_size += total_size

        if total_size > 500 * 1024 * 1024:  # > 500MB
            print("  💡 Recommendation: Run optimize_database.py")

    def analyze_ml_data(self):
        """Analyze ML data files"""
        print("\n🧠 Analyzing ML data...")

        data_dirs = [
            Path("/var/lib/trading-bot/historical_data"),
            Path("/var/lib/trading-bot/models")
        ]

        total_size = 0

        for data_dir in data_dirs:
            if data_dir.exists():
                size = self.get_directory_size(data_dir)
                total_size += size
                if size > 0:
                    print(f"  {data_dir.name}: {self.format_size(size)}")

        self.component_sizes['ml_data'] = total_size
        self.total_size += total_size

        if total_size > 3 * 1024 * 1024 * 1024:  # > 3GB
            print(f"  ⚠️  ML data is large ({self.format_size(total_size)})")
            print("  💡 Recommendation: Clean old historical data")

    def analyze_backups(self):
        """Analyze backup files"""
        print("\n💾 Analyzing backups...")

        backup_dir = Path("/var/backups/trading-bot")

        if not backup_dir.exists():
            print("  No backups found")
            return

        size = self.get_directory_size(backup_dir)
        self.component_sizes['backups'] = size
        self.total_size += size

        # Count backup files
        backup_files = list(backup_dir.glob("*.tar.gz")) + list(backup_dir.glob("*.db"))
        print(f"  Backup files: {len(backup_files)}")
        print(f"  Total size: {self.format_size(size)}")

        if size > 1024 * 1024 * 1024:  # > 1GB
            print("  💡 Recommendation: Remove old backups")

    def analyze_cache(self):
        """Analyze cache files"""
        print("\n🗃️  Analyzing cache...")

        cache_dirs = [
            self.base_dir / "cache",
            Path("/tmp/trading-bot-cache")
        ]

        total_size = 0

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                size = self.get_directory_size(cache_dir)
                total_size += size
                if size > 0:
                    print(f"  {cache_dir}: {self.format_size(size)}")

        self.component_sizes['cache'] = total_size
        self.total_size += total_size

    def analyze_temp_files(self):
        """Analyze temporary files"""
        print("\n🗑️  Analyzing temp files...")

        temp_dirs = [
            self.base_dir / "tmp",
            Path("/tmp/trading-bot")
        ]

        total_size = 0

        for temp_dir in temp_dirs:
            if temp_dir.exists():
                size = self.get_directory_size(temp_dir)
                total_size += size
                if size > 0:
                    print(f"  {temp_dir}: {self.format_size(size)}")

        self.component_sizes['temp'] = total_size
        self.total_size += total_size

        if total_size > 100 * 1024 * 1024:  # > 100MB
            print("  💡 Recommendation: Clean temp files")

    def show_summary(self):
        """Show summary of disk usage"""
        print("\n" + "=" * 60)
        print("📊 Disk Usage Summary")
        print("=" * 60)

        print(f"\nTotal usage: {self.format_size(self.total_size)}\n")

        # Sort by size
        sorted_components = sorted(
            self.component_sizes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for component, size in sorted_components:
            if size > 0:
                percent = (size / self.total_size * 100) if self.total_size > 0 else 0
                print(f"  {component:15} {self.format_size(size):>10} ({percent:5.1f}%)")

        print("\n" + "=" * 60)
        print("💡 Recommendations")
        print("=" * 60)

        # Recommendations
        if self.component_sizes.get('logs', 0) > 1024 * 1024 * 1024:
            print("\n  🔸 Logs are large (>1GB)")
            print("     Run: python3 scripts/cleanup_old_data.py")

        if self.component_sizes.get('database', 0) > 500 * 1024 * 1024:
            print("\n  🔸 Database is large (>500MB)")
            print("     Run: python3 scripts/optimize_database.py")

        if self.component_sizes.get('ml_data', 0) > 3 * 1024 * 1024 * 1024:
            print("\n  🔸 ML data is large (>3GB)")
            print("     Run: python3 scripts/cleanup_old_data.py --aggressive")

        print()

    def run(self):
        """Run full disk analysis"""
        print("=" * 60)
        print("🔍 Trading Bot - Disk Usage Analysis")
        print("=" * 60)

        self.analyze_logs()
        self.analyze_database()
        self.analyze_ml_data()
        self.analyze_backups()
        self.analyze_cache()
        self.analyze_temp_files()

        self.show_summary()

        return self.total_size


def main():
    parser = argparse.ArgumentParser(description='Analyze trading bot disk usage')
    parser.add_argument('--base-dir', default='/opt/trading-bot',
                        help='Base directory of trading bot (default: /opt/trading-bot)')

    args = parser.parse_args()

    analyzer = DiskAnalyzer(base_dir=args.base_dir)
    analyzer.run()

    sys.exit(0)


if __name__ == '__main__':
    main()
