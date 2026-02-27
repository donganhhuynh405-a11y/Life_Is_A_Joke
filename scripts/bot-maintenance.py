#!/usr/bin/env python3
"""
Trading Bot - Maintenance Tool
Maintenance and cleanup tool with interactive menu.
"""

import os
import sys
import argparse
import subprocess
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Colors:
    """ANSI color codes"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    BOLD = '\033[1m'
    NC = '\033[0m'


def print_info(message):
    """Print info message"""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


def run_command(cmd, show_output=True, check=True):
    """
    Run shell command and return output.

    SECURITY NOTE: This function uses shell=True for convenience with internal
    commands. All user input passed to this function MUST be sanitized first.
    """
    try:
        if show_output:
            result = subprocess.run(cmd, shell=True, text=True, check=check)
            return result.returncode == 0
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
            return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if show_output:
            print_error(f"Command failed: {e}")
        return None if not show_output else False


def get_size(path):
    """Get size of file or directory in bytes"""
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    if os.path.exists(fp):
                        total += os.path.getsize(fp)
                except (OSError, PermissionError):
                    pass
        return total
    return 0


def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def show_progress(message, duration=2):
    """Show progress indicator"""
    print(f"\r{message}", end='', flush=True)
    for i in range(duration * 10):
        print(".", end='', flush=True)
        time.sleep(0.1)
    print(" Done!")


def clean_cache():
    """Clean cache directories"""
    print_info("Cleaning cache...")

    project_root = Path(__file__).parent.parent
    cache_dirs = [
        project_root / "__pycache__",
        project_root / "src" / "__pycache__",
        project_root / ".pytest_cache",
        project_root / ".mypy_cache",
        project_root / "data" / "cache",
    ]

    total_freed = 0

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            size = get_size(cache_dir)
            print(f"  Removing {cache_dir.name}... ({format_size(size)})")
            try:
                shutil.rmtree(cache_dir)
                total_freed += size
                print_success(f"  Removed {cache_dir.name}")
            except Exception as e:
                print_error(f"  Failed to remove {cache_dir.name}: {e}")

    # Clean Python cache files
    print_info("Cleaning .pyc files...")
    run_command(f"find {project_root} -type f -name '*.pyc' -delete")

    print_success(f"Total space freed: {format_size(total_freed)}")


def clean_logs(keep_days=7):
    """Clean old log files"""
    print_info(f"Cleaning logs older than {keep_days} days...")

    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "logs"

    if not logs_dir.exists():
        print_warning("Logs directory not found")
        return

    cutoff_date = datetime.now() - timedelta(days=keep_days)
    total_freed = 0
    files_removed = 0

    for log_file in logs_dir.glob("*.log*"):
        if log_file.is_file():
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if mtime < cutoff_date:
                size = log_file.stat().st_size
                print(f"  Removing {log_file.name}... ({format_size(size)})")
                try:
                    log_file.unlink()
                    total_freed += size
                    files_removed += 1
                except Exception as e:
                    print_error(f"  Failed to remove {log_file.name}: {e}")

    print_success(f"Removed {files_removed} files, freed {format_size(total_freed)}")


def clean_old_data(keep_days=30):
    """Clean old data files"""
    print_info(f"Cleaning data older than {keep_days} days...")

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    if not data_dir.exists():
        print_warning("Data directory not found")
        return

    cutoff_date = datetime.now() - timedelta(days=keep_days)
    total_freed = 0
    files_removed = 0

    for data_file in data_dir.rglob("*"):
        if data_file.is_file() and data_file.suffix in ['.json', '.csv', '.pickle', '.pkl']:
            mtime = datetime.fromtimestamp(data_file.stat().st_mtime)
            if mtime < cutoff_date:
                size = data_file.stat().st_size
                if size > 1024 * 1024:  # Only show files > 1MB
                    print(f"  Removing {data_file.name}... ({format_size(size)})")
                try:
                    data_file.unlink()
                    total_freed += size
                    files_removed += 1
                except Exception as e:
                    print_error(f"  Failed to remove {data_file.name}: {e}")

    print_success(f"Removed {files_removed} files, freed {format_size(total_freed)}")


def optimize_database():
    """Optimize SQLite database"""
    print_info("Optimizing database...")

    project_root = Path(__file__).parent.parent
    db_files = list(project_root.glob("**/*.db")) + list((project_root / "data").glob("**/*.db"))

    if not db_files:
        print_warning("No database files found")
        return

    for db_file in db_files:
        print(f"\n  Processing {db_file.name}...")

        try:
            # Get size before
            size_before = db_file.stat().st_size

            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()

            # VACUUM
            print("    Running VACUUM...")
            cursor.execute("VACUUM")

            # ANALYZE
            print("    Running ANALYZE...")
            cursor.execute("ANALYZE")

            # Reindex
            print("    Running REINDEX...")
            cursor.execute("REINDEX")

            conn.commit()
            conn.close()

            # Get size after
            size_after = db_file.stat().st_size
            saved = size_before - size_after

            if saved > 0:
                print_success(f"    Optimized {db_file.name}: saved {format_size(saved)}")
            else:
                print_success(f"    Optimized {db_file.name}: no space saved")

        except Exception as e:
            print_error(f"    Failed to optimize {db_file.name}: {e}")


def remove_old_trades(days=90):
    """Remove old trade records from database"""
    print_info(f"Removing trades older than {days} days...")

    project_root = Path(__file__).parent.parent
    db_file = project_root / "data" / "trading.db"

    if not db_file.exists():
        print_warning("Trading database not found")
        return

    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_timestamp = int(cutoff_date.timestamp())

        # Count records to delete
        cursor.execute("SELECT COUNT(*) FROM trades WHERE timestamp < ?", (cutoff_timestamp,))
        count = cursor.fetchone()[0]

        if count == 0:
            print_warning("No old trades to remove")
            conn.close()
            return

        # Delete old records
        cursor.execute("DELETE FROM trades WHERE timestamp < ?", (cutoff_timestamp,))
        conn.commit()
        conn.close()

        print_success(f"Removed {count} old trade records")

    except Exception as e:
        print_error(f"Failed to remove old trades: {e}")


def analyze_disk_usage():
    """Analyze disk usage"""
    print_info("Analyzing disk usage...")
    print("=" * 80)

    project_root = Path(__file__).parent.parent

    directories = {
        'Logs': project_root / 'logs',
        'Data': project_root / 'data',
        'Cache': project_root / '__pycache__',
        'Backups': project_root / 'backups',
        'Source': project_root / 'src',
    }

    print(f"\n{Colors.BOLD}Directory Sizes:{Colors.NC}")
    total_size = 0

    for name, path in directories.items():
        if path.exists():
            size = get_size(path)
            total_size += size
            print(f"  {name:20} {format_size(size):>15}")

    print(f"  {'-' * 35}")
    print(f"  {'Total':20} {format_size(total_size):>15}")

    # System disk usage
    print(f"\n{Colors.BOLD}System Disk Usage:{Colors.NC}")
    run_command("df -h /")

    # Memory usage
    print(f"\n{Colors.BOLD}Memory Usage:{Colors.NC}")
    run_command("free -h")


def find_large_files(min_size_mb=10):
    """Find large files"""
    print_info(f"Finding files larger than {min_size_mb} MB...")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    min_size = min_size_mb * 1024 * 1024

    large_files = []

    for path in project_root.rglob("*"):
        if path.is_file():
            try:
                size = path.stat().st_size
                if size > min_size:
                    large_files.append((path, size))
            except (OSError, PermissionError):
                pass

    # Sort by size
    large_files.sort(key=lambda x: x[1], reverse=True)

    if not large_files:
        print_warning(f"No files larger than {min_size_mb} MB found")
        return

    print(f"\n{Colors.BOLD}Large Files:{Colors.NC}")
    for path, size in large_files[:20]:  # Show top 20
        rel_path = path.relative_to(project_root)
        print(f"  {format_size(size):>12}  {rel_path}")


def kill_duplicate_processes():
    """Kill duplicate bot processes"""
    print_info("Checking for duplicate processes...")

    # Find all bot processes
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'python.*main.py|python.*telegram_bot.py'],
            capture_output=True, text=True, check=False
        )
        pids_output = result.stdout.strip()
    except Exception:
        print_error("Failed to check for processes")
        return

    if not pids_output:
        print_warning("No bot processes found")
        return

    pids = pids_output.split('\n')

    if len(pids) <= 1:
        print_success("No duplicate processes found")
        return

    print_warning(f"Found {len(pids)} processes: {', '.join(pids)}")

    # Keep the oldest process, kill the rest
    print_info(f"Keeping oldest process (PID: {pids[0]})")

    for pid in pids[1:]:
        try:
            os.kill(int(pid), 15)  # SIGTERM
            print_success(f"Killed process {pid}")
        except Exception as e:
            print_error(f"Failed to kill process {pid}: {e}")


def system_info():
    """Display system information"""
    print_info("System Information")
    print("=" * 80)

    # OS Info
    print(f"\n{Colors.BOLD}Operating System:{Colors.NC}")
    run_command("uname -a")

    # Python version
    print(f"\n{Colors.BOLD}Python Version:{Colors.NC}")
    run_command("python3 --version")

    # Disk space
    print(f"\n{Colors.BOLD}Disk Space:{Colors.NC}")
    run_command("df -h /")

    # Memory
    print(f"\n{Colors.BOLD}Memory:{Colors.NC}")
    run_command("free -h")

    # CPU
    print(f"\n{Colors.BOLD}CPU:{Colors.NC}")
    run_command("lscpu | grep 'Model name\\|CPU(s)\\|Thread'")

    # Uptime
    print(f"\n{Colors.BOLD}System Uptime:{Colors.NC}")
    run_command("uptime")


def interactive_menu():
    """Display interactive menu"""
    while True:
        os.system('clear' if os.name != 'nt' else 'cls')

        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.NC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'Trading Bot - Maintenance Tool':^70}{Colors.NC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.NC}")
        print()

        print(f"{Colors.BOLD}Cleanup Operations:{Colors.NC}")
        print(f"  {Colors.GREEN}1{Colors.NC}. Clean cache directories")
        print(f"  {Colors.GREEN}2{Colors.NC}. Clean old logs (7 days)")
        print(f"  {Colors.GREEN}3{Colors.NC}. Clean old data files (30 days)")
        print(f"  {Colors.YELLOW}4{Colors.NC}. Full cleanup (all above)")

        print(f"\n{Colors.BOLD}Database Operations:{Colors.NC}")
        print(f"  {Colors.CYAN}5{Colors.NC}. Optimize database (vacuum, analyze, reindex)")
        print(f"  {Colors.YELLOW}6{Colors.NC}. Remove old trades (90 days)")
        print(f"  {Colors.YELLOW}7{Colors.NC}. Remove old news (90 days)")
        print(f"  {Colors.YELLOW}8{Colors.NC}. Remove old positions (90 days)")

        print(f"\n{Colors.BOLD}Analysis:{Colors.NC}")
        print(f"  {Colors.BLUE}9{Colors.NC}. Analyze disk usage")
        print(f"  {Colors.BLUE}10{Colors.NC}. Find large files")
        print(f"  {Colors.BLUE}11{Colors.NC}. System information")

        print(f"\n{Colors.BOLD}Process Management:{Colors.NC}")
        print(f"  {Colors.MAGENTA}12{Colors.NC}. Kill duplicate processes")

        print(f"\n  {Colors.RED}0{Colors.NC}. Exit")
        print()

        choice = input(f"{Colors.BOLD}Select option: {Colors.NC}").strip()

        if choice == '1':
            clean_cache()
            input("\nPress Enter to continue...")
        elif choice == '2':
            days = input("Keep logs for how many days? (default 7): ").strip() or "7"
            clean_logs(int(days))
            input("\nPress Enter to continue...")
        elif choice == '3':
            days = input("Keep data for how many days? (default 30): ").strip() or "30"
            clean_old_data(int(days))
            input("\nPress Enter to continue...")
        elif choice == '4':
            print_info("Running full cleanup...")
            clean_cache()
            print()
            clean_logs()
            print()
            clean_old_data()
            input("\nPress Enter to continue...")
        elif choice == '5':
            optimize_database()
            input("\nPress Enter to continue...")
        elif choice == '6':
            days = input("Remove trades older than how many days? (default 90): ").strip() or "90"
            remove_old_trades(int(days))
            input("\nPress Enter to continue...")
        elif choice == '7':
            print_warning("News cleanup not implemented yet")
            input("\nPress Enter to continue...")
        elif choice == '8':
            print_warning("Positions cleanup not implemented yet")
            input("\nPress Enter to continue...")
        elif choice == '9':
            analyze_disk_usage()
            input("\nPress Enter to continue...")
        elif choice == '10':
            size = input("Minimum file size in MB (default 10): ").strip() or "10"
            find_large_files(int(size))
            input("\nPress Enter to continue...")
        elif choice == '11':
            system_info()
            input("\nPress Enter to continue...")
        elif choice == '12':
            kill_duplicate_processes()
            input("\nPress Enter to continue...")
        elif choice == '0':
            print_info("Goodbye!")
            break
        else:
            print_error("Invalid option!")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description='Trading Bot Maintenance Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Interactive menu
  %(prog)s --clean-cache        # Clean cache
  %(prog)s --clean-logs         # Clean old logs
  %(prog)s --clean-logs --days 7  # Keep 7 days
  %(prog)s --optimize-db        # Optimize database
  %(prog)s --remove-old-trades  # Remove old trades
  %(prog)s --disk-usage         # Analyze disk
  %(prog)s --large-files        # Find large files
  %(prog)s --kill-duplicates    # Kill duplicate processes
        """
    )

    parser.add_argument('--clean-cache', action='store_true', help='Clean cache directories')
    parser.add_argument('--clean-logs', action='store_true', help='Clean old log files')
    parser.add_argument('--clean-data', action='store_true', help='Clean old data files')
    parser.add_argument('--full-cleanup', action='store_true', help='Run full cleanup')
    parser.add_argument('--optimize-db', action='store_true', help='Optimize database')
    parser.add_argument('--remove-old-trades', action='store_true', help='Remove old trade records')
    parser.add_argument('--disk-usage', action='store_true', help='Analyze disk usage')
    parser.add_argument('--large-files', action='store_true', help='Find large files')
    parser.add_argument('--min-size', type=int, default=10,
                        help='Minimum file size in MB (default: 10)')
    parser.add_argument('--kill-duplicates', action='store_true', help='Kill duplicate processes')
    parser.add_argument('--system-info', action='store_true', help='Display system information')
    parser.add_argument('--days', type=int, default=7, help='Number of days to keep (default: 7)')

    args = parser.parse_args()

    # If no arguments, show interactive menu
    if len(sys.argv) == 1:
        interactive_menu()
        return

    # Execute commands
    if args.clean_cache:
        clean_cache()
    elif args.clean_logs:
        clean_logs(args.days)
    elif args.clean_data:
        clean_old_data(args.days)
    elif args.full_cleanup:
        clean_cache()
        print()
        clean_logs()
        print()
        clean_old_data()
    elif args.optimize_db:
        optimize_database()
    elif args.remove_old_trades:
        remove_old_trades(args.days)
    elif args.disk_usage:
        analyze_disk_usage()
    elif args.large_files:
        find_large_files(args.min_size)
    elif args.kill_duplicates:
        kill_duplicate_processes()
    elif args.system_info:
        system_info()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Operation cancelled by user{Colors.NC}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
