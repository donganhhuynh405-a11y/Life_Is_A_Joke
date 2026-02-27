#!/usr/bin/env python3
"""
Trading Bot - Diagnostics Tool
Diagnostics and analysis tool with interactive menu.
"""

import os
import sys
import argparse
import subprocess
import sqlite3
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


def analyze_trades():
    """Analyze trade performance"""
    print_info("Analyzing trades...")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "analyze_trades.py"

    if script_path.exists():
        run_command(f"python3 {script_path}")
    else:
        # Manual analysis
        db_file = project_root / "data" / "trading.db"
        if not db_file.exists():
            print_warning("Trading database not found")
            return

        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()

            # Total trades
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]

            # Winning trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
            winning_trades = cursor.fetchone()[0]

            # Losing trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl < 0")
            losing_trades = cursor.fetchone()[0]

            # Total P&L
            cursor.execute("SELECT SUM(pnl) FROM trades")
            total_pnl = cursor.fetchone()[0] or 0

            # Average P&L
            cursor.execute("SELECT AVG(pnl) FROM trades")
            avg_pnl = cursor.fetchone()[0] or 0

            # Win rate
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            print(f"\n{Colors.BOLD}Trade Statistics:{Colors.NC}")
            print(f"  Total Trades: {total_trades}")
            print(f"  Winning Trades: {Colors.GREEN}{winning_trades}{Colors.NC}")
            print(f"  Losing Trades: {Colors.RED}{losing_trades}{Colors.NC}")
            print(f"  Win Rate: {win_rate:.2f}%")
            print(f"  Total P&L: ${total_pnl:.2f}")
            print(f"  Average P&L: ${avg_pnl:.2f}")

            conn.close()

        except Exception as e:
            print_error(f"Failed to analyze trades: {e}")


def diagnose_positions():
    """Diagnose current positions"""
    print_info("Diagnosing positions...")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "diagnose_positions.py"

    if script_path.exists():
        run_command(f"python3 {script_path}")
    else:
        print_warning("Position diagnostics script not found")

        # Try to read positions from database
        db_file = project_root / "data" / "trading.db"
        if db_file.exists():
            try:
                conn = sqlite3.connect(str(db_file))
                cursor = conn.cursor()

                # Check if positions table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='positions'")
                if cursor.fetchone():
                    cursor.execute("SELECT * FROM positions WHERE status='open'")
                    positions = cursor.fetchall()

                    if positions:
                        print(f"\n{Colors.BOLD}Open Positions:{Colors.NC}")
                        for pos in positions:
                            print(f"  {pos}")
                    else:
                        print_warning("No open positions found")
                else:
                    print_warning("Positions table not found in database")

                conn.close()
            except Exception as e:
                print_error(f"Failed to query positions: {e}")


def test_ai_system():
    """Test AI prediction system"""
    print_info("Testing AI system...")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "test_ai_system.py"

    if script_path.exists():
        run_command(f"python3 {script_path}")
    else:
        print_warning("AI test script not found")


def health_check():
    """Comprehensive health check"""
    print_info("Running health check...")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "health_check.py"

    if script_path.exists():
        run_command(f"python3 {script_path}")
    else:
        print_warning("Health check script not found")

        # Manual health check
        print(f"\n{Colors.BOLD}Bot Status:{Colors.NC}")
        pid = run_command("pgrep -f 'python.*main.py'", show_output=False, check=False)
        if pid:
            print(f"  {Colors.GREEN}✓{Colors.NC} Bot is running (PID: {pid})")
        else:
            print(f"  {Colors.RED}✗{Colors.NC} Bot is not running")

        print(f"\n{Colors.BOLD}Disk Space:{Colors.NC}")
        run_command("df -h / | tail -1")

        print(f"\n{Colors.BOLD}Memory:{Colors.NC}")
        run_command("free -h | grep Mem")


def test_connectivity():
    """Test connectivity to external services"""
    print_info("Testing connectivity...")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "test_connectivity.py"

    if script_path.exists():
        run_command(f"python3 {script_path}")
    else:
        print_warning("Connectivity test script not found")

        # Manual connectivity test
        print(f"\n{Colors.BOLD}Internet Connectivity:{Colors.NC}")
        try:
            result = subprocess.run(['ping', '-c', '1', '8.8.8.8'],
                                    capture_output=True, timeout=5)
            if result.returncode == 0:
                print(f"  {Colors.GREEN}✓{Colors.NC} Internet connection OK")
            else:
                print(f"  {Colors.RED}✗{Colors.NC} No internet connection")
        except Exception:
            print(f"  {Colors.RED}✗{Colors.NC} Connectivity test failed")

        print(f"\n{Colors.BOLD}Exchange API:{Colors.NC}")
        try:
            result = subprocess.run(['curl', '-s', 'https://api.binance.com/api/v3/ping'],
                                    capture_output=True, timeout=10)
            if result.returncode == 0:
                print(f"  {Colors.GREEN}✓{Colors.NC} Binance API reachable")
            else:
                print(f"  {Colors.RED}✗{Colors.NC} Binance API unreachable")
        except Exception:
            print(f"  {Colors.RED}✗{Colors.NC} API test failed")


def generate_report(report_type='weekly'):
    """Generate performance report"""
    print_info(f"Generating {report_type} report...")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "generate_weekly_report.py"

    if script_path.exists():
        run_command(f"python3 {script_path}")
    else:
        print_warning("Report generation script not found")

        # Manual report generation
        db_file = project_root / "data" / "trading.db"
        if not db_file.exists():
            print_warning("Trading database not found")
            return

        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()

            # Determine date range
            if report_type == 'weekly':
                days = 7
            elif report_type == 'monthly':
                days = 30
            else:
                days = 7

            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_timestamp = int(cutoff_date.timestamp())

            # Get trades in period
            cursor.execute(
                "SELECT COUNT(*), SUM(pnl) FROM trades WHERE timestamp > ?", (cutoff_timestamp,))
            trade_count, total_pnl = cursor.fetchone()
            total_pnl = total_pnl or 0

            print(f"\n{Colors.BOLD}{report_type.title()} Report:{Colors.NC}")
            print(
                f"  Period: {cutoff_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
            print(f"  Total Trades: {trade_count}")
            print(f"  Total P&L: ${total_pnl:.2f}")

            conn.close()

        except Exception as e:
            print_error(f"Failed to generate report: {e}")


def database_query():
    """Interactive database query"""
    print_info("Database Query")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    db_file = project_root / "data" / "trading.db"

    if not db_file.exists():
        print_error("Database not found!")
        return

    print("\nAvailable queries:")
    print("  1. Show all tables")
    print("  2. Show recent trades")
    print("  3. Show open positions")
    print("  4. Show trade statistics")
    print("  5. Custom SQL query")
    print()

    choice = input("Select query: ").strip()

    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        if choice == '1':
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"\n{Colors.BOLD}Tables:{Colors.NC}")
            for table in tables:
                print(f"  - {table[0]}")

        elif choice == '2':
            cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10")
            trades = cursor.fetchall()
            print(f"\n{Colors.BOLD}Recent Trades:{Colors.NC}")
            for trade in trades:
                print(f"  {trade}")

        elif choice == '3':
            cursor.execute("SELECT * FROM positions WHERE status='open'")
            positions = cursor.fetchall()
            print(f"\n{Colors.BOLD}Open Positions:{Colors.NC}")
            if positions:
                for pos in positions:
                    print(f"  {pos}")
            else:
                print("  No open positions")

        elif choice == '4':
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl
                FROM trades
            """)
            stats = cursor.fetchone()
            print(f"\n{Colors.BOLD}Trade Statistics:{Colors.NC}")
            print(f"  Total Trades: {stats[0]}")
            print(f"  Winning Trades: {stats[1]}")
            print(f"  Losing Trades: {stats[2]}")
            print(f"  Total P&L: ${stats[3]:.2f}")
            print(f"  Average P&L: ${stats[4]:.2f}")

        elif choice == '5':
            query = input("Enter SQL query (SELECT only): ").strip()
            # Validate query - only allow SELECT statements
            if not query.upper().startswith('SELECT'):
                print_error("Only SELECT queries are allowed for security reasons")
            else:
                try:
                    cursor.execute(query)
                    results = cursor.fetchall()
                    print(f"\n{Colors.BOLD}Results:{Colors.NC}")
                    for row in results:
                        print(f"  {row}")
                except Exception as e:
                    print_error(f"Query failed: {e}")

        conn.close()

    except Exception as e:
        print_error(f"Query failed: {e}")


def check_configuration():
    """Check configuration files"""
    print_info("Checking configuration...")
    print("=" * 80)

    project_root = Path(__file__).parent.parent

    config_files = {
        'config.yaml': project_root / 'config.yaml',
        '.env': project_root / '.env',
        'requirements.txt': project_root / 'requirements.txt',
    }

    print(f"\n{Colors.BOLD}Configuration Files:{Colors.NC}")
    for name, path in config_files.items():
        if path.exists():
            size = path.stat().st_size
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            print(
                f"  {
                    Colors.GREEN}✓{
                    Colors.NC} {
                    name:20} ({size} bytes, modified {
                    mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"  {Colors.RED}✗{Colors.NC} {name:20} (not found)")

    # Check environment variables
    print(f"\n{Colors.BOLD}Environment Variables:{Colors.NC}")
    important_vars = [
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET',
        'TELEGRAM_TOKEN',
        'ENVIRONMENT',
    ]

    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {Colors.GREEN}✓{Colors.NC} {var:25} (set)")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.NC} {var:25} (not set)")


def test_exchange_api():
    """Test exchange API connectivity"""
    print_info("Testing exchange API...")
    print("=" * 80)

    try:
        # Try to import ccxt
        import ccxt

        print(f"\n{Colors.BOLD}CCXT Version:{Colors.NC} {ccxt.__version__}")

        # Test Binance
        print(f"\n{Colors.BOLD}Binance API:{Colors.NC}")
        exchange = ccxt.binance({
            'apiKey': os.environ.get('BINANCE_API_KEY', ''),
            'secret': os.environ.get('BINANCE_API_SECRET', ''),
            'enableRateLimit': True,
        })

        # Test public endpoint
        try:
            ticker = exchange.fetch_ticker('BTC/USDT')
            print(f"  {Colors.GREEN}✓{Colors.NC} Public API OK (BTC/USDT: ${ticker['last']:.2f})")
        except Exception as e:
            print(f"  {Colors.RED}✗{Colors.NC} Public API failed: {e}")

        # Test private endpoint
        if os.environ.get('BINANCE_API_KEY'):
            try:
                exchange.fetch_balance()
                print(f"  {Colors.GREEN}✓{Colors.NC} Private API OK (Authenticated)")
            except Exception as e:
                print(f"  {Colors.RED}✗{Colors.NC} Private API failed: {e}")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.NC} API key not set, skipping private API test")

    except ImportError:
        print_error("CCXT library not installed")
        print("Install with: pip install ccxt")


def test_telegram_bot():
    """Test Telegram bot connectivity"""
    print_info("Testing Telegram bot...")
    print("=" * 80)

    token = os.environ.get('TELEGRAM_TOKEN')

    if not token:
        print_warning("TELEGRAM_TOKEN not set")
        return

    try:
        import requests

        # Test bot API
        response = requests.get(f"https://api.telegram.org/bot{token}/getMe")

        if response.status_code == 200:
            data = response.json()
            if data['ok']:
                bot = data['result']
                print(f"  {Colors.GREEN}✓{Colors.NC} Bot API OK")
                print(f"  Bot Name: {bot['first_name']}")
                print(f"  Bot Username: @{bot['username']}")
            else:
                print(f"  {Colors.RED}✗{Colors.NC} Bot API error: {data.get('description')}")
        else:
            print(f"  {Colors.RED}✗{Colors.NC} HTTP error: {response.status_code}")

    except ImportError:
        print_error("requests library not installed")
    except Exception as e:
        print_error(f"Failed to test Telegram bot: {e}")


def interactive_menu():
    """Display interactive menu"""
    while True:
        os.system('clear' if os.name != 'nt' else 'cls')

        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.NC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'Trading Bot - Diagnostics Tool':^70}{Colors.NC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.NC}")
        print()

        print(f"{Colors.BOLD}Performance Analysis:{Colors.NC}")
        print(f"  {Colors.GREEN}1{Colors.NC}. Analyze trades (P&L, win rate, statistics)")
        print(f"  {Colors.GREEN}2{Colors.NC}. Diagnose positions")
        print(f"  {Colors.GREEN}3{Colors.NC}. Generate weekly report")
        print(f"  {Colors.GREEN}4{Colors.NC}. Generate monthly report")

        print(f"\n{Colors.BOLD}System Health:{Colors.NC}")
        print(f"  {Colors.CYAN}5{Colors.NC}. Comprehensive health check")
        print(f"  {Colors.CYAN}6{Colors.NC}. Check configuration")

        print(f"\n{Colors.BOLD}Connectivity Tests:{Colors.NC}")
        print(f"  {Colors.BLUE}7{Colors.NC}. Test all connectivity")
        print(f"  {Colors.BLUE}8{Colors.NC}. Test exchange API")
        print(f"  {Colors.BLUE}9{Colors.NC}. Test Telegram bot")

        print(f"\n{Colors.BOLD}AI & ML:{Colors.NC}")
        print(f"  {Colors.MAGENTA}10{Colors.NC}. Test AI prediction system")

        print(f"\n{Colors.BOLD}Database:{Colors.NC}")
        print(f"  {Colors.YELLOW}11{Colors.NC}. Database queries")

        print(f"\n  {Colors.RED}0{Colors.NC}. Exit")
        print()

        choice = input(f"{Colors.BOLD}Select option: {Colors.NC}").strip()

        if choice == '1':
            analyze_trades()
            input("\nPress Enter to continue...")
        elif choice == '2':
            diagnose_positions()
            input("\nPress Enter to continue...")
        elif choice == '3':
            generate_report('weekly')
            input("\nPress Enter to continue...")
        elif choice == '4':
            generate_report('monthly')
            input("\nPress Enter to continue...")
        elif choice == '5':
            health_check()
            input("\nPress Enter to continue...")
        elif choice == '6':
            check_configuration()
            input("\nPress Enter to continue...")
        elif choice == '7':
            test_connectivity()
            input("\nPress Enter to continue...")
        elif choice == '8':
            test_exchange_api()
            input("\nPress Enter to continue...")
        elif choice == '9':
            test_telegram_bot()
            input("\nPress Enter to continue...")
        elif choice == '10':
            test_ai_system()
            input("\nPress Enter to continue...")
        elif choice == '11':
            database_query()
            input("\nPress Enter to continue...")
        elif choice == '0':
            print_info("Goodbye!")
            break
        else:
            print_error("Invalid option!")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description='Trading Bot Diagnostics Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     # Interactive menu
  %(prog)s --analyze-trades    # Analyze trade performance
  %(prog)s --diagnose-positions  # Diagnose positions
  %(prog)s --health-check      # Comprehensive health check
  %(prog)s --test-connectivity # Test all connectivity
  %(prog)s --test-exchange     # Test exchange API
  %(prog)s --test-telegram     # Test Telegram bot
  %(prog)s --test-ai           # Test AI system
  %(prog)s --weekly-report     # Generate weekly report
  %(prog)s --monthly-report    # Generate monthly report
        """
    )

    parser.add_argument('--analyze-trades', action='store_true', help='Analyze trade performance')
    parser.add_argument(
        '--diagnose-positions',
        action='store_true',
        help='Diagnose current positions')
    parser.add_argument('--test-ai', action='store_true', help='Test AI prediction system')
    parser.add_argument('--health-check', action='store_true', help='Comprehensive health check')
    parser.add_argument('--test-connectivity', action='store_true', help='Test connectivity')
    parser.add_argument('--test-exchange', action='store_true', help='Test exchange API')
    parser.add_argument('--test-telegram', action='store_true', help='Test Telegram bot')
    parser.add_argument('--weekly-report', action='store_true', help='Generate weekly report')
    parser.add_argument('--monthly-report', action='store_true', help='Generate monthly report')
    parser.add_argument('--check-config', action='store_true', help='Check configuration')
    parser.add_argument('--database-query', action='store_true', help='Interactive database query')

    args = parser.parse_args()

    # If no arguments, show interactive menu
    if len(sys.argv) == 1:
        interactive_menu()
        return

    # Execute commands
    if args.analyze_trades:
        analyze_trades()
    elif args.diagnose_positions:
        diagnose_positions()
    elif args.test_ai:
        test_ai_system()
    elif args.health_check:
        health_check()
    elif args.test_connectivity:
        test_connectivity()
    elif args.test_exchange:
        test_exchange_api()
    elif args.test_telegram:
        test_telegram_bot()
    elif args.weekly_report:
        generate_report('weekly')
    elif args.monthly_report:
        generate_report('monthly')
    elif args.check_config:
        check_configuration()
    elif args.database_query:
        database_query()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Operation cancelled by user{Colors.NC}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
