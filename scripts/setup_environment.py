#!/usr/bin/env python3
"""
Trading Bot - Interactive Environment Configuration Script
This script helps users configure their trading bot environment interactively.
"""

import os
import sys
import getpass
import re
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color


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


def validate_api_key(api_key):
    """Validate Binance API key format"""
    if len(api_key) < 16:
        return False
    if not re.match(r'^[A-Za-z0-9]+$', api_key):
        return False
    return True


def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def get_input(prompt, default=None, required=True, validator=None, secret=False):
    """Get user input with validation"""
    while True:
        if default:
            display_prompt = f"{prompt} [{default}]: "
        else:
            display_prompt = f"{prompt}: "
        
        if secret:
            value = getpass.getpass(display_prompt)
        else:
            value = input(display_prompt).strip()
        
        if not value and default:
            value = default
        
        if not value and required:
            print_error("This field is required")
            continue
        
        if not value and not required:
            return ""
        
        if validator and value:
            if not validator(value):
                print_error("Invalid input format")
                continue
        
        return value


def get_yes_no(prompt, default=None):
    """Get yes/no input"""
    while True:
        if default is not None:
            default_str = "Y/n" if default else "y/N"
            value = input(f"{prompt} [{default_str}]: ").strip().lower()
            if not value:
                return default
        else:
            value = input(f"{prompt} [y/n]: ").strip().lower()
        
        if value in ['y', 'yes']:
            return True
        elif value in ['n', 'no']:
            return False
        else:
            print_error("Please enter 'y' or 'n'")


def get_number(prompt, default=None, min_val=None, max_val=None):
    """Get numeric input with validation"""
    while True:
        value = get_input(prompt, default=str(default) if default else None, required=default is None)
        
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                print_error(f"Value must be at least {min_val}")
                continue
            if max_val is not None and num > max_val:
                print_error(f"Value must be at most {max_val}")
                continue
            return num
        except ValueError:
            print_error("Please enter a valid number")


def setup_basic_config():
    """Setup basic application configuration"""
    print_info("=== Basic Configuration ===")
    
    config = {}
    
    config['APP_NAME'] = get_input("Application name", default="trading-bot", required=False)
    config['APP_ENV'] = get_input("Environment (development/staging/production)", default="production", required=False)
    config['DEBUG'] = "true" if get_yes_no("Enable debug mode?", default=False) else "false"
    config['LOG_LEVEL'] = get_input("Log level (DEBUG/INFO/WARNING/ERROR)", default="INFO", required=False)
    
    return config


def setup_binance_config():
    """Setup Binance API configuration"""
    print_info("\n=== Binance API Configuration ===")
    print_warning("Your API credentials will be stored securely in the .env file")
    
    config = {}
    
    while True:
        api_key = get_input("Binance API Key", secret=True, validator=validate_api_key)
        if validate_api_key(api_key):
            config['BINANCE_API_KEY'] = api_key
            break
        print_error("Invalid API key format")
    
    config['BINANCE_API_SECRET'] = get_input("Binance API Secret", secret=True)
    config['BINANCE_TESTNET'] = "true" if get_yes_no("Use Binance Testnet?", default=False) else "false"
    
    return config


def setup_database_config():
    """Setup database configuration"""
    print_info("\n=== Database Configuration ===")
    
    config = {}
    
    db_types = ['sqlite', 'postgresql']
    print("Available database types:")
    for i, db_type in enumerate(db_types, 1):
        print(f"  {i}. {db_type}")
    
    db_choice = get_input("Select database type", default="1", required=False)
    if db_choice.isdigit():
        choice_num = int(db_choice)
        if 1 <= choice_num <= len(db_types):
            db_type = db_types[choice_num - 1]
        else:
            db_type = 'sqlite'
    else:
        db_type = 'sqlite'
    
    config['DB_TYPE'] = db_type
    
    if db_type == 'sqlite':
        config['DB_PATH'] = get_input("SQLite database path", default="/var/lib/trading-bot/trading_bot.db", required=False)
    else:
        config['DB_HOST'] = get_input("Database host", default="localhost", required=False)
        config['DB_PORT'] = get_input("Database port", default="5432", required=False)
        config['DB_NAME'] = get_input("Database name", default="trading_bot", required=False)
        config['DB_USER'] = get_input("Database user", default="trading_bot_user", required=False)
        config['DB_PASSWORD'] = get_input("Database password", secret=True)
    
    return config


def setup_trading_config():
    """Setup trading configuration"""
    print_info("\n=== Trading Configuration ===")
    
    config = {}
    
    config['TRADING_ENABLED'] = "true" if get_yes_no("Enable live trading?", default=False) else "false"
    config['DEFAULT_SYMBOL'] = get_input("Default trading symbol", default="BTCUSDT", required=False)
    config['MAX_POSITION_SIZE'] = str(get_number("Maximum position size", default=0.1, min_val=0.001))
    config['STOP_LOSS_PERCENTAGE'] = str(get_number("Stop loss percentage", default=2.0, min_val=0.1, max_val=50))
    config['TAKE_PROFIT_PERCENTAGE'] = str(get_number("Take profit percentage", default=5.0, min_val=0.1, max_val=100))
    
    return config


def setup_risk_management():
    """Setup risk management configuration"""
    print_info("\n=== Risk Management ===")
    
    config = {}
    
    config['MAX_DAILY_TRADES'] = str(int(get_number("Maximum daily trades", default=10, min_val=1)))
    config['MAX_OPEN_POSITIONS'] = str(int(get_number("Maximum open positions", default=3, min_val=1)))
    config['MAX_DAILY_LOSS_PERCENTAGE'] = str(get_number("Maximum daily loss percentage", default=5.0, min_val=0.1, max_val=100))
    config['POSITION_SIZE_PERCENTAGE'] = str(get_number("Position size percentage of capital", default=2.0, min_val=0.1, max_val=100))
    
    return config


def setup_notifications():
    """Setup notification configuration"""
    print_info("\n=== Notification Settings ===")
    
    config = {}
    
    config['ENABLE_NOTIFICATIONS'] = "true" if get_yes_no("Enable notifications?", default=False) else "false"
    
    if config['ENABLE_NOTIFICATIONS'] == "true":
        if get_yes_no("Configure Telegram notifications?", default=False):
            config['TELEGRAM_BOT_TOKEN'] = get_input("Telegram Bot Token", secret=True)
            config['TELEGRAM_CHAT_ID'] = get_input("Telegram Chat ID")
        
        if get_yes_no("Configure email notifications?", default=False):
            config['EMAIL_ENABLED'] = "true"
            config['SMTP_HOST'] = get_input("SMTP Host", default="smtp.gmail.com", required=False)
            config['SMTP_PORT'] = get_input("SMTP Port", default="587", required=False)
            config['SMTP_USER'] = get_input("SMTP Username")
            config['SMTP_PASSWORD'] = get_input("SMTP Password", secret=True)
            config['ALERT_EMAIL'] = get_input("Alert email address", validator=validate_email)
    
    return config


def write_env_file(config, output_path):
    """Write configuration to .env file"""
    print_info(f"\nWriting configuration to {output_path}")
    
    # Read template if exists
    template_path = Path(__file__).parent.parent / ".env.template"
    
    if template_path.exists():
        with open(template_path, 'r') as f:
            template_content = f.read()
    else:
        template_content = ""
    
    # Update template with user values
    env_content = template_content
    for key, value in config.items():
        pattern = f"{key}=.*"
        replacement = f"{key}={value}"
        if re.search(pattern, env_content):
            env_content = re.sub(pattern, replacement, env_content)
        else:
            env_content += f"\n{key}={value}"
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(env_content)
    
    # Set secure permissions
    os.chmod(output_path, 0o600)
    
    print_success(f"Configuration written to {output_path}")
    print_warning("Please review the file and make any additional changes as needed")


def main():
    """Main function"""
    print("=" * 70)
    print("Trading Bot - Interactive Environment Configuration")
    print("=" * 70)
    print()
    
    # Determine output path
    default_output = os.environ.get('CONFIG_DIR', '/etc/trading-bot') + '/.env'
    if not os.access(os.path.dirname(default_output), os.W_OK):
        default_output = str(Path.home() / '.env')
    
    output_path = get_input("Output file path", default=default_output, required=False)
    
    # Create directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Collect configuration
    config = {}
    config.update(setup_basic_config())
    config.update(setup_binance_config())
    config.update(setup_database_config())
    config.update(setup_trading_config())
    config.update(setup_risk_management())
    config.update(setup_notifications())
    
    # Write configuration
    write_env_file(config, output_path)
    
    print()
    print("=" * 70)
    print_success("Environment configuration completed!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_warning("Configuration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        sys.exit(1)
