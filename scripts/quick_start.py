#!/usr/bin/env python3
"""
Quick start utility for the trading bot
Helps set up and validate the environment
"""
import os
import sys
import subprocess


def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_section(text):
    print(f"\n--- {text} ---")


def check_python_version():
    """Check if Python version is compatible"""
    import sys
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("✓ Python version OK")
        return True
    else:
        print("✗ Python 3.9+ required")
        return False


def check_env_file():
    """Check if .env file exists"""
    if os.path.exists('.env'):
        print("✓ .env file found")
        return True
    else:
        print("⚠ .env file not found")
        print("\nCreating template .env file...")
        with open('.env', 'w') as f:
            f.write("""# Binance Testnet API Keys (get from https://testnet.binance.vision/)
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_secret_here

# Telegram Bot (optional)
TELEGRAM_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Redis (if using external Redis)
REDIS_URL=redis://localhost:6379/0
""")
        print("✓ Created .env template - please edit with your API keys!")
        return False


def install_dependencies():
    """Install required dependencies"""
    print("Installing core dependencies...")
    packages = ['pandas', 'numpy', 'ccxt', 'pyyaml', 'cachetools', 'python-dotenv']
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q', '--upgrade'
        ] + packages)
        print(f"✓ Installed: {', '.join(packages)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def run_health_check():
    """Run system health check"""
    print("Running health check...")
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/health_check.py'],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def show_next_steps():
    """Show user what to do next"""
    print_header("NEXT STEPS")
    
    print("""
1. Edit .env file with your API keys:
   - Get Binance Testnet keys: https://testnet.binance.vision/
   - NEVER use real API keys with production funds!

2. Test the strategy:
   python test_classic_strategy.py

3. Run health check:
   python scripts/health_check.py

4. Run the bot (paper trading on testnet):
   python -m src.main

5. (Optional) Set up Telegram bot:
   - Create bot with @BotFather
   - Add TELEGRAM_TOKEN to .env

For more information, see README.md
""")


def main():
    """Main setup flow"""
    print_header("CRYPTO TRADING BOT - QUICK START")
    
    # Step 1: Check Python version
    print_section("Python Version")
    if not check_python_version():
        print("\nPlease upgrade to Python 3.9 or higher")
        return 1
    
    # Step 2: Check/create .env file
    print_section("Environment Configuration")
    env_ok = check_env_file()
    
    # Step 3: Install dependencies
    print_section("Dependencies")
    if not install_dependencies():
        print("\nPlease install dependencies manually:")
        print("pip install pandas numpy ccxt pyyaml cachetools python-dotenv")
        return 1
    
    # Step 4: Run health check
    print_section("Health Check")
    health_ok = run_health_check()
    
    # Step 5: Show next steps
    if health_ok and env_ok:
        print("\n✓ Setup complete! System ready to run.")
    elif health_ok:
        print("\n⚠ Setup mostly complete. Please edit .env file with your API keys.")
    else:
        print("\n⚠ Some components need attention. See above for details.")
    
    show_next_steps()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
