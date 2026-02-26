#!/usr/bin/env python3
"""
Trading Bot - Exchange API Connectivity Test Script
Tests connectivity and authentication with cryptocurrency exchanges.
Supports Binance (legacy) and CCXT (100+ exchanges).
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path


class Colors:
    """ANSI color codes"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
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


def load_env_file():
    """Load environment variables from .env file"""
    config_dir = os.environ.get('CONFIG_DIR', '/etc/trading-bot')
    env_file = os.path.join(config_dir, '.env')
    
    # Try current directory if config dir doesn't exist
    if not os.path.isfile(env_file):
        env_file = '.env'
    
    if os.path.isfile(env_file):
        print_info(f"Loading environment from: {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print_success("Environment variables loaded")
    else:
        print_warning(f"Environment file not found: {env_file}")


def test_basic_connectivity():
    """Test basic internet connectivity"""
    print_info("\n=== Testing Basic Connectivity ===")
    
    try:
        import socket
        # Test connection to a common crypto exchange
        socket.create_connection(("www.google.com", 443), timeout=5)
        print_success("Internet connectivity: OK")
        return True
    except Exception as e:
        print_error(f"Internet connectivity failed: {str(e)}")
        return False


def test_binance_api_public():
    """Test Binance API public endpoints"""
    print_info("\n=== Testing Binance Public API ===")
    
    try:
        import requests
        
        # Test API endpoint
        testnet = os.environ.get('BINANCE_TESTNET', 'false').lower() == 'true'
        
        if testnet:
            base_url = "https://testnet.binance.vision/api"
            print_info("Using Binance TESTNET")
        else:
            base_url = "https://api.binance.com/api"
            print_info("Using Binance PRODUCTION")
        
        # Test server time
        print_info("Testing server time endpoint...")
        response = requests.get(f"{base_url}/v3/time", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            server_time = datetime.fromtimestamp(data['serverTime'] / 1000)
            print_success(f"Server time: {server_time}")
        else:
            print_error(f"Server time request failed: {response.status_code}")
            return False
        
        # Test exchange info
        print_info("Testing exchange info endpoint...")
        response = requests.get(f"{base_url}/v3/exchangeInfo", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            symbols_count = len(data.get('symbols', []))
            print_success(f"Exchange info: {symbols_count} trading pairs available")
        else:
            print_error(f"Exchange info request failed: {response.status_code}")
            return False
        
        # Test ticker price
        symbol = os.environ.get('DEFAULT_SYMBOL', 'BTCUSDT')
        print_info(f"Testing ticker price for {symbol}...")
        response = requests.get(f"{base_url}/v3/ticker/price", params={'symbol': symbol}, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"{symbol} price: {data.get('price')}")
        else:
            print_error(f"Ticker price request failed: {response.status_code}")
            return False
        
        return True
        
    except ImportError:
        print_error("'requests' library not installed. Run: pip install requests")
        return False
    except Exception as e:
        print_error(f"API test failed: {str(e)}")
        return False


def test_binance_api_authenticated():
    """Test Binance API authenticated endpoints"""
    print_info("\n=== Testing Binance Authenticated API ===")
    
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')
    
    if not api_key or api_key.startswith('your_'):
        print_warning("API key not configured - skipping authenticated tests")
        return None
    
    if not api_secret or api_secret.startswith('your_'):
        print_warning("API secret not configured - skipping authenticated tests")
        return None
    
    try:
        from binance.client import Client
        from binance.exceptions import BinanceAPIException
        
        testnet = os.environ.get('BINANCE_TESTNET', 'false').lower() == 'true'
        
        print_info("Initializing Binance client...")
        client = Client(api_key, api_secret, testnet=testnet)
        
        # Test account status
        print_info("Testing account status...")
        try:
            account = client.get_account()
            print_success("Account status: OK")
            print_info(f"  Can trade: {account.get('canTrade', False)}")
            print_info(f"  Can withdraw: {account.get('canWithdraw', False)}")
            print_info(f"  Can deposit: {account.get('canDeposit', False)}")
            
            # Show balances
            balances = [b for b in account.get('balances', []) if float(b['free']) > 0 or float(b['locked']) > 0]
            if balances:
                print_info(f"  Non-zero balances: {len(balances)}")
                for balance in balances[:5]:  # Show first 5
                    total = float(balance['free']) + float(balance['locked'])
                    print_info(f"    {balance['asset']}: {total}")
            else:
                print_warning("  No balances found")
        
        except BinanceAPIException as e:
            print_error(f"Account status failed: {e.message}")
            return False
        
        # Test API permissions
        print_info("Testing API permissions...")
        try:
            api_perms = client.get_account_api_permissions()
            print_success("API permissions retrieved")
            print_info(f"  IP restrict: {api_perms.get('ipRestrict', False)}")
            print_info(f"  Trading enabled: {api_perms.get('enableSpotAndMarginTrading', False)}")
            print_info(f"  Withdrawals enabled: {api_perms.get('enableWithdrawals', False)}")
        except BinanceAPIException as e:
            print_warning(f"Could not retrieve API permissions: {e.message}")
        
        # Test market data with authentication
        print_info("Testing authenticated market data...")
        try:
            symbol = os.environ.get('DEFAULT_SYMBOL', 'BTCUSDT')
            ticker = client.get_symbol_ticker(symbol=symbol)
            print_success(f"{symbol} ticker: {ticker.get('price')}")
        except BinanceAPIException as e:
            print_error(f"Market data failed: {e.message}")
            return False
        
        return True
        
    except ImportError:
        print_error("'python-binance' library not installed. Run: pip install python-binance")
        return None
    except Exception as e:
        print_error(f"Authentication test failed: {str(e)}")
        return False


def test_ccxt_exchange():
    """Test CCXT exchange connectivity"""
    print_info("\n=== Testing CCXT Exchange ===")
    
    exchange_id = os.environ.get('EXCHANGE_ID', 'binance')
    api_key = os.environ.get('EXCHANGE_API_KEY', '')
    api_secret = os.environ.get('EXCHANGE_API_SECRET', '')
    testnet = os.environ.get('EXCHANGE_TESTNET', 'false').lower() == 'true'
    
    try:
        import ccxt
        
        print_info(f"Exchange: {exchange_id}")
        print_info(f"Testnet: {testnet}")
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        config = {
            'enableRateLimit': True,
        }
        
        if testnet:
            config['sandbox'] = True
        
        if api_key and not api_key.startswith('your_'):
            config['apiKey'] = api_key
            config['secret'] = api_secret
        
        exchange = exchange_class(config)
        
        # Test public API - fetch ticker
        print_info("Testing public API (fetchTicker)...")
        try:
            symbol = os.environ.get('DEFAULT_SYMBOL', 'BTC/USDT')
            ticker = exchange.fetchTicker(symbol)
            print_success(f"{symbol} price: {ticker.get('last', 'N/A')}")
        except Exception as e:
            print_error(f"fetchTicker failed: {str(e)}")
            return False
        
        # Test markets
        print_info("Testing markets...")
        try:
            exchange.load_markets()
            markets_count = len(exchange.markets)
            print_success(f"Loaded {markets_count} trading pairs")
        except Exception as e:
            print_error(f"load_markets failed: {str(e)}")
            return False
        
        # Test authenticated API if credentials provided
        if api_key and not api_key.startswith('your_'):
            print_info("Testing authenticated API (fetchBalance)...")
            try:
                balance = exchange.fetchBalance()
                total_balance = balance.get('total', {})
                non_zero = {k: v for k, v in total_balance.items() if v and float(v) > 0}
                
                if non_zero:
                    print_success(f"Account balance retrieved: {len(non_zero)} non-zero assets")
                    for asset, amount in list(non_zero.items())[:5]:
                        print_info(f"  {asset}: {amount}")
                else:
                    print_warning("No non-zero balances found")
                    
            except Exception as e:
                print_error(f"fetchBalance failed: {str(e)}")
                return False
        else:
            print_warning("API credentials not configured - skipping authenticated tests")
        
        return True
        
    except ImportError:
        print_error("'ccxt' library not installed. Run: pip install ccxt")
        return False
    except Exception as e:
        print_error(f"CCXT test failed: {str(e)}")
        return False


def test_api_rate_limits():
    """Check API rate limit information"""
    print_info("\n=== API Rate Limit Information ===")
    
    use_ccxt = os.environ.get('USE_CCXT', 'false').lower() == 'true'
    
    if use_ccxt:
        print_info("Using CCXT - rate limits handled automatically")
        print_info("  enableRateLimit is set to True")
        print_info("  CCXT will throttle requests as needed")
        return True
    
    try:
        import requests
        
        testnet = os.environ.get('BINANCE_TESTNET', 'false').lower() == 'true'
        base_url = "https://testnet.binance.vision/api" if testnet else "https://api.binance.com/api"
        
        response = requests.get(f"{base_url}/v3/exchangeInfo", timeout=10)
        
        if response.status_code == 200:
            # Check rate limit headers
            weight = response.headers.get('X-MBX-USED-WEIGHT-1M', 'N/A')
            order_count = response.headers.get('X-MBX-ORDER-COUNT-10S', 'N/A')
            
            print_info(f"Used weight (1 min): {weight}")
            print_info(f"Order count (10 sec): {order_count}")
            
            data = response.json()
            rate_limits = data.get('rateLimits', [])
            
            print_info("\nRate limits:")
            for limit in rate_limits:
                limit_type = limit.get('rateLimitType', 'UNKNOWN')
                interval = limit.get('interval', 'UNKNOWN')
                interval_num = limit.get('intervalNum', 1)
                max_limit = limit.get('limit', 'UNKNOWN')
                
                print_info(f"  {limit_type}: {max_limit} per {interval_num} {interval}")
            
            return True
        else:
            print_error(f"Could not retrieve rate limit info: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Rate limit check failed: {str(e)}")
        return False


def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 70)
    print("Connectivity Test Summary")
    print("=" * 70)
    
    total_tests = len([r for r in results.values() if r is not None])
    passed_tests = len([r for r in results.values() if r is True])
    failed_tests = len([r for r in results.values() if r is False])
    skipped_tests = len([r for r in results.values() if r is None])
    
    print(f"{Colors.GREEN}Passed:{Colors.NC}  {passed_tests}/{total_tests}")
    print(f"{Colors.RED}Failed:{Colors.NC}  {failed_tests}/{total_tests}")
    if skipped_tests > 0:
        print(f"{Colors.YELLOW}Skipped:{Colors.NC} {skipped_tests}")
    
    print()
    
    if failed_tests == 0 and passed_tests > 0:
        print(f"{Colors.GREEN}All connectivity tests passed!{Colors.NC}")
        return 0
    elif failed_tests == 0:
        print(f"{Colors.YELLOW}No tests could be run (API not configured){Colors.NC}")
        return 0
    else:
        print(f"{Colors.RED}Some connectivity tests failed!{Colors.NC}")
        return 1


def main():
    """Main entry point"""
    print("=" * 70)
    print("Trading Bot - Exchange API Connectivity Test")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load environment
    load_env_file()
    
    # Check which mode to use
    use_ccxt = os.environ.get('USE_CCXT', 'false').lower() == 'true'
    
    # Run tests
    results = {}
    results['basic_connectivity'] = test_basic_connectivity()
    
    if use_ccxt:
        print_info("\nUsing CCXT mode for multi-exchange support")
        results['ccxt_exchange'] = test_ccxt_exchange()
    else:
        print_info("\nUsing Binance legacy mode")
        results['public_api'] = test_binance_api_public()
        results['authenticated_api'] = test_binance_api_authenticated()
    
    results['rate_limits'] = test_api_rate_limits()
    
    # Print summary
    exit_code = print_summary(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
