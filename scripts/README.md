# Scripts Directory

This directory contains utility scripts for managing and deploying the trading bot.

## Installation & Update Scripts

### `quick_install.sh` - Quick Installation
Download and install the bot in one command:
```bash
curl -sL https://raw.githubusercontent.com/matthew3f2eb8c4-pixel/life_is_a_joke/copilot/transfer-files-to-empty-repo/scripts/quick_install.sh | bash
```

Or run locally:
```bash
chmod +x scripts/quick_install.sh
./scripts/quick_install.sh
```

### `download_update.sh` - Download Latest Updates
Download the latest version from the repository:
```bash
chmod +x scripts/download_update.sh
./scripts/download_update.sh [destination_directory]
```

Example:
```bash
./scripts/download_update.sh /opt/trading-bot
```

### `install.sh` - Install Dependencies
Install Python dependencies and set up the environment:
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

### `update_bot.sh` - Update Running Bot (Server)
Update the bot on a production server (requires sudo):
```bash
sudo ./scripts/update_bot.sh
```

This script will:
- Clone the repository if it doesn't exist
- Stop the trading bot service
- Pull the latest changes
- Update files in the deployment directory
- Restart the bot service

### `update_env_file.sh` - Update .env Configuration
Update your existing .env file with new parameters:
```bash
./scripts/update_env_file.sh
```

Or specify a custom path:
```bash
./scripts/update_env_file.sh /custom/path/.env
```

This script will:
- Create a backup of your existing .env file
- Check for missing parameters (like NOTIFICATION_LANGUAGE)
- Add new parameters with default values
- Preserve all your existing settings

**When to use:** After updating the bot code, if new configuration parameters were added.

### `verify_and_update.sh` - Verify and Update Code
Check if your code is up to date and optionally update it:
```bash
./scripts/verify_and_update.sh
```

## Diagnostic & Monitoring Scripts

### `health_check.py` - Health Check
Check the health status of the running bot:
```bash
python3 scripts/health_check.py
```

### `test_connectivity.py` - Test Exchange Connection
Test connectivity to the exchange API:
```bash
python3 scripts/test_connectivity.py
```

### `diagnose_positions.py` - Diagnose Positions
Analyze and diagnose open positions:
```bash
python3 scripts/diagnose_positions.py
```

### `analyze_trades.py` - Analyze Trades
Analyze trading performance and statistics:
```bash
python3 scripts/analyze_trades.py
```

## Configuration & Setup Scripts

### `setup_environment.py` - Setup Environment
Interactive script to set up environment and configuration:
```bash
python3 scripts/setup_environment.py
```

### `quick_start.py` - Quick Start
Quick start guide and initial configuration:
```bash
python3 scripts/quick_start.py
```

## Testing Scripts

### `test_ai_system.py` - Test AI System
Test the AI prediction system:
```bash
python3 scripts/test_ai_system.py
```

### `backtest_sim` - Backtesting Simulation
Run backtesting simulations:
```bash
./scripts/backtest_sim
```

## Maintenance Scripts

### `reset_daily_limit.py` - Reset Daily Limits
Reset daily trading limits:
```bash
python3 scripts/reset_daily_limit.py
```

### `failover_demo.py` - Failover Demo
Demonstrate failover capabilities:
```bash
python3 scripts/failover_demo.py
```

### `generate_weekly_report.py` - Generate Reports
Generate weekly performance reports:
```bash
python3 scripts/generate_weekly_report.py
```

## Repository Information

- **Repository URL**: https://github.com/matthew3f2eb8c4-pixel/life_is_a_joke
- **Current Branch**: copilot/transfer-files-to-empty-repo

## Quick Reference

```bash
# Install the bot
./scripts/quick_install.sh

# Download updates
./scripts/download_update.sh /path/to/bot

# Update running server bot (requires sudo)
sudo ./scripts/update_bot.sh

# Check bot health
python3 scripts/health_check.py

# Test exchange connectivity
python3 scripts/test_connectivity.py

# Verify code version
./scripts/verify_and_update.sh
```

## Notes

- Most shell scripts (`.sh`) need to be made executable: `chmod +x script.sh`
- Server update scripts require sudo/root privileges
- Always backup your configuration before updating
- Test in a development environment before deploying to production
