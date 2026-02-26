# Trading Bot Admin Scripts

Three comprehensive multifunctional Python scripts for managing, maintaining, and diagnosing the trading bot.

## Overview

- **bot-admin.py** - Main administration tool for bot management
- **bot-maintenance.py** - Maintenance and cleanup operations
- **bot-diagnostics.py** - Diagnostics and analysis tools

## Installation

All scripts are already executable and ready to use:

```bash
cd /path/to/life_is_a_joke/scripts
```

## Usage

### Interactive Mode

Run any script without arguments to enter interactive menu mode:

```bash
./bot-admin.py          # Interactive admin menu
./bot-maintenance.py    # Interactive maintenance menu
./bot-diagnostics.py    # Interactive diagnostics menu
```

### Command Line Mode

Use command-line arguments for scripting and automation:

```bash
# View help
./bot-admin.py --help
./bot-maintenance.py --help
./bot-diagnostics.py --help
```

---

## 1. bot-admin.py - Bot Administration

Main administration tool with bot management, updates, and configuration.

### Features

**Bot Management:**
- Start/Stop/Restart bot
- Check bot status
- Enable/Disable autostart
- View live logs
- Search logs
- Quick diagnostics

**Updates & Configuration:**
- Update bot (git pull + pip install + restart)
- Change git repository source
- Edit configuration files

### Usage Examples

```bash
# Interactive menu
./bot-admin.py

# Start/stop/restart bot
./bot-admin.py --start
./bot-admin.py --stop
./bot-admin.py --restart

# Check status
./bot-admin.py --status

# View logs
./bot-admin.py --logs                    # Last 50 lines
./bot-admin.py --logs --lines 100        # Last 100 lines
./bot-admin.py --logs --live             # Live stream (Ctrl+C to stop)
./bot-admin.py --logs --search ERROR     # Search for errors

# Update bot
./bot-admin.py --update                  # Git pull + install deps + restart

# Quick diagnostics
./bot-admin.py --diagnostics

# Edit configuration
./bot-admin.py --config
```

### Interactive Menu Options

1. Start bot
2. Stop bot
3. Restart bot
4. Bot status
5. Enable autostart
6. Disable autostart
7. Update bot (git pull & restart)
8. Change git repository
9. Edit configuration
10. View live logs
11. View last N lines
12. Search logs
13. Quick diagnostics
0. Exit

---

## 2. bot-maintenance.py - Maintenance & Cleanup

Maintenance and cleanup tool for disk space, database, and processes.

### Features

**Cleanup Operations:**
- Clean cache directories
- Clean old log files
- Clean old data files
- Full cleanup (all above)

**Database Operations:**
- Optimize database (VACUUM, ANALYZE, REINDEX)
- Remove old trades
- Remove old news
- Remove old positions

**Analysis:**
- Analyze disk usage
- Find large files
- System information

**Process Management:**
- Kill duplicate processes

### Usage Examples

```bash
# Interactive menu
./bot-maintenance.py

# Clean operations
./bot-maintenance.py --clean-cache
./bot-maintenance.py --clean-logs              # Keep 7 days (default)
./bot-maintenance.py --clean-logs --days 14    # Keep 14 days
./bot-maintenance.py --clean-data --days 30
./bot-maintenance.py --full-cleanup            # All cleanup operations

# Database operations
./bot-maintenance.py --optimize-db
./bot-maintenance.py --remove-old-trades --days 90

# Analysis
./bot-maintenance.py --disk-usage
./bot-maintenance.py --large-files               # Files > 10MB
./bot-maintenance.py --large-files --min-size 50 # Files > 50MB
./bot-maintenance.py --system-info

# Process management
./bot-maintenance.py --kill-duplicates
```

### Interactive Menu Options

**Cleanup Operations:**
1. Clean cache directories
2. Clean old logs (7 days)
3. Clean old data files (30 days)
4. Full cleanup (all above)

**Database Operations:**
5. Optimize database (vacuum, analyze, reindex)
6. Remove old trades (90 days)
7. Remove old news (90 days)
8. Remove old positions (90 days)

**Analysis:**
9. Analyze disk usage
10. Find large files
11. System information

**Process Management:**
12. Kill duplicate processes

0. Exit

---

## 3. bot-diagnostics.py - Diagnostics & Analysis

Diagnostics and analysis tool for trades, connectivity, and system health.

### Features

**Performance Analysis:**
- Analyze trades (P&L, win rate, statistics)
- Diagnose positions
- Generate reports (weekly/monthly)

**System Health:**
- Comprehensive health check
- Check configuration

**Connectivity Tests:**
- Test all connectivity
- Test exchange API
- Test Telegram bot

**AI & ML:**
- Test AI prediction system

**Database:**
- Interactive database queries

### Usage Examples

```bash
# Interactive menu
./bot-diagnostics.py

# Performance analysis
./bot-diagnostics.py --analyze-trades
./bot-diagnostics.py --diagnose-positions
./bot-diagnostics.py --weekly-report
./bot-diagnostics.py --monthly-report

# System health
./bot-diagnostics.py --health-check
./bot-diagnostics.py --check-config

# Connectivity tests
./bot-diagnostics.py --test-connectivity    # Test all
./bot-diagnostics.py --test-exchange        # Test exchange API only
./bot-diagnostics.py --test-telegram        # Test Telegram only

# AI system
./bot-diagnostics.py --test-ai

# Database queries
./bot-diagnostics.py --database-query       # Interactive SQL queries
```

### Interactive Menu Options

**Performance Analysis:**
1. Analyze trades (P&L, win rate, statistics)
2. Diagnose positions
3. Generate weekly report
4. Generate monthly report

**System Health:**
5. Comprehensive health check
6. Check configuration

**Connectivity Tests:**
7. Test all connectivity
8. Test exchange API
9. Test Telegram bot

**AI & ML:**
10. Test AI prediction system

**Database:**
11. Database queries

0. Exit

---

## Features

### Color-Coded Output

All scripts use color-coded output for easy readability:

- 🔵 **Blue** - Information messages
- 🟢 **Green** - Success messages
- 🟡 **Yellow** - Warning messages
- 🔴 **Red** - Error messages

### Progress Indicators

Long-running operations show progress indicators to keep you informed.

### Error Handling

Comprehensive error handling with user-friendly messages.

### Standalone Operation

Each script works independently and can be used without the others.

---

## Common Use Cases

### Daily Operations

```bash
# Morning check
./bot-admin.py --status
./bot-diagnostics.py --health-check

# View recent activity
./bot-admin.py --logs --lines 100

# Check performance
./bot-diagnostics.py --analyze-trades
```

### Weekly Maintenance

```bash
# Generate report
./bot-diagnostics.py --weekly-report

# Clean old data
./bot-maintenance.py --clean-logs --days 7
./bot-maintenance.py --optimize-db

# Check system health
./bot-maintenance.py --disk-usage
./bot-diagnostics.py --test-connectivity
```

### Monthly Maintenance

```bash
# Generate report
./bot-diagnostics.py --monthly-report

# Deep cleanup
./bot-maintenance.py --full-cleanup
./bot-maintenance.py --remove-old-trades --days 90

# System analysis
./bot-maintenance.py --large-files
./bot-maintenance.py --system-info
```

### Troubleshooting

```bash
# Check what's wrong
./bot-admin.py --diagnostics
./bot-diagnostics.py --health-check

# View errors
./bot-admin.py --logs --search ERROR
./bot-admin.py --logs --search CRITICAL

# Test connectivity
./bot-diagnostics.py --test-connectivity
./bot-diagnostics.py --test-exchange

# Check configuration
./bot-diagnostics.py --check-config

# Kill duplicate processes
./bot-maintenance.py --kill-duplicates

# Restart bot
./bot-admin.py --restart
```

### Updating the Bot

```bash
# Full update process
./bot-admin.py --update

# Manual update
./bot-admin.py --stop
cd /path/to/life_is_a_joke
git pull
pip install -r requirements.txt
./bot-admin.py --start
```

---

## Automation Examples

### Cron Jobs

Add to crontab for automation:

```bash
# Daily cleanup at 2 AM
0 2 * * * /path/to/scripts/bot-maintenance.py --clean-logs --days 7

# Weekly report on Sunday at 9 AM
0 9 * * 0 /path/to/scripts/bot-diagnostics.py --weekly-report

# Database optimization every Monday at 3 AM
0 3 * * 1 /path/to/scripts/bot-maintenance.py --optimize-db

# Health check every hour
0 * * * * /path/to/scripts/bot-diagnostics.py --health-check
```

### Shell Scripts

Create custom automation scripts:

```bash
#!/bin/bash
# maintenance.sh - Run full maintenance

echo "Starting maintenance..."

# Cleanup
./bot-maintenance.py --full-cleanup

# Optimize database
./bot-maintenance.py --optimize-db

# Health check
./bot-diagnostics.py --health-check

echo "Maintenance complete!"
```

---

## Requirements

### Python Dependencies

- Python 3.8+
- Standard library modules (os, sys, argparse, subprocess, etc.)
- Optional: psutil (for enhanced system monitoring)

```bash
pip install psutil  # Optional but recommended
```

### System Requirements

- Linux/Unix system (tested on Ubuntu)
- Git (for update functionality)
- Database access (SQLite)
- Internet connection (for connectivity tests)

---

## Tips & Best Practices

1. **Use interactive mode** when learning - it's more user-friendly
2. **Use command-line mode** for automation and scripting
3. **Run health checks regularly** to catch issues early
4. **Generate reports weekly** to track performance
5. **Clean old data monthly** to save disk space
6. **Test connectivity** after configuration changes
7. **Always check status** before making changes
8. **View logs** when troubleshooting issues
9. **Backup before updates** (automatic in update process)
10. **Monitor disk space** with regular maintenance

---

## Troubleshooting

### Script Won't Run

```bash
# Make executable
chmod +x bot-admin.py bot-maintenance.py bot-diagnostics.py

# Check Python version
python3 --version  # Should be 3.8+

# Check if script exists
ls -la bot-*.py
```

### Bot Not Found

```bash
# Check if bot is running
ps aux | grep python

# Check bot files
ls -la ../src/main.py ../telegram_bot.py
```

### Database Errors

```bash
# Check database exists
ls -la ../data/*.db

# Check database permissions
chmod 664 ../data/*.db
```

### Permission Errors

```bash
# Fix script permissions
chmod +x bot-*.py

# Fix log directory permissions
mkdir -p ../logs
chmod 755 ../logs
```

---

## Contributing

Feel free to enhance these scripts with additional features:

- Add more diagnostic checks
- Implement additional cleanup operations
- Add more report types
- Improve error handling
- Add more automation options

---

## License

Part of the life_is_a_joke trading bot project.

---

## Support

For issues or questions:

1. Check script help: `./script-name.py --help`
2. Run diagnostics: `./bot-diagnostics.py --health-check`
3. View logs: `./bot-admin.py --logs`
4. Check configuration: `./bot-diagnostics.py --check-config`

---

**Last Updated:** 2026-02-07
**Version:** 1.0.0
