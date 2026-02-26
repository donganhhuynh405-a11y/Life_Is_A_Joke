#!/bin/bash
# Quick Reference for Admin Scripts

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRADING BOT - ADMIN SCRIPTS QUICK REFERENCE                ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. BOT-ADMIN.PY - Bot Administration                                         │
└─────────────────────────────────────────────────────────────────────────────┘

  Interactive Menu:           ./bot-admin.py

  Bot Control:
    Start bot:                ./bot-admin.py --start
    Stop bot:                 ./bot-admin.py --stop
    Restart bot:              ./bot-admin.py --restart
    Status:                   ./bot-admin.py --status

  Logs:
    View logs:                ./bot-admin.py --logs
    Live logs:                ./bot-admin.py --logs --live
    Last N lines:             ./bot-admin.py --logs --lines 100
    Search logs:              ./bot-admin.py --logs --search ERROR

  Maintenance:
    Update bot:               ./bot-admin.py --update
    Quick diagnostics:        ./bot-admin.py --diagnostics
    Edit config:              ./bot-admin.py --config

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. BOT-MAINTENANCE.PY - Maintenance & Cleanup                                │
└─────────────────────────────────────────────────────────────────────────────┘

  Interactive Menu:           ./bot-maintenance.py

  Cleanup:
    Clean cache:              ./bot-maintenance.py --clean-cache
    Clean logs:               ./bot-maintenance.py --clean-logs --days 7
    Clean data:               ./bot-maintenance.py --clean-data --days 30
    Full cleanup:             ./bot-maintenance.py --full-cleanup

  Database:
    Optimize DB:              ./bot-maintenance.py --optimize-db
    Remove old trades:        ./bot-maintenance.py --remove-old-trades --days 90

  Analysis:
    Disk usage:               ./bot-maintenance.py --disk-usage
    Large files:              ./bot-maintenance.py --large-files --min-size 10
    System info:              ./bot-maintenance.py --system-info

  Process:
    Kill duplicates:          ./bot-maintenance.py --kill-duplicates

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. BOT-DIAGNOSTICS.PY - Diagnostics & Analysis                               │
└─────────────────────────────────────────────────────────────────────────────┘

  Interactive Menu:           ./bot-diagnostics.py

  Performance:
    Analyze trades:           ./bot-diagnostics.py --analyze-trades
    Diagnose positions:       ./bot-diagnostics.py --diagnose-positions
    Weekly report:            ./bot-diagnostics.py --weekly-report
    Monthly report:           ./bot-diagnostics.py --monthly-report

  Health:
    Health check:             ./bot-diagnostics.py --health-check
    Check config:             ./bot-diagnostics.py --check-config

  Connectivity:
    Test all:                 ./bot-diagnostics.py --test-connectivity
    Test exchange:            ./bot-diagnostics.py --test-exchange
    Test telegram:            ./bot-diagnostics.py --test-telegram

  AI & Database:
    Test AI:                  ./bot-diagnostics.py --test-ai
    Database query:           ./bot-diagnostics.py --database-query

┌─────────────────────────────────────────────────────────────────────────────┐
│ COMMON WORKFLOWS                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

  Daily Check:
    ./bot-admin.py --status
    ./bot-admin.py --logs --lines 50

  Weekly Maintenance:
    ./bot-diagnostics.py --weekly-report
    ./bot-maintenance.py --clean-logs --days 7
    ./bot-maintenance.py --optimize-db

  Troubleshooting:
    ./bot-diagnostics.py --health-check
    ./bot-admin.py --logs --search ERROR
    ./bot-diagnostics.py --test-connectivity

  Full Update:
    ./bot-admin.py --update

┌─────────────────────────────────────────────────────────────────────────────┐
│ AUTOMATION EXAMPLES                                                          │
└─────────────────────────────────────────────────────────────────────────────┘

  Crontab Examples:
    # Daily cleanup at 2 AM
    0 2 * * * /path/to/scripts/bot-maintenance.py --clean-logs --days 7

    # Weekly report on Sunday at 9 AM
    0 9 * * 0 /path/to/scripts/bot-diagnostics.py --weekly-report

    # Database optimization every Monday at 3 AM
    0 3 * * 1 /path/to/scripts/bot-maintenance.py --optimize-db

┌─────────────────────────────────────────────────────────────────────────────┐
│ HELP & DOCUMENTATION                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

  Get help:                   ./bot-admin.py --help
                              ./bot-maintenance.py --help
                              ./bot-diagnostics.py --help

  Full documentation:         cat ADMIN_SCRIPTS_README.md

╔══════════════════════════════════════════════════════════════════════════════╗
║ Color Codes: 🔵 INFO  🟢 SUCCESS  🟡 WARNING  🔴 ERROR                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF
