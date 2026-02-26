#!/bin/bash
# ==============================================================================
# Trading Bot - Automated Backup Cron Setup Script
# ==============================================================================
# Configures automated backups using cron
# ==============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
APP_NAME="${APP_NAME:-trading-bot}"
BACKUP_SCHEDULE="${BACKUP_SCHEDULE:-0 2 * * *}"  # Default: 2 AM daily
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_SCRIPT="${SCRIPT_DIR}/backup.sh"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handler
error_exit() {
    log_error "$1"
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root or with sudo"
    fi
}

# Check if backup script exists
check_backup_script() {
    log_info "Checking for backup script..."
    
    if [ ! -f "$BACKUP_SCRIPT" ]; then
        # Try to find it in the deployment directory
        local alt_path="/opt/trading-bot/deployment/scripts/backup.sh"
        if [ -f "$alt_path" ]; then
            BACKUP_SCRIPT="$alt_path"
            log_info "Found backup script at: $BACKUP_SCRIPT"
        else
            error_exit "Backup script not found at: $BACKUP_SCRIPT"
        fi
    else
        log_success "Backup script found: $BACKUP_SCRIPT"
    fi
    
    # Make sure it's executable
    chmod +x "$BACKUP_SCRIPT"
}

# Parse cron schedule
parse_schedule() {
    log_info "Parsing backup schedule..."
    
    local schedule="$1"
    
    # Validate cron format (basic validation)
    if [[ ! $schedule =~ ^[0-9*,/-]+[[:space:]]+[0-9*,/-]+[[:space:]]+[0-9*,/-]+[[:space:]]+[0-9*,/-]+[[:space:]]+[0-9*,/-]+$ ]]; then
        log_warning "Invalid cron format. Using default: 0 2 * * *"
        schedule="0 2 * * *"
    fi
    
    echo "$schedule"
}

# Display schedule in human-readable format
describe_schedule() {
    local schedule="$1"
    
    case "$schedule" in
        "0 2 * * *")
            echo "Daily at 2:00 AM"
            ;;
        "0 */6 * * *")
            echo "Every 6 hours"
            ;;
        "0 0 * * 0")
            echo "Weekly on Sunday at midnight"
            ;;
        "0 0 1 * *")
            echo "Monthly on the 1st at midnight"
            ;;
        *)
            echo "Custom schedule: $schedule"
            ;;
    esac
}

# Setup cron job for root user
setup_root_cron() {
    local schedule="$1"
    local description=$(describe_schedule "$schedule")
    
    log_info "Setting up cron job for root user..."
    log_info "Schedule: $description"
    
    # Create temporary cron file
    local temp_cron=$(mktemp)
    
    # Get current crontab
    crontab -l > "$temp_cron" 2>/dev/null || true
    
    # Remove any existing trading bot backup jobs
    sed -i "/# Trading Bot - Automated Backup/d" "$temp_cron"
    sed -i "\|$BACKUP_SCRIPT|d" "$temp_cron"
    
    # Add new cron job
    cat >> "$temp_cron" << EOF

# Trading Bot - Automated Backup
# Schedule: $description
$schedule /bin/bash $BACKUP_SCRIPT >> /var/log/trading-bot/backup.log 2>&1
EOF
    
    # Install new crontab
    crontab "$temp_cron"
    rm "$temp_cron"
    
    log_success "Cron job installed for root user"
}

# Setup cron job for trading bot user
setup_user_cron() {
    local schedule="$1"
    local user="tradingbot"
    local description=$(describe_schedule "$schedule")
    
    log_info "Setting up cron job for $user user..."
    
    # Check if user exists
    if ! id "$user" &>/dev/null; then
        log_warning "User $user does not exist. Skipping user cron setup."
        return
    fi
    
    # Create temporary cron file
    local temp_cron=$(mktemp)
    
    # Get current crontab for user
    crontab -u "$user" -l > "$temp_cron" 2>/dev/null || true
    
    # Remove any existing trading bot backup jobs
    sed -i "/# Trading Bot - Automated Backup/d" "$temp_cron"
    sed -i "\|$BACKUP_SCRIPT|d" "$temp_cron"
    
    # Add new cron job
    cat >> "$temp_cron" << EOF

# Trading Bot - Automated Backup
# Schedule: $description
$schedule /bin/bash $BACKUP_SCRIPT >> /var/log/trading-bot/backup.log 2>&1
EOF
    
    # Install new crontab
    crontab -u "$user" "$temp_cron"
    rm "$temp_cron"
    
    log_success "Cron job installed for $user user"
}

# Create log directory for backup logs
setup_log_directory() {
    log_info "Setting up log directory..."
    
    local log_dir="/var/log/trading-bot"
    
    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
    fi
    
    # Create backup log file
    touch "${log_dir}/backup.log"
    
    # Set permissions
    chown -R tradingbot:tradingbot "$log_dir" 2>/dev/null || true
    chmod 750 "$log_dir"
    chmod 644 "${log_dir}/backup.log"
    
    log_success "Log directory configured: $log_dir"
}

# Setup logrotate for backup logs
setup_logrotate() {
    log_info "Setting up log rotation for backup logs..."
    
    local logrotate_config="/etc/logrotate.d/trading-bot-backup"
    
    cat > "$logrotate_config" << 'EOF'
/var/log/trading-bot/backup.log {
    weekly
    rotate 12
    compress
    delaycompress
    missingok
    notifempty
    create 0644 tradingbot tradingbot
}
EOF
    
    chmod 644 "$logrotate_config"
    
    log_success "Log rotation configured"
}

# Test backup script
test_backup() {
    log_info "Testing backup script..."
    
    if bash -n "$BACKUP_SCRIPT"; then
        log_success "Backup script syntax is valid"
    else
        log_error "Backup script has syntax errors"
        return 1
    fi
}

# Display current cron jobs
show_cron_jobs() {
    log_info "Current cron jobs:"
    echo ""
    
    echo "Root crontab:"
    crontab -l 2>/dev/null | grep -A 1 "Trading Bot" || echo "  No trading bot cron jobs"
    echo ""
    
    if id "tradingbot" &>/dev/null; then
        echo "tradingbot user crontab:"
        crontab -u tradingbot -l 2>/dev/null | grep -A 1 "Trading Bot" || echo "  No trading bot cron jobs"
        echo ""
    fi
}

# Interactive schedule selection
select_schedule() {
    log_info "Select backup schedule:"
    echo ""
    echo "  1) Daily at 2:00 AM (default)"
    echo "  2) Every 6 hours"
    echo "  3) Weekly on Sunday at midnight"
    echo "  4) Monthly on the 1st at midnight"
    echo "  5) Custom cron expression"
    echo ""
    
    read -p "Select option (1-5) [1]: " choice
    choice=${choice:-1}
    
    case $choice in
        1)
            echo "0 2 * * *"
            ;;
        2)
            echo "0 */6 * * *"
            ;;
        3)
            echo "0 0 * * 0"
            ;;
        4)
            echo "0 0 1 * *"
            ;;
        5)
            read -p "Enter cron expression (e.g., '0 2 * * *'): " custom
            echo "$custom"
            ;;
        *)
            log_warning "Invalid option. Using default."
            echo "0 2 * * *"
            ;;
    esac
}

# Main function
main() {
    log_info "Starting automated backup cron setup..."
    echo "========================================"
    
    check_root
    check_backup_script
    test_backup
    
    # Get schedule
    if [ $# -gt 0 ]; then
        local schedule="$1"
    else
        local schedule=$(select_schedule)
    fi
    
    schedule=$(parse_schedule "$schedule")
    local description=$(describe_schedule "$schedule")
    
    log_info "Selected schedule: $description ($schedule)"
    
    # Confirm
    read -p "Do you want to continue with this schedule? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Cron setup cancelled"
        exit 0
    fi
    
    # Setup
    setup_log_directory
    setup_logrotate
    setup_root_cron "$schedule"
    
    # Optionally setup for tradingbot user
    if id "tradingbot" &>/dev/null; then
        read -p "Also setup cron for tradingbot user? (yes/no): " user_confirm
        if [ "$user_confirm" = "yes" ]; then
            setup_user_cron "$schedule"
        fi
    fi
    
    echo "========================================"
    log_success "Automated backup cron setup completed!"
    echo ""
    
    # Show current cron jobs
    show_cron_jobs
    
    echo "========================================"
    log_info "Backup Information:"
    echo "  Schedule: $description"
    echo "  Script: $BACKUP_SCRIPT"
    echo "  Log file: /var/log/trading-bot/backup.log"
    echo ""
    log_info "To view backup logs:"
    echo "  tail -f /var/log/trading-bot/backup.log"
    echo ""
    log_info "To run backup manually:"
    echo "  sudo bash $BACKUP_SCRIPT"
    echo ""
    log_info "To modify schedule:"
    echo "  sudo crontab -e"
    echo "========================================"
}

# Run main function
main "$@"
