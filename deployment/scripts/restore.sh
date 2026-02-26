#!/bin/bash
# ==============================================================================
# Trading Bot - Restore Script
# ==============================================================================
# Restores trading bot data from a backup archive
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
DATA_DIR="${DATA_DIR:-/var/lib/trading-bot}"
LOG_DIR="${LOG_DIR:-/var/log/trading-bot}"
CONFIG_DIR="${CONFIG_DIR:-/etc/trading-bot}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/trading-bot}"

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

# List available backups
list_backups() {
    log_info "Available backups in $BACKUP_DIR:"
    echo ""
    
    local backups=($(find "$BACKUP_DIR" -name "${APP_NAME}_backup_*.tar.gz" -type f 2>/dev/null | sort -r))
    
    if [ ${#backups[@]} -eq 0 ]; then
        log_error "No backups found in $BACKUP_DIR"
        exit 1
    fi
    
    local i=1
    for backup in "${backups[@]}"; do
        local size=$(du -h "$backup" | cut -f1)
        local date=$(stat -c %y "$backup" | cut -d' ' -f1,2 | cut -d'.' -f1)
        printf "%2d) %s\n" "$i" "$(basename "$backup")"
        printf "    Size: %-8s  Date: %s\n" "$size" "$date"
        ((i++))
    done
    
    echo ""
}

# Select backup to restore
select_backup() {
    local backups=($(find "$BACKUP_DIR" -name "${APP_NAME}_backup_*.tar.gz" -type f 2>/dev/null | sort -r))
    
    if [ ${#backups[@]} -eq 0 ]; then
        error_exit "No backups found"
    fi
    
    # If backup file provided as argument, use it
    if [ $# -gt 0 ]; then
        local backup_file="$1"
        
        # Check if it's a full path or just filename
        if [ ! -f "$backup_file" ]; then
            backup_file="${BACKUP_DIR}/${backup_file}"
        fi
        
        if [ -f "$backup_file" ]; then
            echo "$backup_file"
            return
        else
            error_exit "Backup file not found: $backup_file"
        fi
    fi
    
    # Interactive selection
    list_backups
    
    while true; do
        read -p "Select backup to restore (1-${#backups[@]}): " selection
        
        if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le ${#backups[@]} ]; then
            local selected_backup="${backups[$((selection-1))]}"
            echo "$selected_backup"
            return
        else
            log_error "Invalid selection. Please enter a number between 1 and ${#backups[@]}"
        fi
    done
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    
    log_info "Verifying backup integrity..."
    
    if tar -tzf "$backup_file" > /dev/null 2>&1; then
        log_success "Backup integrity verified"
        return 0
    else
        error_exit "Backup integrity check failed"
    fi
}

# Display backup contents
show_backup_info() {
    local backup_file="$1"
    
    log_info "Backup Information:"
    echo "  File: $(basename "$backup_file")"
    echo "  Size: $(du -h "$backup_file" | cut -f1)"
    echo "  Date: $(stat -c %y "$backup_file" | cut -d'.' -f1)"
    echo ""
    
    log_info "Backup Contents:"
    tar -tzf "$backup_file" | head -20
    
    local total_files=$(tar -tzf "$backup_file" | wc -l)
    if [ $total_files -gt 20 ]; then
        echo "  ... and $((total_files - 20)) more files"
    fi
    echo ""
}

# Stop application service
stop_service() {
    log_info "Stopping ${APP_NAME} service..."
    
    if systemctl is-active --quiet "${APP_NAME}.service" 2>/dev/null; then
        systemctl stop "${APP_NAME}.service"
        log_success "Service stopped"
    else
        log_info "Service is not running"
    fi
}

# Start application service
start_service() {
    log_info "Starting ${APP_NAME} service..."
    
    if systemctl is-enabled --quiet "${APP_NAME}.service" 2>/dev/null; then
        systemctl start "${APP_NAME}.service"
        
        # Wait a moment and check status
        sleep 2
        
        if systemctl is-active --quiet "${APP_NAME}.service"; then
            log_success "Service started successfully"
        else
            log_warning "Service may have failed to start. Check with: systemctl status ${APP_NAME}"
        fi
    else
        log_info "Service not enabled, skipping start"
    fi
}

# Create backup of current state before restore
backup_current_state() {
    log_info "Creating backup of current state before restore..."
    
    local pre_restore_backup="${BACKUP_DIR}/${APP_NAME}_pre_restore_$(date +%Y%m%d_%H%M%S).tar.gz"
    local temp_dir=$(mktemp -d -t pre-restore-XXXXXXXXXX)
    
    # Backup current state
    [ -d "$DATA_DIR" ] && cp -r "$DATA_DIR" "${temp_dir}/data" 2>/dev/null || true
    [ -d "$CONFIG_DIR" ] && cp -r "$CONFIG_DIR" "${temp_dir}/config" 2>/dev/null || true
    [ -d "$LOG_DIR" ] && cp -r "$LOG_DIR" "${temp_dir}/logs" 2>/dev/null || true
    
    # Create archive
    tar -czf "$pre_restore_backup" -C "$(dirname "$temp_dir")" "$(basename "$temp_dir")" 2>/dev/null
    rm -rf "$temp_dir"
    
    log_success "Current state backed up to: $pre_restore_backup"
    echo "$pre_restore_backup"
}

# Extract backup
extract_backup() {
    local backup_file="$1"
    local extract_dir="$2"
    
    log_info "Extracting backup..."
    
    mkdir -p "$extract_dir"
    tar -xzf "$backup_file" -C "$extract_dir" --strip-components=1 2>/dev/null
    
    log_success "Backup extracted to temporary location"
}

# Restore database
restore_database() {
    local extract_dir="$1"
    local db_backup_dir="${extract_dir}/database"
    
    log_info "Restoring database..."
    
    if [ ! -d "$db_backup_dir" ]; then
        log_warning "No database backup found in archive"
        return
    fi
    
    # Ensure data directory exists
    mkdir -p "$DATA_DIR"
    
    # Check database type
    local db_type="${DB_TYPE:-sqlite}"
    
    if [ "$db_type" = "sqlite" ]; then
        if [ -f "${db_backup_dir}/trading_bot.db" ]; then
            local db_path="${DB_PATH:-${DATA_DIR}/trading_bot.db}"
            cp "${db_backup_dir}/trading_bot.db" "$db_path"
            chown tradingbot:tradingbot "$db_path" 2>/dev/null || true
            log_success "SQLite database restored"
        else
            log_warning "SQLite database not found in backup"
        fi
    elif [ "$db_type" = "postgresql" ]; then
        if [ -f "${db_backup_dir}/"*.sql ]; then
            local db_name="${DB_NAME:-trading_bot}"
            local db_user="${DB_USER:-trading_bot_user}"
            local db_host="${DB_HOST:-localhost}"
            
            if command -v psql &> /dev/null; then
                PGPASSWORD="${DB_PASSWORD}" psql -h "$db_host" -U "$db_user" "$db_name" < "${db_backup_dir}/"*.sql
                log_success "PostgreSQL database restored"
            else
                log_error "psql not found - cannot restore PostgreSQL database"
            fi
        else
            log_warning "PostgreSQL database dump not found in backup"
        fi
    fi
}

# Restore configuration
restore_config() {
    local extract_dir="$1"
    local config_backup_dir="${extract_dir}/config"
    
    log_info "Restoring configuration..."
    
    if [ ! -d "$config_backup_dir" ]; then
        log_warning "No configuration backup found in archive"
        return
    fi
    
    # Ensure config directory exists
    mkdir -p "$CONFIG_DIR"
    
    # Copy configuration files
    cp -r "$config_backup_dir"/* "$CONFIG_DIR/" 2>/dev/null || true
    
    # Set permissions
    chmod 750 "$CONFIG_DIR"
    [ -f "$CONFIG_DIR/.env" ] && chmod 600 "$CONFIG_DIR/.env"
    chown -R tradingbot:tradingbot "$CONFIG_DIR" 2>/dev/null || true
    
    log_success "Configuration restored"
}

# Restore logs
restore_logs() {
    local extract_dir="$1"
    local log_backup_dir="${extract_dir}/logs"
    
    log_info "Restoring logs..."
    
    if [ ! -d "$log_backup_dir" ]; then
        log_warning "No log backup found in archive"
        return
    fi
    
    # Ensure log directory exists
    mkdir -p "$LOG_DIR"
    
    # Copy log files
    cp -r "$log_backup_dir"/* "$LOG_DIR/" 2>/dev/null || true
    
    # Set permissions
    chmod 750 "$LOG_DIR"
    chown -R tradingbot:tradingbot "$LOG_DIR" 2>/dev/null || true
    
    log_success "Logs restored"
}

# Restore application data
restore_data() {
    local extract_dir="$1"
    local data_backup_dir="${extract_dir}/data"
    
    log_info "Restoring application data..."
    
    if [ ! -d "$data_backup_dir" ]; then
        log_warning "No application data backup found in archive"
        return
    fi
    
    # Ensure data directory exists
    mkdir -p "$DATA_DIR"
    
    # Copy data files
    cp -r "$data_backup_dir"/* "$DATA_DIR/" 2>/dev/null || true
    
    # Set permissions
    chmod 755 "$DATA_DIR"
    chown -R tradingbot:tradingbot "$DATA_DIR" 2>/dev/null || true
    
    log_success "Application data restored"
}

# Main restore function
main() {
    log_info "Starting restore process for ${APP_NAME}..."
    echo "========================================"
    
    check_root
    
    # Select backup
    local backup_file=$(select_backup "$@")
    
    # Show backup info
    show_backup_info "$backup_file"
    
    # Verify backup
    verify_backup "$backup_file"
    
    # Confirm restore
    log_warning "This will restore the selected backup and overwrite current data!"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Restore cancelled by user"
        exit 0
    fi
    
    # Backup current state
    local pre_restore_backup=$(backup_current_state)
    
    # Stop service
    stop_service
    
    # Extract backup
    local extract_dir=$(mktemp -d -t restore-XXXXXXXXXX)
    extract_backup "$backup_file" "$extract_dir"
    
    # Restore components
    restore_database "$extract_dir"
    restore_config "$extract_dir"
    restore_logs "$extract_dir"
    restore_data "$extract_dir"
    
    # Clean up temporary directory
    rm -rf "$extract_dir"
    log_info "Temporary files cleaned up"
    
    # Start service
    start_service
    
    echo "========================================"
    log_success "Restore completed successfully!"
    echo ""
    log_info "Restored from: $(basename "$backup_file")"
    log_info "Pre-restore backup: $(basename "$pre_restore_backup")"
    echo ""
    log_warning "Please verify that the application is working correctly"
    log_info "Check status with: systemctl status ${APP_NAME}"
    log_info "View logs with: journalctl -u ${APP_NAME} -f"
    echo "========================================"
}

# Run main function
main "$@"
