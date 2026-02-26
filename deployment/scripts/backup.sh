#!/bin/bash
# ==============================================================================
# Trading Bot - Backup Script
# ==============================================================================
# Creates timestamped backups of database, configuration, and logs
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
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

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

# Check if running as root or appropriate user
check_permissions() {
    if [[ $EUID -ne 0 ]] && [[ $(whoami) != "tradingbot" ]]; then
        log_warning "This script should be run as root or tradingbot user"
    fi
}

# Create backup directory if needed
create_backup_dir() {
    if [ ! -d "$BACKUP_DIR" ]; then
        log_info "Creating backup directory: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
    fi
    
    # Set permissions
    chmod 750 "$BACKUP_DIR"
}

# Generate backup filename with timestamp
generate_backup_name() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    echo "${APP_NAME}_backup_${timestamp}.tar.gz"
}

# Create temporary directory for backup staging
create_temp_dir() {
    local temp_dir=$(mktemp -d -t backup-XXXXXXXXXX)
    echo "$temp_dir"
}

# Backup database
backup_database() {
    local temp_dir="$1"
    local db_backup_dir="${temp_dir}/database"
    
    log_info "Backing up database..."
    
    mkdir -p "$db_backup_dir"
    
    # Check database type
    local db_type="${DB_TYPE:-sqlite}"
    
    if [ "$db_type" = "sqlite" ]; then
        local db_path="${DB_PATH:-${DATA_DIR}/trading_bot.db}"
        
        if [ -f "$db_path" ]; then
            # Use sqlite3 to create a backup (safer than copying)
            if command -v sqlite3 &> /dev/null; then
                sqlite3 "$db_path" ".backup '${db_backup_dir}/trading_bot.db'"
                log_success "SQLite database backed up"
            else
                # Fallback to copying
                cp "$db_path" "${db_backup_dir}/trading_bot.db"
                log_success "SQLite database copied"
            fi
        else
            log_warning "Database file not found: $db_path"
        fi
    elif [ "$db_type" = "postgresql" ]; then
        local db_name="${DB_NAME:-trading_bot}"
        local db_user="${DB_USER:-trading_bot_user}"
        local db_host="${DB_HOST:-localhost}"
        
        if command -v pg_dump &> /dev/null; then
            PGPASSWORD="${DB_PASSWORD}" pg_dump -h "$db_host" -U "$db_user" "$db_name" > "${db_backup_dir}/${db_name}.sql"
            log_success "PostgreSQL database backed up"
        else
            log_warning "pg_dump not found - skipping PostgreSQL backup"
        fi
    else
        log_warning "Unknown database type: $db_type"
    fi
}

# Backup configuration files
backup_config() {
    local temp_dir="$1"
    local config_backup_dir="${temp_dir}/config"
    
    log_info "Backing up configuration..."
    
    mkdir -p "$config_backup_dir"
    
    if [ -d "$CONFIG_DIR" ]; then
        cp -r "$CONFIG_DIR"/* "$config_backup_dir/" 2>/dev/null || true
        log_success "Configuration files backed up"
    else
        log_warning "Configuration directory not found: $CONFIG_DIR"
    fi
}

# Backup logs
backup_logs() {
    local temp_dir="$1"
    local log_backup_dir="${temp_dir}/logs"
    
    log_info "Backing up logs..."
    
    mkdir -p "$log_backup_dir"
    
    if [ -d "$LOG_DIR" ]; then
        # Only backup .log files, not compressed archives
        find "$LOG_DIR" -type f -name "*.log" -exec cp {} "$log_backup_dir/" \; 2>/dev/null || true
        log_success "Log files backed up"
    else
        log_warning "Log directory not found: $LOG_DIR"
    fi
}

# Backup application data
backup_data() {
    local temp_dir="$1"
    local data_backup_dir="${temp_dir}/data"
    
    log_info "Backing up application data..."
    
    mkdir -p "$data_backup_dir"
    
    if [ -d "$DATA_DIR" ]; then
        # Copy all data except database (already backed up separately)
        find "$DATA_DIR" -type f ! -name "*.db" ! -name "*.db-*" -exec cp --parents {} "$data_backup_dir/" \; 2>/dev/null || true
        log_success "Application data backed up"
    else
        log_warning "Data directory not found: $DATA_DIR"
    fi
}

# Create backup metadata
create_metadata() {
    local temp_dir="$1"
    local metadata_file="${temp_dir}/backup_metadata.txt"
    
    log_info "Creating backup metadata..."
    
    cat > "$metadata_file" << EOF
Backup Metadata
===============
Application: ${APP_NAME}
Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')
Hostname: $(hostname)
User: $(whoami)

Directories Backed Up:
- Database: ${DB_PATH:-${DATA_DIR}/trading_bot.db}
- Configuration: ${CONFIG_DIR}
- Logs: ${LOG_DIR}
- Data: ${DATA_DIR}

System Information:
- OS: $(uname -s)
- Kernel: $(uname -r)
- Architecture: $(uname -m)

Disk Usage:
$(df -h / | tail -n 1)
EOF
    
    log_success "Metadata created"
}

# Compress backup
compress_backup() {
    local temp_dir="$1"
    local backup_name="$2"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    
    log_info "Compressing backup..."
    
    tar -czf "$backup_path" -C "$(dirname "$temp_dir")" "$(basename "$temp_dir")" 2>/dev/null
    
    if [ -f "$backup_path" ]; then
        local size=$(du -h "$backup_path" | cut -f1)
        log_success "Backup compressed: $backup_path (${size})"
        echo "$backup_path"
    else
        error_exit "Failed to create backup archive"
    fi
}

# Clean up old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups (retention: ${RETENTION_DAYS} days)..."
    
    local deleted_count=0
    
    # Find and delete backups older than retention period
    while IFS= read -r -d '' backup_file; do
        rm -f "$backup_file"
        ((deleted_count++))
        log_info "Deleted old backup: $(basename "$backup_file")"
    done < <(find "$BACKUP_DIR" -name "${APP_NAME}_backup_*.tar.gz" -type f -mtime "+${RETENTION_DAYS}" -print0 2>/dev/null)
    
    if [ $deleted_count -gt 0 ]; then
        log_success "Deleted $deleted_count old backup(s)"
    else
        log_info "No old backups to delete"
    fi
}

# Verify backup integrity
verify_backup() {
    local backup_path="$1"
    
    log_info "Verifying backup integrity..."
    
    if tar -tzf "$backup_path" > /dev/null 2>&1; then
        log_success "Backup integrity verified"
        return 0
    else
        log_error "Backup integrity check failed"
        return 1
    fi
}

# Main backup function
main() {
    log_info "Starting backup process for ${APP_NAME}..."
    echo "========================================"
    
    check_permissions
    create_backup_dir
    
    # Create temporary directory
    local temp_dir=$(create_temp_dir)
    log_info "Using temporary directory: $temp_dir"
    
    # Perform backups
    backup_database "$temp_dir"
    backup_config "$temp_dir"
    backup_logs "$temp_dir"
    backup_data "$temp_dir"
    create_metadata "$temp_dir"
    
    # Compress and save
    local backup_name=$(generate_backup_name)
    local backup_path=$(compress_backup "$temp_dir" "$backup_name")
    
    # Verify backup
    verify_backup "$backup_path"
    
    # Clean up temp directory
    rm -rf "$temp_dir"
    log_info "Temporary files cleaned up"
    
    # Clean up old backups
    cleanup_old_backups
    
    # Display backup information
    echo "========================================"
    log_success "Backup completed successfully!"
    echo ""
    log_info "Backup Details:"
    echo "  Location: $backup_path"
    echo "  Size: $(du -h "$backup_path" | cut -f1)"
    echo "  Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # List all available backups
    local backup_count=$(find "$BACKUP_DIR" -name "${APP_NAME}_backup_*.tar.gz" -type f 2>/dev/null | wc -l)
    log_info "Total backups available: $backup_count"
    
    echo "========================================"
}

# Run main function
main "$@"
