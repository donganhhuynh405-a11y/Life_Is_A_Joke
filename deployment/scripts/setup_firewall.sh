#!/bin/bash
# ==============================================================================
# Trading Bot - UFW Firewall Configuration Script
# ==============================================================================
# Configures UFW firewall with secure defaults for the trading bot
# ==============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check if UFW is installed
check_ufw() {
    if ! command -v ufw &> /dev/null; then
        log_warning "UFW not installed. Installing..."
        apt-get update -qq
        apt-get install -y -qq ufw
        log_success "UFW installed"
    else
        log_info "UFW is already installed"
    fi
}

# Reset UFW to defaults
reset_ufw() {
    log_info "Resetting UFW to defaults..."
    
    # Disable UFW first
    ufw --force disable
    
    # Reset rules
    echo "y" | ufw reset
    
    log_success "UFW reset to defaults"
}

# Configure default policies
set_default_policies() {
    log_info "Setting default policies..."
    
    # Default deny incoming
    ufw default deny incoming
    
    # Default allow outgoing
    ufw default allow outgoing
    
    log_success "Default policies set (deny incoming, allow outgoing)"
}

# Allow SSH access
allow_ssh() {
    log_info "Configuring SSH access..."
    
    # Get SSH port from sshd_config or use default
    SSH_PORT=$(grep "^Port" /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}')
    SSH_PORT=${SSH_PORT:-22}
    
    # Allow SSH
    ufw allow $SSH_PORT/tcp comment 'SSH access'
    
    log_success "SSH access allowed on port $SSH_PORT"
    log_warning "Make sure you can connect via SSH before enabling the firewall!"
}

# Allow HTTP/HTTPS
allow_web() {
    log_info "Configuring web access..."
    
    # Check if web interface is enabled
    local web_enabled=${WEB_ENABLED:-false}
    
    if [ "$web_enabled" = "true" ]; then
        # Allow HTTP
        ufw allow 80/tcp comment 'HTTP'
        log_success "HTTP (port 80) allowed"
        
        # Allow HTTPS
        ufw allow 443/tcp comment 'HTTPS'
        log_success "HTTPS (port 443) allowed"
    else
        log_info "Web interface not enabled, skipping HTTP/HTTPS"
    fi
}

# Rate limiting for SSH
setup_rate_limiting() {
    log_info "Setting up rate limiting for SSH..."
    
    # Get SSH port
    SSH_PORT=$(grep "^Port" /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}')
    SSH_PORT=${SSH_PORT:-22}
    
    # Remove existing SSH rule
    ufw delete allow $SSH_PORT/tcp 2>/dev/null || true
    
    # Add rate-limited SSH rule
    ufw limit $SSH_PORT/tcp comment 'SSH with rate limiting'
    
    log_success "Rate limiting configured for SSH"
}

# Allow specific IP addresses (whitelist)
allow_whitelist() {
    local whitelist="${IP_WHITELIST:-}"
    
    if [ -n "$whitelist" ]; then
        log_info "Configuring IP whitelist..."
        
        IFS=',' read -ra IPS <<< "$whitelist"
        for ip in "${IPS[@]}"; do
            ip=$(echo "$ip" | xargs)  # Trim whitespace
            if [ -n "$ip" ]; then
                ufw allow from "$ip" comment "Whitelisted IP"
                log_success "Whitelisted IP: $ip"
            fi
        done
    else
        log_info "No IP whitelist configured"
    fi
}

# Block specific countries (optional - requires geoip)
setup_geoip_blocking() {
    log_info "Checking GeoIP blocking capability..."
    
    # This is optional and requires additional setup
    # For now, just log that it's available
    log_info "GeoIP blocking can be configured manually if needed"
    log_info "Refer to: https://github.com/xtenduke/ufw-geoip"
}

# Configure application-specific rules
configure_app_rules() {
    log_info "Configuring application-specific rules..."
    
    # Allow localhost
    ufw allow from 127.0.0.1 comment 'Localhost'
    
    # If using PostgreSQL
    local db_type="${DB_TYPE:-sqlite}"
    if [ "$db_type" = "postgresql" ]; then
        local db_host="${DB_HOST:-localhost}"
        
        if [ "$db_host" != "localhost" ] && [ "$db_host" != "127.0.0.1" ]; then
            log_info "PostgreSQL on remote host detected"
            # Allow PostgreSQL (only from specific IP if needed)
            # ufw allow from $db_host to any port 5432 proto tcp
        fi
    fi
    
    # If using Redis on remote host
    local redis_enabled="${REDIS_ENABLED:-false}"
    if [ "$redis_enabled" = "true" ]; then
        local redis_host="${REDIS_HOST:-localhost}"
        
        if [ "$redis_host" != "localhost" ] && [ "$redis_host" != "127.0.0.1" ]; then
            log_info "Redis on remote host detected"
            # Allow Redis (only from specific IP if needed)
            # ufw allow from $redis_host to any port 6379 proto tcp
        fi
    fi
    
    # Allow Prometheus metrics (only from localhost by default)
    local metrics_enabled="${METRICS_ENABLED:-false}"
    if [ "$metrics_enabled" = "true" ]; then
        # Metrics should only be accessible from localhost
        # Nginx will proxy if needed
        log_info "Metrics enabled (accessible via localhost only)"
    fi
    
    log_success "Application rules configured"
}

# Enable logging
enable_logging() {
    log_info "Configuring firewall logging..."
    
    # Set logging to low (can be: off, low, medium, high, full)
    ufw logging low
    
    log_success "Firewall logging enabled (low level)"
}

# Display configuration
show_configuration() {
    log_info "Current UFW configuration:"
    echo ""
    ufw status verbose
    echo ""
}

# Enable UFW
enable_firewall() {
    log_warning "About to enable firewall..."
    log_warning "Make sure you have configured SSH access correctly!"
    
    read -p "Do you want to enable the firewall now? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        # Enable UFW
        echo "y" | ufw enable
        
        log_success "Firewall enabled"
        
        # Show status
        ufw status verbose
    else
        log_info "Firewall not enabled. Run 'sudo ufw enable' when ready."
    fi
}

# Create backup of current rules
backup_rules() {
    local backup_dir="/var/backups/ufw"
    local backup_file="${backup_dir}/ufw-rules-$(date +%Y%m%d_%H%M%S).txt"
    
    mkdir -p "$backup_dir"
    
    if ufw status &>/dev/null; then
        ufw status numbered > "$backup_file" 2>/dev/null || true
        log_info "Current rules backed up to: $backup_file"
    fi
}

# Main function
main() {
    log_info "Starting UFW firewall configuration..."
    echo "========================================"
    
    check_root
    check_ufw
    
    # Backup current rules
    backup_rules
    
    # Ask if user wants to reset
    read -p "Do you want to reset UFW to defaults? (yes/no): " reset_confirm
    if [ "$reset_confirm" = "yes" ]; then
        reset_ufw
    fi
    
    # Configure firewall
    set_default_policies
    allow_ssh
    allow_web
    setup_rate_limiting
    allow_whitelist
    configure_app_rules
    enable_logging
    
    # Additional security
    setup_geoip_blocking
    
    echo "========================================"
    log_success "Firewall configuration completed!"
    echo ""
    
    # Show configuration
    show_configuration
    
    # Enable firewall
    enable_firewall
    
    echo "========================================"
    log_info "Firewall setup complete!"
    log_info "View status: sudo ufw status verbose"
    log_info "View logs: sudo tail -f /var/log/ufw.log"
    echo "========================================"
}

# Run main function
main "$@"
