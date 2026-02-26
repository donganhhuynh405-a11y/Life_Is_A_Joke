#!/bin/bash
# ==============================================================================
# Trading Bot - Security Hardening Script
# ==============================================================================
# Applies security best practices and hardening measures to the system
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

# Update system packages
update_system() {
    log_info "Updating system packages..."
    
    apt-get update -qq
    apt-get upgrade -y -qq
    apt-get autoremove -y -qq
    apt-get autoclean -qq
    
    log_success "System packages updated"
}

# Configure automatic security updates
setup_auto_updates() {
    log_info "Configuring automatic security updates..."
    
    # Install unattended-upgrades if not present
    if ! dpkg -l | grep -q unattended-upgrades; then
        apt-get install -y -qq unattended-upgrades
    fi
    
    # Configure automatic updates
    cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Kernel-Packages "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
Unattended-Upgrade::Automatic-Reboot-Time "03:00";
EOF
    
    # Enable automatic updates
    cat > /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Download-Upgradeable-Packages "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Unattended-Upgrade "1";
EOF
    
    log_success "Automatic security updates configured"
}

# Harden SSH configuration
harden_ssh() {
    log_info "Hardening SSH configuration..."
    
    local sshd_config="/etc/ssh/sshd_config"
    local backup_file="${sshd_config}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Backup current configuration
    cp "$sshd_config" "$backup_file"
    log_info "SSH config backed up to: $backup_file"
    
    # Apply hardening settings
    declare -A settings=(
        ["PermitRootLogin"]="no"
        ["PasswordAuthentication"]="no"
        ["PubkeyAuthentication"]="yes"
        ["PermitEmptyPasswords"]="no"
        ["ChallengeResponseAuthentication"]="no"
        ["X11Forwarding"]="no"
        ["MaxAuthTries"]="3"
        ["ClientAliveInterval"]="300"
        ["ClientAliveCountMax"]="2"
        ["UsePAM"]="yes"
        ["Protocol"]="2"
    )
    
    for key in "${!settings[@]}"; do
        value="${settings[$key]}"
        
        # Remove existing line
        sed -i "/^#*${key}/d" "$sshd_config"
        
        # Add new line
        echo "${key} ${value}" >> "$sshd_config"
    done
    
    # Test SSH configuration
    if sshd -t; then
        log_success "SSH configuration hardened"
        log_warning "Restart SSH service: systemctl restart sshd"
    else
        log_error "SSH configuration test failed"
        cp "$backup_file" "$sshd_config"
        log_info "Configuration restored from backup"
        return 1
    fi
}

# Configure fail2ban
setup_fail2ban() {
    log_info "Configuring fail2ban..."
    
    # Install if not present
    if ! dpkg -l | grep -q fail2ban; then
        apt-get install -y -qq fail2ban
    fi
    
    # Create local jail configuration
    cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
destemail = root@localhost
sendername = Fail2Ban
action = %(action_mwl)s

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
port = http,https
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 5

[nginx-limit-req]
enabled = true
port = http,https
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 10

[nginx-noscript]
enabled = true
port = http,https
filter = nginx-noscript
logpath = /var/log/nginx/access.log
maxretry = 6
EOF
    
    # Start and enable fail2ban
    systemctl enable fail2ban
    systemctl restart fail2ban
    
    log_success "fail2ban configured and started"
}

# Set file permissions
secure_file_permissions() {
    log_info "Securing file permissions..."
    
    # Secure sensitive directories
    local config_dir="${CONFIG_DIR:-/etc/trading-bot}"
    local data_dir="${DATA_DIR:-/var/lib/trading-bot}"
    local log_dir="${LOG_DIR:-/var/log/trading-bot}"
    
    if [ -d "$config_dir" ]; then
        chmod 750 "$config_dir"
        [ -f "$config_dir/.env" ] && chmod 600 "$config_dir/.env"
        log_success "Configuration directory secured"
    fi
    
    if [ -d "$data_dir" ]; then
        chmod 755 "$data_dir"
        find "$data_dir" -type f -name "*.db" -exec chmod 600 {} \;
        log_success "Data directory secured"
    fi
    
    if [ -d "$log_dir" ]; then
        chmod 750 "$log_dir"
        log_success "Log directory secured"
    fi
    
    # Secure important system files
    chmod 644 /etc/passwd
    chmod 640 /etc/shadow
    chmod 644 /etc/group
    chmod 640 /etc/gshadow
    
    log_success "File permissions secured"
}

# Disable unnecessary services
disable_services() {
    log_info "Checking for unnecessary services..."
    
    local services_to_disable=(
        "avahi-daemon"
        "cups"
        "bluetooth"
    )
    
    for service in "${services_to_disable[@]}"; do
        if systemctl is-enabled "$service" &>/dev/null; then
            systemctl disable "$service" 2>/dev/null || true
            systemctl stop "$service" 2>/dev/null || true
            log_info "Disabled service: $service"
        fi
    done
    
    log_success "Unnecessary services checked"
}

# Configure system limits
configure_limits() {
    log_info "Configuring system limits..."
    
    # Set limits for the trading bot user
    cat >> /etc/security/limits.conf << 'EOF'

# Trading Bot limits
tradingbot soft nofile 65536
tradingbot hard nofile 65536
tradingbot soft nproc 4096
tradingbot hard nproc 4096
EOF
    
    log_success "System limits configured"
}

# Harden kernel parameters
harden_kernel() {
    log_info "Hardening kernel parameters..."
    
    local sysctl_conf="/etc/sysctl.d/99-trading-bot-hardening.conf"
    
    cat > "$sysctl_conf" << 'EOF'
# Trading Bot - Security Hardening

# Network security
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# IP forwarding (disable if not a router)
net.ipv4.ip_forward = 0
net.ipv6.conf.all.forwarding = 0

# ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Ignore ICMP ping requests
# net.ipv4.icmp_echo_ignore_all = 1

# Log suspicious packets
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# SYN flood protection
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# Disable IPv6 if not needed
# net.ipv6.conf.all.disable_ipv6 = 1
# net.ipv6.conf.default.disable_ipv6 = 1

# Increase system file descriptor limit
fs.file-max = 65535

# Protect against buffer overflow attacks
# kernel.exec-shield = 1  # Deprecated on modern kernels
kernel.randomize_va_space = 2

# Restrict kernel pointers
kernel.kptr_restrict = 2

# Restrict kernel logs
kernel.dmesg_restrict = 1
EOF
    
    # Apply settings
    sysctl -p "$sysctl_conf" > /dev/null
    
    log_success "Kernel parameters hardened"
}

# Setup audit logging
setup_auditd() {
    log_info "Setting up audit logging..."
    
    # Install auditd if not present
    if ! dpkg -l | grep -q auditd; then
        apt-get install -y -qq auditd audispd-plugins
    fi
    
    # Basic audit rules
    cat >> /etc/audit/rules.d/trading-bot.rules << 'EOF'
# Trading Bot - Audit Rules

# Monitor configuration changes
-w /etc/trading-bot/ -p wa -k trading_bot_config

# Monitor binary changes
-w /opt/trading-bot/ -p wa -k trading_bot_binary

# Monitor authentication
-w /var/log/auth.log -p wa -k auth_log

# Monitor system calls
-a always,exit -F arch=b64 -S adjtimex -S settimeofday -k time-change
-a always,exit -F arch=b32 -S adjtimex -S settimeofday -S stime -k time-change
EOF
    
    # Restart auditd
    service auditd restart
    
    log_success "Audit logging configured"
}

# Configure timezone and NTP
configure_time() {
    log_info "Configuring system time..."
    
    # Set timezone to UTC (recommended for servers)
    timedatectl set-timezone UTC
    
    # Enable NTP
    timedatectl set-ntp true
    
    log_success "System time configured (UTC, NTP enabled)"
}

# Create security report
create_security_report() {
    local report_file="/root/security_hardening_report_$(date +%Y%m%d_%H%M%S).txt"
    
    log_info "Creating security report..."
    
    cat > "$report_file" << EOF
Trading Bot - Security Hardening Report
========================================
Date: $(date)
Hostname: $(hostname)

System Information:
- OS: $(lsb_release -d | cut -f2-)
- Kernel: $(uname -r)
- Uptime: $(uptime -p)

Security Measures Applied:
- System packages updated
- Automatic security updates enabled
- SSH hardened
- fail2ban configured
- File permissions secured
- Unnecessary services disabled
- System limits configured
- Kernel parameters hardened
- Audit logging enabled
- Time synchronization configured

Active Services:
$(systemctl list-units --type=service --state=running | grep -E "ssh|fail2ban|auditd|ntp|systemd-timesyncd")

Firewall Status:
$(ufw status verbose 2>/dev/null || echo "UFW not configured")

fail2ban Status:
$(fail2ban-client status 2>/dev/null || echo "fail2ban not running")

Recommendations:
1. Review SSH configuration: /etc/ssh/sshd_config
2. Check fail2ban jails: fail2ban-client status
3. Monitor audit logs: ausearch -k trading_bot_config
4. Review firewall rules: ufw status verbose
5. Keep system updated: apt update && apt upgrade
6. Rotate logs regularly
7. Monitor system resources
8. Backup configuration regularly
9. Use strong passwords and key-based authentication
10. Limit user access and privileges

EOF
    
    log_success "Security report created: $report_file"
    cat "$report_file"
}

# Main function
main() {
    log_info "Starting security hardening process..."
    echo "========================================"
    
    check_root
    
    # Ask for confirmation
    log_warning "This script will apply security hardening measures to your system."
    log_warning "Some changes may affect system functionality."
    read -p "Do you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Security hardening cancelled"
        exit 0
    fi
    
    # Apply hardening measures
    update_system
    setup_auto_updates
    harden_ssh
    setup_fail2ban
    secure_file_permissions
    disable_services
    configure_limits
    harden_kernel
    setup_auditd
    configure_time
    
    echo "========================================"
    log_success "Security hardening completed!"
    echo ""
    
    # Create report
    create_security_report
    
    echo ""
    log_warning "Important next steps:"
    echo "  1. Review the security report"
    echo "  2. Restart SSH service: systemctl restart sshd"
    echo "  3. Test SSH access before closing this session"
    echo "  4. Consider rebooting the system: reboot"
    echo "========================================"
}

# Run main function
main "$@"
