#!/bin/bash
# ==============================================================================
# Trading Bot - Automated Deployment Script
# ==============================================================================
# This script automates the deployment process for the trading bot application
# including system setup, dependency installation, configuration, and service
# management.
# ==============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
APP_NAME="trading-bot"
APP_USER="${APP_USER:-tradingbot}"
APP_DIR="${APP_DIR:-/opt/trading-bot}"
DATA_DIR="${DATA_DIR:-/var/lib/trading-bot}"
LOG_DIR="${LOG_DIR:-/var/log/trading-bot}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/trading-bot}"
CONFIG_DIR="${CONFIG_DIR:-/etc/trading-bot}"
VENV_DIR="${APP_DIR}/venv"
PYTHON_VERSION="${PYTHON_VERSION:-python3}"

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

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check for required commands
    local required_commands=("git" "curl" "wget" "$PYTHON_VERSION" "pip3")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "Required command not found: $cmd"
        fi
    done
    
    # Check Python version
    local python_version=$($PYTHON_VERSION --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    local required_version="3.8"
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        error_exit "Python $required_version or higher is required. Found: $python_version"
    fi
    
    log_success "System requirements check passed"
}

# Create system user
create_user() {
    log_info "Creating application user: $APP_USER"
    
    if id "$APP_USER" &>/dev/null; then
        log_warning "User $APP_USER already exists"
    else
        useradd --system --no-create-home --shell /bin/false "$APP_USER"
        log_success "User $APP_USER created"
    fi
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    local directories=("$APP_DIR" "$DATA_DIR" "$LOG_DIR" "$BACKUP_DIR" "$CONFIG_DIR")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        else
            log_warning "Directory already exists: $dir"
        fi
    done
    
    # Set ownership and permissions
    chown -R "$APP_USER:$APP_USER" "$DATA_DIR" "$LOG_DIR" "$BACKUP_DIR"
    chmod 755 "$APP_DIR" "$DATA_DIR"
    chmod 750 "$LOG_DIR" "$BACKUP_DIR" "$CONFIG_DIR"
    
    log_success "Directory structure created"
}

# Install system dependencies
install_dependencies() {
    log_info "Installing system dependencies..."
    
    # Update package list
    apt-get update -qq
    
    # Install required packages
    apt-get install -y -qq \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        git \
        curl \
        wget \
        nginx \
        supervisor \
        logrotate \
        ufw \
        fail2ban \
        unattended-upgrades \
        sqlite3 \
        redis-server \
        || error_exit "Failed to install system dependencies"
    
    log_success "System dependencies installed"
}

# Setup Python virtual environment
setup_virtualenv() {
    log_info "Setting up Python virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        log_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            log_info "Skipping virtual environment creation"
            return
        fi
    fi
    
    $PYTHON_VERSION -m venv "$VENV_DIR"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Python virtual environment created"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    if [ ! -f "$APP_DIR/requirements.txt" ]; then
        log_warning "requirements.txt not found at $APP_DIR/requirements.txt"
        log_info "Creating basic requirements.txt..."
        cat > "$APP_DIR/requirements.txt" << EOF
python-binance>=1.0.19
requests>=2.31.0
python-dotenv>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
redis>=5.0.0
SQLAlchemy>=2.0.0
psycopg2-binary>=2.9.0
fastapi>=0.104.0
uvicorn>=0.24.0
python-telegram-bot>=20.0
prometheus-client>=0.19.0
aiohttp>=3.9.0
pydantic>=2.5.0
cryptography>=41.0.0
EOF
    fi
    
    source "$VENV_DIR/bin/activate"
    pip install -r "$APP_DIR/requirements.txt" || error_exit "Failed to install Python dependencies"
    
    log_success "Python dependencies installed"
}

# Copy application files
deploy_application() {
    log_info "Deploying application files..."
    
    local current_dir=$(pwd)
    
    # Copy application files
    if [ -d "$current_dir/src" ]; then
        cp -r "$current_dir/src" "$APP_DIR/"
        log_success "Application source files copied"
    else
        log_warning "No src directory found in current directory"
    fi
    
    # Copy scripts
    if [ -d "$current_dir/scripts" ]; then
        cp -r "$current_dir/scripts" "$APP_DIR/"
        chmod +x "$APP_DIR/scripts"/*.py 2>/dev/null || true
        log_success "Scripts copied"
    fi
    
    # Set ownership
    chown -R "$APP_USER:$APP_USER" "$APP_DIR"
    
    log_success "Application deployed"
}

# Setup environment configuration
setup_environment() {
    log_info "Setting up environment configuration..."
    
    local env_file="$CONFIG_DIR/.env"
    
    if [ -f "$env_file" ]; then
        log_warning "Environment file already exists at $env_file"
    else
        if [ -f ".env.template" ]; then
            cp .env.template "$env_file"
            chmod 600 "$env_file"
            chown "$APP_USER:$APP_USER" "$env_file"
            log_success "Environment template copied to $env_file"
            log_warning "Please edit $env_file with your configuration"
        else
            log_warning "No .env.template found"
        fi
    fi
}

# Install systemd service
install_service() {
    log_info "Installing systemd service..."
    
    local service_file="/etc/systemd/system/${APP_NAME}.service"
    
    if [ -f "deployment/systemd/trading-bot.service" ]; then
        cp "deployment/systemd/trading-bot.service" "$service_file"
        
        # Replace placeholders in service file
        sed -i "s|{{APP_DIR}}|${APP_DIR}|g" "$service_file"
        sed -i "s|{{APP_USER}}|${APP_USER}|g" "$service_file"
        sed -i "s|{{CONFIG_DIR}}|${CONFIG_DIR}|g" "$service_file"
        sed -i "s|{{DATA_DIR}}|${DATA_DIR}|g" "$service_file"
        sed -i "s|{{LOG_DIR}}|${LOG_DIR}|g" "$service_file"
        
        # Reload systemd
        systemctl daemon-reload
        systemctl enable "${APP_NAME}.service"
        
        log_success "Systemd service installed and enabled"
    else
        log_warning "Service file not found: deployment/systemd/trading-bot.service"
    fi
}

# Setup nginx reverse proxy
setup_nginx() {
    log_info "Setting up Nginx reverse proxy..."
    
    if [ -f "deployment/nginx/trading-bot.conf" ]; then
        cp "deployment/nginx/trading-bot.conf" "/etc/nginx/sites-available/${APP_NAME}"
        
        # Create symbolic link
        if [ ! -L "/etc/nginx/sites-enabled/${APP_NAME}" ]; then
            ln -s "/etc/nginx/sites-available/${APP_NAME}" "/etc/nginx/sites-enabled/${APP_NAME}"
        fi
        
        # Test nginx configuration
        nginx -t && systemctl reload nginx
        
        log_success "Nginx configuration installed"
    else
        log_warning "Nginx config not found: deployment/nginx/trading-bot.conf"
    fi
}

# Setup log rotation
setup_logrotate() {
    log_info "Setting up log rotation..."
    
    if [ -f "deployment/logrotate/trading-bot" ]; then
        cp "deployment/logrotate/trading-bot" "/etc/logrotate.d/${APP_NAME}"
        chmod 644 "/etc/logrotate.d/${APP_NAME}"
        
        log_success "Log rotation configured"
    else
        log_warning "Logrotate config not found: deployment/logrotate/trading-bot"
    fi
}

# Setup firewall
setup_firewall() {
    log_info "Setting up firewall..."
    
    if [ -f "deployment/scripts/setup_firewall.sh" ]; then
        bash "deployment/scripts/setup_firewall.sh"
        log_success "Firewall configured"
    else
        log_warning "Firewall script not found"
    fi
}

# Setup security hardening
setup_security() {
    log_info "Applying security hardening..."
    
    if [ -f "deployment/scripts/security_hardening.sh" ]; then
        bash "deployment/scripts/security_hardening.sh"
        log_success "Security hardening applied"
    else
        log_warning "Security hardening script not found"
    fi
}

# Setup automated backups
setup_backups() {
    log_info "Setting up automated backups..."
    
    if [ -f "deployment/scripts/setup_cron.sh" ]; then
        bash "deployment/scripts/setup_cron.sh"
        log_success "Automated backups configured"
    else
        log_warning "Backup setup script not found"
    fi
}

# Run health check
run_health_check() {
    log_info "Running health check..."
    
    if [ -f "$APP_DIR/scripts/health_check.py" ]; then
        source "$VENV_DIR/bin/activate"
        $PYTHON_VERSION "$APP_DIR/scripts/health_check.py"
        log_success "Health check completed"
    else
        log_warning "Health check script not found"
    fi
}

# Main deployment function
main() {
    log_info "Starting Trading Bot deployment..."
    echo "========================================"
    
    check_root
    check_requirements
    create_user
    create_directories
    install_dependencies
    setup_virtualenv
    install_python_deps
    deploy_application
    setup_environment
    install_service
    setup_nginx
    setup_logrotate
    setup_firewall
    setup_security
    setup_backups
    
    echo "========================================"
    log_success "Deployment completed successfully!"
    echo ""
    log_info "Next steps:"
    echo "  1. Edit configuration: sudo nano $CONFIG_DIR/.env"
    echo "  2. Start the service: sudo systemctl start $APP_NAME"
    echo "  3. Check status: sudo systemctl status $APP_NAME"
    echo "  4. View logs: sudo journalctl -u $APP_NAME -f"
    echo ""
    log_warning "Important: Make sure to configure your API keys in $CONFIG_DIR/.env"
}

# Run main function
main "$@"
