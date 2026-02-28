#!/bin/bash
# Quick download and install script for the trading bot
# Usage: curl -sL https://raw.githubusercontent.com/donganhhuynh405-a11y/Life_Is_A_Joke/main/scripts/quick_install.sh | bash

set -e

echo "======================================================"
echo "  Trading Bot - Quick Installation Script"
echo "======================================================"
echo ""

# Configuration
REPO_URL="https://github.com/donganhhuynh405-a11y/Life_Is_A_Joke.git"
BRANCH="main"
INSTALL_DIR="$HOME/life_is_a_joke"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Error: Git is not installed!"
    echo "Please install git first:"
    echo "  Ubuntu/Debian: sudo apt-get install git"
    echo "  CentOS/RHEL:   sudo yum install git"
    exit 1
fi

# Clone or update repository
if [ -d "$INSTALL_DIR" ]; then
    echo "📁 Directory exists, updating..."
    cd "$INSTALL_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "📥 Cloning repository..."
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📍 Installed to: $INSTALL_DIR"
echo "📝 Current version:"
git log -1 --oneline
echo ""
echo "📚 Next steps:"
echo "  1. Configure settings:"
echo "     cd $INSTALL_DIR"
echo "     cp .env.template .env"
echo "     nano .env"
echo ""
echo "  2. Install dependencies:"
echo "     pip3 install -r requirements.txt"
echo ""
echo "  3. Run the bot:"
echo "     python3 -m src.main"
echo ""
echo "  4. Or use deployment script for server setup:"
echo "     sudo ./deployment/deploy.sh"
echo ""
