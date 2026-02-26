#!/bin/bash
echo "🚀 Installing Trading Bot..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3 first:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    exit 1
fi

# Install dependencies
echo "📥 Installing dependencies..."
pip3 install -r requirements.txt

# Create directories
echo "📁 Creating necessary directories..."
mkdir -p logs data configs backups

# Copy environment template
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env file with your API keys"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📚 Available features:"
echo "  📊 Backtest: python -m backtester.cli test BTCUSDT 90"
echo "  🌐 Dashboard: uvicorn src.dashboard:app --reload --port 8080"
echo "  🤖 Telegram: python telegram_bot.py"
echo "  🚀 Main bot: python -m src.main"
echo ""

