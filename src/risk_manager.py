import asyncio
import logging
from utils import WALLogger

logger = logging.getLogger('bot.risk')

class RiskManager:
    def __init__(self, cfg, redis_url=None):
        self.cfg = cfg
        self.redis_url = redis_url
        self.wal = WALLogger()
        self.running = False

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    def compute_position_size(self, account_balance, edge, winrate=0.5):
        # Kelly Criterion (fraction): f = (bp - q)/b ; simplify with edge
        # edge: expected return per trade; assume b=1 (win multiplier)
        b = 1
        p = winrate
        q = 1-p
        try:
            f = (b*p - q)/b
            f = max(0.0, min(f, 0.2))
            size = account_balance * f
        except Exception:
            size = account_balance * 0.01
        return size

    def record_trade(self, trade):
        self.wal.write(str(trade))
