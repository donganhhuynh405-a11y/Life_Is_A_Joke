import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class Trade:
    timestamp: str
    symbol: str
    side: str
    amount: float
    price: float
    profit: float

class MetricsTracker:
    def __init__(self):
        self.trades: List[Trade] = []
        self.daily_pnl = {}
    
    def add_trade(self, trade: Trade):
        self.trades.append(trade)
    
    def get_metrics(self) -> Dict:
        if not self.trades:
            return {"total_trades": 0}
        
        profits = [t.profit for t in self.trades]
        returns = np.array(profits)
        
        return {
            "total_trades": len(self.trades),
            "total_profit": sum(profits),
            "win_rate": (sum(1 for p in profits if p > 0) / len(profits)) * 100,
            "sharpe_ratio": np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            "max_drawdown": self._max_drawdown(profits),
            "profit_factor": self._profit_factor(profits)
        }
    
    def _max_drawdown(self, profits: List[float]) -> float:
        cumsum = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = running_max - cumsum
        return drawdown.max()
    
    def _profit_factor(self, profits: List[float]) -> float:
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        return sum(wins) / abs(sum(losses)) if losses else float('inf')
    
    def to_json(self) -> str:
        return json.dumps(self.get_metrics(), indent=2)
