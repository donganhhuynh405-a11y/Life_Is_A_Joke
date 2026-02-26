"""
Signal Scorer - Scores trading signals based on historical patterns
"""

from typing import Dict, Optional
import sqlite3


class SignalScorer:
    """Scores trading signals based on historical success patterns"""
    
    def __init__(self, db_path: str = '/var/lib/trading-bot/trading_bot.db'):
        self.db_path = db_path
    
    def get_symbol_success_rate(self, symbol: str, days: int = 30) -> float:
        """Get historical success rate for a symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as profitable
            FROM positions
            WHERE symbol = ?
            AND status = 'closed'
            AND pnl IS NOT NULL
            AND DATE(closed_at) >= DATE('now', '-' || ? || ' days', 'localtime')
        ''', (symbol, days))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result or result[0] == 0:
            return 0.5
        
        total, profitable = result
        return profitable / total
    
    def get_side_success_rate(self, symbol: str, side: str, days: int = 30) -> float:
        """Get success rate for specific side (BUY/SELL) on a symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as profitable
            FROM positions
            WHERE symbol = ?
            AND side = ?
            AND status = 'closed'
            AND pnl IS NOT NULL
            AND DATE(closed_at) >= DATE('now', '-' || ? || ' days', 'localtime')
        ''', (symbol, side, days))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result or result[0] == 0:
            return 0.5
        
        total, profitable = result
        return profitable / total
    
    def get_symbol_stats(self, symbol: str, days: int = 30) -> Dict:
        """
        Get comprehensive stats for a symbol
        
        Args:
            symbol: Trading pair
            days: Days of history to analyze
            
        Returns:
            Dict with detailed stats including win_rate, total_trades, avg_pnl
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as profitable,
                       AVG(pnl) as avg_pnl
                FROM positions
                WHERE symbol = ?
                AND status = 'closed'
                AND pnl IS NOT NULL
                AND DATE(closed_at) >= DATE('now', '-' || ? || ' days', 'localtime')
            ''', (symbol, days))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result or result[0] == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 50.0,
                    'avg_pnl': 0.0
                }
            
            total, profitable, avg_pnl = result
            win_rate = (profitable / total) * 100 if total > 0 else 50.0
            
            return {
                'total_trades': total,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl if avg_pnl else 0.0
            }
        except Exception:
            return {
                'total_trades': 0,
                'win_rate': 50.0,
                'avg_pnl': 0.0
            }
    
    def get_side_stats(self, symbol: str, side: str, days: int = 30) -> Dict:
        """
        Get comprehensive stats for a specific side on a symbol
        
        Args:
            symbol: Trading pair
            side: BUY or SELL
            days: Days of history to analyze
            
        Returns:
            Dict with detailed stats including win_rate, trades
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as profitable
                FROM positions
                WHERE symbol = ?
                AND side = ?
                AND status = 'closed'
                AND pnl IS NOT NULL
                AND DATE(closed_at) >= DATE('now', '-' || ? || ' days', 'localtime')
            ''', (symbol, side, days))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result or result[0] == 0:
                return {
                    'trades': 0,
                    'win_rate': 50.0
                }
            
            total, profitable = result
            win_rate = (profitable / total) * 100 if total > 0 else 50.0
            
            return {
                'trades': total,
                'win_rate': win_rate
            }
        except Exception:
            return {
                'trades': 0,
                'win_rate': 50.0
            }
    
    def score_signal(self, symbol: str, side: str, confidence: float = 0.5) -> Dict:
        """
        Score a trading signal based on historical patterns
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            confidence: Base confidence from technical indicators (0-1)
        
        Returns:
            Dict with scoring details
        """
        # Get historical success rates
        symbol_rate_30d = self.get_symbol_success_rate(symbol, days=30)
        symbol_rate_7d = self.get_symbol_success_rate(symbol, days=7)
        side_rate = self.get_side_success_rate(symbol, side, days=30)
        
        # Weight the scores
        # Recent performance (7 days) weighted higher
        historical_score = (symbol_rate_7d * 0.4 + symbol_rate_30d * 0.3 + side_rate * 0.3)
        
        # Combine with technical confidence
        # Historical patterns: 40%, Technical indicators: 60%
        final_score = (historical_score * 0.4) + (confidence * 0.6)
        
        # Convert to 0-100 scale
        final_score_pct = final_score * 100
        
        return {
            'score': final_score_pct,
            'symbol_success_rate_30d': symbol_rate_30d * 100,
            'symbol_success_rate_7d': symbol_rate_7d * 100,
            'side_success_rate': side_rate * 100,
            'technical_confidence': confidence * 100,
            'recommendation': self._get_recommendation(final_score_pct)
        }
    
    def _get_recommendation(self, score: float) -> str:
        """Get trading recommendation based on score"""
        if score >= 75:
            return "STRONG - High probability trade"
        elif score >= 60:
            return "MODERATE - Good probability trade"
        elif score >= 50:
            return "WEAK - Neutral probability"
        else:
            return "AVOID - Low probability trade"
    
    def get_best_performing_pairs(self, limit: int = 5, days: int = 30) -> list:
        """Get top performing trading pairs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol,
                   COUNT(*) as trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as total_pnl,
                   AVG(pnl) as avg_pnl
            FROM positions
            WHERE status = 'closed'
            AND pnl IS NOT NULL
            AND DATE(closed_at) >= DATE('now', '-' || ? || ' days', 'localtime')
            GROUP BY symbol
            HAVING COUNT(*) >= 3
            ORDER BY total_pnl DESC
            LIMIT ?
        ''', (days, limit))
        
        results = []
        for row in cursor.fetchall():
            symbol, trades, wins, total_pnl, avg_pnl = row
            win_rate = (wins / trades * 100) if trades > 0 else 0
            results.append({
                'symbol': symbol,
                'trades': trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl
            })
        
        conn.close()
        return results
    
    def get_worst_performing_pairs(self, limit: int = 5, days: int = 30) -> list:
        """Get worst performing trading pairs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol,
                   COUNT(*) as trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as total_pnl,
                   AVG(pnl) as avg_pnl
            FROM positions
            WHERE status = 'closed'
            AND pnl IS NOT NULL
            AND DATE(closed_at) >= DATE('now', '-' || ? || ' days', 'localtime')
            GROUP BY symbol
            HAVING COUNT(*) >= 3
            ORDER BY total_pnl ASC
            LIMIT ?
        ''', (days, limit))
        
        results = []
        for row in cursor.fetchall():
            symbol, trades, wins, total_pnl, avg_pnl = row
            win_rate = (wins / trades * 100) if trades > 0 else 0
            results.append({
                'symbol': symbol,
                'trades': trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl
            })
        
        conn.close()
        return results
