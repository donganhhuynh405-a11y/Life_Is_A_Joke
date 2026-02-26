"""
AI Commentary Generator for Trading Bot

Generates intelligent commentary and insights for trading notifications based on
historical performance analysis and current market conditions.

Enhanced with:
- Performance caching for efficiency
- More sophisticated pattern recognition
- Context-aware recommendations
- Adaptive thresholds based on historical data
"""

import logging
import os
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from .trade_analyzer import TradeAnalyzer
from .performance_analyzer import PerformanceAnalyzer
from .signal_scorer import SignalScorer

# Import translation manager
try:
    from utils.translations import get_translation_manager
    TRANSLATIONS_AVAILABLE = True
except ImportError:
    TRANSLATIONS_AVAILABLE = False


class AICommentaryGenerator:
    """
    Generates AI-powered commentary for trading notifications
    
    Features:
    - Performance-based caching to reduce database queries
    - Adaptive thresholds based on historical performance
    - Context-aware insights considering multiple timeframes
    - Pattern recognition for trade sequences
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, db_path: str = '/var/lib/trading-bot/trading_bot.db', language: str = None):
        """
        Initialize AI Commentary Generator
        
        Args:
            logger: Optional logger instance
            db_path: Path to the database file
            language: Notification language code (default: from env or 'en')
        """
        self.logger = logger or logging.getLogger(__name__)
        self.trade_analyzer = TradeAnalyzer(db_path=db_path)
        self.perf_analyzer = PerformanceAnalyzer(db_path=db_path)
        self.signal_scorer = SignalScorer(db_path=db_path)
        
        # Initialize translation manager
        self.language = language or os.getenv('NOTIFICATION_LANGUAGE', 'en')
        if TRANSLATIONS_AVAILABLE:
            self.translator = get_translation_manager(self.language)
            self.logger.info(f"AI Commentary language: {self.language}")
        else:
            self.translator = None
            self.logger.warning("Translations not available, using English")
        
        # Performance cache with timestamps
        self._cache = {
            'symbol_stats': {},
            'performance_7d': None,
            'performance_30d': None,
            'advanced_metrics': None,
            'last_update': None
        }
        self._cache_ttl = 300  # 5 minutes cache TTL
    
    def _t(self, key: str, **kwargs) -> str:
        """Get translated string with optional formatting"""
        if self.translator:
            return self.translator.get(key, **kwargs)
        # Fallback to key if no translator
        return key
    
    def _refresh_cache_if_needed(self):
        """Refresh cache if TTL expired"""
        now = datetime.now()
        if (self._cache['last_update'] is None or 
            (now - self._cache['last_update']).total_seconds() > self._cache_ttl):
            
            try:
                self._cache['performance_7d'] = self.trade_analyzer.analyze_performance(days=7)
                self._cache['performance_30d'] = self.trade_analyzer.analyze_performance(days=30)
                self._cache['advanced_metrics'] = self.perf_analyzer.get_performance_summary()
                self._cache['last_update'] = now
                self.logger.debug("Performance cache refreshed")
            except Exception as e:
                self.logger.error(f"Error refreshing cache: {e}")
    
    def _get_adaptive_thresholds(self) -> Dict[str, float]:
        """
        Calculate adaptive thresholds based on historical performance
        
        Returns better calibrated thresholds instead of fixed values
        """
        self._refresh_cache_if_needed()
        
        # Get performance with safe defaults
        perf_30d = self._cache.get('performance_30d') or {}
        avg_win_rate = perf_30d.get('win_rate', 50) if perf_30d else 50
        
        # Adjust confidence thresholds based on actual historical performance
        # If bot historically wins at 70%, then 70% confidence is "moderate"
        base_threshold = max(50, min(70, avg_win_rate))
        
        return {
            'high_confidence': base_threshold + 15,
            'moderate_confidence': base_threshold,
            'good_win_rate': avg_win_rate + 10,
            'poor_win_rate': avg_win_rate - 10
        }
    
    def generate_position_open_commentary(self, symbol: str, side: str, 
                                         confidence: float = None) -> str:
        """
        Generate AI commentary for position opening with adaptive analysis
        
        Args:
            symbol: Trading pair symbol
            side: BUY or SELL
            confidence: Signal confidence score (0-1 or 0-100)
            
        Returns:
            Commentary string with context-aware insights
        """
        try:
            # Normalize confidence to 0-1 range
            if confidence and confidence > 1:
                confidence = confidence / 100
            
            # Get adaptive thresholds
            thresholds = self._get_adaptive_thresholds()
            
            # Get historical performance for this symbol (with caching)
            cache_key = f"{symbol}_{side}"
            if cache_key not in self._cache['symbol_stats']:
                pair_stats = self.signal_scorer.get_symbol_stats(symbol)
                side_stats = self.signal_scorer.get_side_stats(symbol, side)
                self._cache['symbol_stats'][cache_key] = (pair_stats, side_stats)
            else:
                pair_stats, side_stats = self._cache['symbol_stats'][cache_key]
            
            # Build commentary parts with adaptive logic
            parts = []
            
            # Enhanced confidence commentary with context
            if confidence:
                conf_pct = confidence * 100
                if conf_pct >= thresholds['high_confidence']:
                    parts.append(f"🎯 <b>High confidence signal ({conf_pct:.0f}%)</b> - Strong indicator alignment detected.")
                elif conf_pct >= thresholds['moderate_confidence']:
                    parts.append(f"📊 <b>Moderate confidence ({conf_pct:.0f}%)</b> - Good technical setup.")
                else:
                    parts.append(f"⚠️ <b>Lower confidence ({conf_pct:.0f}%)</b> - Proceed with caution, consider smaller position.")
            
            # Enhanced historical performance commentary with sample size awareness
            if pair_stats and pair_stats['total_trades'] >= 3:
                win_rate = pair_stats['win_rate']
                total_trades = pair_stats['total_trades']
                avg_pnl = pair_stats.get('avg_pnl', 0)
                
                # More nuanced assessment
                if total_trades >= 10:
                    reliability = "highly reliable"
                elif total_trades >= 5:
                    reliability = "reliable"
                else:
                    reliability = "early"
                
                if win_rate >= thresholds['good_win_rate']:
                    parts.append(f"✅ <b>{symbol} historically strong</b> ({win_rate:.0f}% win rate, {total_trades} trades - {reliability} data)")
                elif win_rate >= thresholds['poor_win_rate']:
                    parts.append(f"📊 <b>{symbol} mixed performance</b> ({win_rate:.0f}% win rate, {total_trades} trades)")
                else:
                    parts.append(f"⚠️ <b>{symbol} challenging pair</b> ({win_rate:.0f}% win rate, {total_trades} trades - consider avoiding)")
                
                # Add average P&L context
                if avg_pnl != 0:
                    if avg_pnl > 0:
                        parts.append(f"💰 Average profit: <b>${avg_pnl:.2f}</b> per trade")
                    else:
                        parts.append(f"⚠️ Average loss: <b>${abs(avg_pnl):.2f}</b> per trade")
            
            # Enhanced side-specific commentary
            if side_stats and side_stats['trades'] >= 2:
                side_win_rate = side_stats['win_rate']
                if side_win_rate >= 70:
                    parts.append(f"💪 <b>{side} trades excellent</b> on this pair ({side_win_rate:.0f}% win rate)")
                elif side_win_rate >= 50:
                    parts.append(f"✓ <b>{side} trades solid</b> on this pair ({side_win_rate:.0f}% win rate)")
                elif side_win_rate >= 30:
                    parts.append(f"⚡ <b>{side} trades challenging</b> on this pair ({side_win_rate:.0f}% win rate)")
                else:
                    parts.append(f"🔍 <b>{side} trades struggling</b> on this pair ({side_win_rate:.0f}% win rate - consider avoiding {side}s)")
            
            # Enhanced tactic commentary with more detail
            tactic = self._get_enhanced_tactic_comment(symbol, side, confidence, pair_stats)
            if tactic:
                parts.append(tactic)
            
            if parts:
                header = self._t('ai_insight')
                return f"\n\n{header}\n" + "\n".join(parts)
            return ""
            
        except Exception as e:
            self.logger.error(f"Error generating position open commentary: {e}", exc_info=True)
            return ""
    
    def generate_position_close_commentary(self, symbol: str, side: str, pnl: float,
                                          pnl_percent: float, holding_time_hours: float = None) -> str:
        """
        Generate AI commentary for position closing with enhanced analysis
        
        Args:
            symbol: Trading pair symbol
            side: BUY or SELL  
            pnl: Profit/loss amount
            pnl_percent: Profit/loss percentage
            holding_time_hours: Optional holding time in hours
            
        Returns:
            Commentary string with outcome analysis and learning insights
        """
        try:
            parts = []
            
            # Enhanced outcome commentary with risk-reward context
            if pnl > 0:
                if pnl_percent > 10:
                    parts.append(f"🎉 <b>Exceptional trade!</b> {pnl_percent:.1f}% gain - Excellent execution.")
                elif pnl_percent > 5:
                    parts.append(f"🎯 <b>Strong profit!</b> {pnl_percent:.1f}% gain achieved.")
                elif pnl_percent > 2:
                    parts.append(f"✅ <b>Good trade!</b> {pnl_percent:.1f}% solid profit captured.")
                elif pnl_percent > 0.5:
                    parts.append(f"👍 <b>Profitable trade</b> - {pnl_percent:.1f}% gain. Small but consistent wins build success.")
                else:
                    parts.append(f"✓ <b>Minor profit</b> - {pnl_percent:.1f}% gain. Consider longer hold times for better returns.")
            elif pnl < 0:
                if pnl_percent < -10:
                    parts.append(f"🚨 <b>Large loss</b> - {abs(pnl_percent):.1f}% loss. Review entry criteria and stop-loss placement.")
                elif pnl_percent < -5:
                    parts.append(f"⚠️ <b>Significant loss</b> - {abs(pnl_percent):.1f}% loss. Consider tighter risk management.")
                elif pnl_percent < -2:
                    parts.append(f"📉 <b>Loss taken</b> - {abs(pnl_percent):.1f}% loss. Part of disciplined risk management.")
                else:
                    parts.append(f"➖ <b>Minor loss</b> - {abs(pnl_percent):.1f}% loss. Well-controlled risk.")
            else:
                parts.append("➖ <b>Breakeven trade</b> - No gain or loss. Capital preserved.")
            
            # Holding time analysis
            if holding_time_hours is not None:
                if holding_time_hours < 1:
                    parts.append(f"⚡ <b>Quick trade</b> ({holding_time_hours*60:.0f} minutes) - Scalping strategy.")
                elif holding_time_hours < 24:
                    parts.append(f"⏱️ <b>Intraday hold</b> ({holding_time_hours:.1f} hours) - Day trading approach.")
                elif holding_time_hours < 168:
                    parts.append(f"📅 <b>Swing trade</b> ({holding_time_hours/24:.1f} days) - Medium-term position.")
                else:
                    parts.append(f"📊 <b>Position trade</b> ({holding_time_hours/24:.0f} days) - Long-term hold.")
            
            # Get recent performance with caching
            self._refresh_cache_if_needed()
            recent_perf = self._cache.get('performance_7d') or {}
            
            if recent_perf and recent_perf.get('total_trades', 0) >= 3:
                win_rate = recent_perf.get('win_rate', 0)
                recent_pnl = recent_perf.get('total_pnl', 0)
                profit_factor = recent_perf.get('profit_factor', 0)
                
                # More detailed performance assessment
                if win_rate >= 70:
                    parts.append(f"📈 <b>Strategy excelling</b> ({win_rate:.0f}% recent win rate, PF: {profit_factor:.2f})")
                elif win_rate >= 60:
                    parts.append(f"📈 <b>Strategy performing well</b> ({win_rate:.0f}% recent win rate)")
                elif win_rate >= 50:
                    parts.append(f"📊 <b>Strategy balanced</b> ({win_rate:.0f}% recent win rate)")
                elif win_rate >= 40:
                    parts.append(f"⚠️ <b>Strategy underperforming</b> ({win_rate:.0f}% recent win rate)")
                else:
                    parts.append(f"🔍 <b>Strategy needs urgent review</b> ({win_rate:.0f}% recent win rate - consider pausing)")
                
                # Trend analysis
                perf_30d = self._cache.get('performance_30d') or {}
                if perf_30d and perf_30d.get('total_trades', 0) >= 5:
                    win_rate_30d = perf_30d.get('win_rate', 0)
                    if win_rate > win_rate_30d + 5:
                        parts.append("📈 <b>Performance improving</b> - Recent changes working well!")
                    elif win_rate < win_rate_30d - 5:
                        parts.append("📉 <b>Performance declining</b> - Time to review and adjust.")
            
            # Enhanced learning commentary
            if pnl < 0:
                # Provide specific learning points
                if abs(pnl_percent) > 5:
                    parts.append("📚 <b>Key learnings:</b> Analyze entry timing and market conditions at entry.")
                else:
                    parts.append("📚 <b>Learning opportunity</b> - Reviewing to improve future signal quality.")
            else:
                # Positive reinforcement for winning trades
                if pnl_percent > 3:
                    parts.append("💡 <b>Success factor:</b> Capture what worked well for future replication.")
            
            if parts:
                header = self._t('ai_analysis')
                return f"\n\n{header}\n" + "\n".join(parts)
            return ""
            
        except Exception as e:
            self.logger.error(f"Error generating position close commentary: {e}", exc_info=True)
            return ""
    
    def generate_daily_summary_commentary(self, daily_pnl: float, 
                                         open_positions: int) -> str:
        """
        Generate AI commentary for daily summary with comprehensive multi-timeframe analysis
        
        Args:
            daily_pnl: Daily profit/loss
            open_positions: Number of open positions
            
        Returns:
            Commentary string with strategic insights
        """
        try:
            parts = []
            
            # Refresh cache for latest data
            self._refresh_cache_if_needed()
            
            # Get comprehensive performance data
            perf_7d = self._cache.get('performance_7d', {})
            perf_30d = self._cache.get('performance_30d', {})
            metrics = self._cache.get('advanced_metrics', {})
            
            # Enhanced daily performance commentary
            if daily_pnl > 50:
                parts.append(f"🎉 <b>Exceptional day!</b> ${daily_pnl:,.2f} profit secured - Outstanding performance!")
            elif daily_pnl > 20:
                parts.append(f"✅ <b>Great day!</b> ${daily_pnl:,.2f} profit secured.")
            elif daily_pnl > 0:
                parts.append(f"✅ <b>Positive day!</b> ${daily_pnl:,.2f} profit secured.")
            elif daily_pnl < -50:
                parts.append(f"🚨 <b>Challenging day:</b> ${abs(daily_pnl):,.2f} loss - Review risk parameters.")
            elif daily_pnl < -20:
                parts.append(f"⚠️ <b>Red day:</b> ${abs(daily_pnl):,.2f} loss - Analyze what went wrong.")
            elif daily_pnl < 0:
                parts.append(f"📊 <b>Negative day:</b> ${abs(daily_pnl):,.2f} loss - Tomorrow is a new opportunity.")
            else:
                parts.append("➖ <b>Neutral day</b> - Waiting for optimal setups. Patience is key.")
            
            # Enhanced weekly trend with context
            if perf_7d and perf_7d.get('total_trades', 0) >= 3:
                weekly_pnl = perf_7d.get('total_pnl', 0)
                win_rate_7d = perf_7d.get('win_rate', 0)
                total_trades_7d = perf_7d.get('total_trades', 0)
                profit_factor_7d = perf_7d.get('profit_factor', 0)
                
                if weekly_pnl > 100:
                    parts.append(f"🚀 <b>Excellent week!</b> ${weekly_pnl:,.2f} ({win_rate_7d:.0f}% win rate, {total_trades_7d} trades)")
                elif weekly_pnl > 0:
                    parts.append(f"📈 <b>Week trending positive:</b> ${weekly_pnl:,.2f} ({win_rate_7d:.0f}% win rate, {total_trades_7d} trades)")
                elif weekly_pnl > -50:
                    parts.append(f"📊 <b>Week slightly negative:</b> ${abs(weekly_pnl):,.2f} ({win_rate_7d:.0f}% win rate, {total_trades_7d} trades)")
                else:
                    parts.append(f"⚠️ <b>Difficult week:</b> ${abs(weekly_pnl):,.2f} ({win_rate_7d:.0f}% win rate - review needed)")
                
                # Profit factor insight
                if profit_factor_7d and profit_factor_7d > 0:
                    if profit_factor_7d > 2.5:
                        parts.append(f"💎 <b>Exceptional profit factor</b> ({profit_factor_7d:.2f}) - Strategy in prime form!")
                    elif profit_factor_7d < 1:
                        parts.append(f"⚠️ <b>Profit factor concerning</b> ({profit_factor_7d:.2f}) - Losses exceeding wins.")
            
            # Enhanced risk metrics with interpretation
            try:
                sharpe = metrics.get('sharpe_ratio')
                max_dd = metrics.get('max_drawdown_pct')
                max_dd_amount = metrics.get('max_drawdown')
                current_streak = metrics.get('current_streak', {})
                
                if sharpe is not None:
                    if sharpe > 3:
                        parts.append(f"⭐ <b>Exceptional risk-adjusted returns</b> (Sharpe: {sharpe:.2f}) - World-class performance!")
                    elif sharpe > 2:
                        parts.append(f"⭐ <b>Excellent risk-adjusted returns</b> (Sharpe: {sharpe:.2f}) - Outstanding!")
                    elif sharpe > 1:
                        parts.append(f"✅ <b>Good risk management</b> (Sharpe: {sharpe:.2f}) - Solid performance.")
                    elif sharpe > 0:
                        parts.append(f"📊 <b>Moderate risk-adjusted returns</b> (Sharpe: {sharpe:.2f}) - Room for improvement.")
                    else:
                        parts.append(f"⚠️ <b>Poor risk-adjusted returns</b> (Sharpe: {sharpe:.2f}) - Risk not justified by returns.")
                
                if max_dd and max_dd > 0:
                    if max_dd < 5:
                        parts.append(f"💪 <b>Excellent risk control</b> ({max_dd:.1f}% max drawdown) - Very low drawdown!")
                    elif max_dd < 10:
                        parts.append(f"💪 <b>Good risk control</b> ({max_dd:.1f}% max drawdown)")
                    elif max_dd < 20:
                        parts.append(f"📊 <b>Acceptable drawdown</b> ({max_dd:.1f}% max drawdown)")
                    elif max_dd < 30:
                        parts.append(f"⚠️ <b>Elevated drawdown</b> ({max_dd:.1f}% max drawdown) - Consider reducing position sizes")
                    else:
                        parts.append(f"🚨 <b>High drawdown alert</b> ({max_dd:.1f}% max drawdown) - Urgent risk review needed!")
                
                # Streak analysis
                if current_streak:
                    streak_type = current_streak.get('type')
                    streak_count = current_streak.get('count', 0)
                    if streak_type == 'win' and streak_count >= 5:
                        parts.append(f"🔥 <b>Hot streak!</b> {streak_count} wins in a row - Maintain discipline!")
                    elif streak_type == 'loss' and streak_count >= 3:
                        parts.append(f"⚠️ <b>Losing streak</b> ({streak_count} trades) - Consider reducing size or pausing.")
                        
            except Exception as e:
                self.logger.debug(f"Could not add advanced metrics: {e}")
            
            # Enhanced monthly performance with trend
            if perf_30d and perf_30d.get('total_trades', 0) >= 10:
                monthly_pnl = perf_30d.get('total_pnl', 0)
                win_rate_30d = perf_30d.get('win_rate', 0)
                profit_factor_30d = perf_30d.get('profit_factor', 0)
                total_trades_30d = perf_30d.get('total_trades', 0)
                
                if monthly_pnl > 0:
                    roi = (monthly_pnl / 10000) * 100  # Assume 10k starting capital
                    parts.append(f"📅 <b>Monthly performance:</b> ${monthly_pnl:,.2f} profit ({win_rate_30d:.0f}% win rate, ~{roi:.1f}% ROI)")
                else:
                    parts.append(f"📅 <b>Monthly performance:</b> ${abs(monthly_pnl):,.2f} loss ({win_rate_30d:.0f}% win rate)")
                
                if profit_factor_30d and profit_factor_30d > 0:
                    if profit_factor_30d > 2.5:
                        parts.append(f"🎯 <b>Exceptional monthly PF</b> ({profit_factor_30d:.2f}) - Strategy very profitable!")
                    elif profit_factor_30d > 1.5:
                        parts.append(f"🎯 <b>Strong monthly performance</b> (PF: {profit_factor_30d:.2f})")
                    elif profit_factor_30d > 1:
                        parts.append(f"📊 <b>Profitable month</b> (PF: {profit_factor_30d:.2f})")
                    else:
                        parts.append(f"⚠️ <b>Monthly optimization needed</b> (PF: {profit_factor_30d:.2f})")
            
            # Enhanced active positions commentary with risk awareness
            if open_positions > 0:
                max_positions = 5  # Could be from config
                utilization = (open_positions / max_positions) * 100
                
                if open_positions >= max_positions:
                    parts.append(f"⚠️ <b>Maximum positions active</b> ({open_positions}/{max_positions}) - At capacity, no new entries.")
                elif utilization > 70:
                    parts.append(f"👀 <b>High utilization</b> ({open_positions}/{max_positions} positions) - Monitor closely.")
                else:
                    parts.append(f"👀 <b>Monitoring {open_positions} position(s)</b> - Active risk management in place.")
            else:
                parts.append("🔎 <b>No open positions</b> - Scanning for high-quality, high-probability setups.")
            
            # Enhanced strategic recommendation
            strategy_tip = self._get_enhanced_strategy_recommendation(perf_7d, perf_30d, metrics, daily_pnl)
            if strategy_tip:
                parts.append(strategy_tip)
            
            if parts:
                header = self._t('ai_daily_insight')
                return f"\n\n{header}\n" + "\n".join(parts)
            return ""
            
        except Exception as e:
            self.logger.error(f"Error generating daily summary commentary: {e}", exc_info=True)
            return ""
    
    def _get_enhanced_tactic_comment(self, symbol: str, side: str, 
                                    confidence: float = None, pair_stats: Dict = None) -> str:
        """Generate enhanced tactical commentary with more context"""
        tactics = []
        
        # Position sizing tactic based on confidence and historical performance
        if confidence:
            if confidence >= 0.85:
                if pair_stats and pair_stats.get('win_rate', 0) >= 60:
                    tactics.append("Using <b>maximum position size</b> - High confidence + strong historical performance")
                else:
                    tactics.append("Using <b>larger position size</b> - High confidence signal")
            elif confidence >= 0.70:
                tactics.append("Using <b>normal position size</b> - Moderate confidence")
            else:
                tactics.append("Using <b>reduced position size</b> - Lower confidence, conservative approach")
        
        # Strategy type based on side and market conditions
        if side == 'BUY':
            tactics.append("Following <b>long momentum</b> strategy - Capturing upside")
        else:
            tactics.append("Following <b>short reversal</b> strategy - Profiting from downside")
        
        # Risk management approach
        if pair_stats and pair_stats.get('avg_pnl', 0) < 0:
            tactics.append("<b>Tight stop-loss</b> active - Extra caution on challenging pair")
        else:
            tactics.append("<b>Trailing stop</b> active - Protecting profits while allowing upside")
        
        if tactics:
            return "🎲 <b>Tactics:</b> " + " • ".join(tactics)
        return ""
    
    def _get_enhanced_strategy_recommendation(self, perf_7d: Dict, perf_30d: Dict, 
                                             metrics: Dict, daily_pnl: float) -> str:
        """Generate enhanced strategic recommendation with multi-factor analysis"""
        try:
            if not perf_7d or not perf_30d:
                return "💡 <b>Building data:</b> Continue trading to gather performance insights."
            
            win_rate_7d = perf_7d.get('win_rate', 0)
            win_rate_30d = perf_30d.get('win_rate', 0)
            profit_factor_7d = perf_7d.get('profit_factor', 0)
            profit_factor_30d = perf_30d.get('profit_factor', 0)
            sharpe = metrics.get('sharpe_ratio', 0) if metrics else 0
            max_dd = metrics.get('max_drawdown_pct', 0) if metrics else 0
            
            # Strong improvement trend
            if win_rate_7d > win_rate_30d + 15 and profit_factor_7d > 1.5:
                return "🚀 <b>Momentum building!</b> Recent optimizations driving strong results. Continue current approach with slight size increase."
            
            # Moderate improvement
            if win_rate_7d > win_rate_30d + 10:
                return "📈 <b>Strategy improving!</b> Recent adjustments showing positive results. Stay the course."
            
            # Declining trend - critical
            if win_rate_7d < win_rate_30d - 15 or (win_rate_7d < 40 and profit_factor_7d < 1):
                return "🚨 <b>Urgent review needed:</b> Reduce trading frequency immediately. Focus only on highest-confidence (>85%) setups."
            
            # Declining trend - moderate
            if win_rate_7d < win_rate_30d - 10 and win_rate_7d < 50:
                return "⚠️ <b>Recommendation:</b> Reduce position sizes by 30%. Focus on quality over quantity - only trade 75%+ confidence signals."
            
            # High drawdown alert
            if max_dd > 25:
                return "🚨 <b>Risk alert:</b> Drawdown elevated. Reduce all position sizes by 50% until drawdown recovers below 15%."
            
            # Excellent performance - scale up carefully
            if win_rate_30d >= 70 and sharpe > 2 and profit_factor_30d > 2:
                return "🎯 <b>Peak performance!</b> Strategy in optimal zone. Can consider 20% position size increase while maintaining discipline."
            
            # Strong consistent performance
            if win_rate_30d >= 60 and profit_factor_30d > 1.5:
                return "🎯 <b>Strategy working excellently</b> - Maintain current approach, risk levels, and discipline."
            
            # Good performance
            if win_rate_30d >= 55:
                return "✅ <b>Solid performance</b> - Stay consistent with current strategy and position sizing."
            
            # Needs improvement but not critical
            if win_rate_30d >= 45:
                return "📊 <b>Performance acceptable</b> - Minor tweaks needed. Review losing trades to identify improvements."
            
            # Poor performance
            if win_rate_30d < 40:
                return "🔧 <b>Strategy optimization critical</b> - Review signal filters, entry/exit rules. Consider paper trading changes first."
            
            # Neutral/building
            return "📊 <b>Continue current approach</b> - Monitor performance and adjust as data accumulates."
            
        except Exception as e:
            self.logger.debug(f"Could not generate strategy recommendation: {e}")
            return ""


# Singleton instance
_commentary_generator = None

def get_commentary_generator(logger: Optional[logging.Logger] = None, 
                            db_path: str = '/var/lib/trading-bot/trading_bot.db',
                            language: str = None) -> AICommentaryGenerator:
    """Get or create the commentary generator singleton"""
    global _commentary_generator
    if _commentary_generator is None:
        _commentary_generator = AICommentaryGenerator(logger=logger, db_path=db_path, language=language)
    return _commentary_generator
