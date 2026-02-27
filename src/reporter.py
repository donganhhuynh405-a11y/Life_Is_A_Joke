import os
import logging
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib.pyplot as plt

logger = logging.getLogger('bot.reporter')

class Reporter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = cfg.get('report',{}).get('output_dir','./reports')
        os.makedirs(self.output_dir, exist_ok=True)

    async def start(self):
        # schedule weekly report job externally via cron or scheduler
        pass

    async def stop(self):
        pass

    def generate_daily_report(self, metrics: dict, trades: list, trends: dict = None, filename=None):
        """Generate daily trading report with trend analysis"""
        filename = filename or f"daily_report_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
        path = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(path)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph(f'Ежедневный отчёт - {datetime.utcnow().strftime("%Y-%m-%d")}', styles['Title']))
        story.append(Spacer(1, 12))
        
        # Trend Analysis Section (if available)
        if trends:
            story.append(Paragraph('Анализ рыночных трендов', styles['Heading2']))
            story.append(Spacer(1, 6))
            
            # Create trend summary table
            trend_data = [['Символ', 'Тренд', 'Сила', 'ADX', 'Рекомендация']]
            for symbol, trend_info in trends.items():
                trend_emoji = {
                    'BULLISH': '📈',
                    'BEARISH': '📉',
                    'SIDEWAYS': '↔️'
                }.get(trend_info.get('trend', 'SIDEWAYS'), '❓')
                
                trend_data.append([
                    symbol,
                    f"{trend_emoji} {trend_info.get('trend', 'N/A')}",
                    f"{trend_info.get('strength', 0)*100:.1f}%",
                    f"{trend_info.get('adx', 0):.1f}",
                    trend_info.get('recommendation', 'N/A')[:30] + '...'
                ])
            
            trend_table = Table(trend_data)
            trend_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(trend_table)
            story.append(Spacer(1, 12))
            
            # Market summary
            bullish = sum(1 for t in trends.values() if t.get('trend') == 'BULLISH')
            bearish = sum(1 for t in trends.values() if t.get('trend') == 'BEARISH')
            sideways = sum(1 for t in trends.values() if t.get('trend') == 'SIDEWAYS')
            
            summary_text = f"Восходящих: {bullish} | Нисходящих: {bearish} | Боковых: {sideways}"
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Trading Metrics Section
        story.append(Paragraph('Торговые показатели', styles['Heading2']))
        story.append(Spacer(1, 6))
        
        for k, v in metrics.items():
            story.append(Paragraph(f"{k}: {v}", styles['Normal']))
        
        story.append(Spacer(1, 12))
        
        # Trades Section
        if trades:
            story.append(Paragraph(f'Сделки (всего: {len(trades)})', styles['Heading2']))
            story.append(Spacer(1, 6))
            
            # Create trades table
            trade_data = [['Символ', 'Тип', 'Цена', 'Количество', 'Время']]
            for trade in trades[-10:]:  # Last 10 trades
                trade_data.append([
                    str(trade.get('symbol', 'N/A')),
                    str(trade.get('side', 'N/A')),
                    str(trade.get('price', 'N/A')),
                    str(trade.get('quantity', 'N/A')),
                    str(trade.get('time', 'N/A'))[:16]
                ])
            
            trades_table = Table(trade_data)
            trades_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(trades_table)
        
        doc.build(story)
        logger.info('Generated daily report at %s', path)
        return path

    def generate_weekly_report(self, metrics: dict, trades: list, trends: dict = None, filename=None):
        """Generate weekly trading report with trend analysis"""
        # Similar to daily but with weekly aggregation
        filename = filename or f"weekly_report_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
        path = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(path)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph('Недельный торговый отчёт', styles['Title']))
        story.append(Spacer(1, 12))
        
        # Add trend analysis if available
        if trends:
            story.append(Paragraph('Анализ недельных трендов', styles['Heading2']))
            story.append(Spacer(1, 6))
            story.append(Paragraph('Обзор рыночных трендов за неделю', styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Add metrics
        for k, v in metrics.items():
            story.append(Paragraph(f"{k}: {v}", styles['Normal']))
        
        story.append(Spacer(1, 12))
        doc.build(story)
        logger.info('Generated weekly PDF report at %s', path)
        return path

    def sample_plot_equity(self, equity_curve, outpath):
        plt.figure(figsize=(8,4))
        plt.plot(equity_curve)
        plt.title('Equity Curve')
        plt.savefig(outpath)
        plt.close()
    
    def generate_hourly_report(self, metrics: dict, trends: dict = None, filename=None):
        """Generate hourly trading report with trend analysis and trading strategy"""
        filename = filename or f"hourly_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf"
        path = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(path)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph(
            f'Ежечасный отчёт - {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}', 
            styles['Title']
        ))
        story.append(Spacer(1, 12))
        
        # Trend Analysis Section (if available)
        if trends:
            story.append(Paragraph('🔍 Анализ рыночных трендов', styles['Heading2']))
            story.append(Spacer(1, 6))
            
            # Overall market sentiment
            bullish = sum(1 for t in trends.values() if t.get('trend') == 'BULLISH')
            bearish = sum(1 for t in trends.values() if t.get('trend') == 'BEARISH')
            sideways = sum(1 for t in trends.values() if t.get('trend') == 'SIDEWAYS')
            total = len(trends)
            
            # Market overview
            overview_text = f"<b>Обзор рынка:</b> "
            if bullish > bearish and bullish > sideways:
                overview_text += f"Преобладает восходящий тренд ({bullish}/{total}). "
                overview_text += "Благоприятные условия для покупок."
            elif bearish > bullish and bearish > sideways:
                overview_text += f"Преобладает нисходящий тренд ({bearish}/{total}). "
                overview_text += "Осторожно с покупками, рассмотреть продажи."
            else:
                overview_text += f"Смешанный рынок. "
                overview_text += "Требуется избирательный подход."
            
            story.append(Paragraph(overview_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Detailed trend table
            trend_data = [['Символ', 'Тренд', 'Сила', 'ADX', 'Стратегия торговли']]
            for symbol, trend_info in trends.items():
                trend_emoji = {
                    'BULLISH': '📈',
                    'BEARISH': '📉',
                    'SIDEWAYS': '↔️'
                }.get(trend_info.get('trend', 'SIDEWAYS'), '❓')
                
                # Generate trading strategy based on trend
                trend_type = trend_info.get('trend', 'SIDEWAYS')
                strength = trend_info.get('strength', 0)
                adx = trend_info.get('adx', 0)
                
                if trend_type == 'BULLISH':
                    if strength > 0.7 and adx > 30:
                        strategy = "Активно покупать при откатах"
                    elif strength > 0.5:
                        strategy = "Покупать на сигналах"
                    else:
                        strategy = "Осторожные покупки"
                elif trend_type == 'BEARISH':
                    if strength > 0.7 and adx > 30:
                        strategy = "Избегать покупок, SHORT"
                    elif strength > 0.5:
                        strategy = "Не покупать, ждать"
                    else:
                        strategy = "Минимальные позиции"
                else:
                    if adx < 20:
                        strategy = "Ждать прорыва"
                    else:
                        strategy = "Range-торговля"
                
                trend_data.append([
                    symbol,
                    f"{trend_emoji} {trend_type}",
                    f"{strength*100:.1f}%",
                    f"{adx:.1f}",
                    strategy
                ])
            
            trend_table = Table(trend_data)
            trend_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(trend_table)
            story.append(Spacer(1, 12))
            
            # Trading plan summary
            story.append(Paragraph('📋 План торговли на следующий час:', styles['Heading3']))
            story.append(Spacer(1, 6))
            
            # Generate specific trading plan
            strong_bullish = [s for s, t in trends.items() 
                            if t.get('trend') == 'BULLISH' and t.get('strength', 0) > 0.6]
            strong_bearish = [s for s, t in trends.items() 
                            if t.get('trend') == 'BEARISH' and t.get('strength', 0) > 0.6]
            
            plan_text = ""
            if strong_bullish:
                plan_text += f"<b>Приоритет на покупку:</b> {', '.join(strong_bullish[:3])}<br/>"
            if strong_bearish:
                plan_text += f"<b>Избегать покупок:</b> {', '.join(strong_bearish[:3])}<br/>"
            if not strong_bullish and not strong_bearish:
                plan_text += "<b>Режим ожидания:</b> Слабые тренды, ждём лучших возможностей<br/>"
            
            # Add risk management note
            plan_text += f"<br/><b>Управление рисками:</b> Адаптация размера позиций под силу тренда"
            
            story.append(Paragraph(plan_text, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Current Metrics Section
        story.append(Paragraph('📊 Текущие показатели', styles['Heading2']))
        story.append(Spacer(1, 6))
        
        for k, v in metrics.items():
            story.append(Paragraph(f"{k}: {v}", styles['Normal']))
        
        story.append(Spacer(1, 12))
        
        # Footer with timestamp
        story.append(Paragraph(
            f"Отчёт сгенерирован: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            styles['Normal']
        ))
        
        doc.build(story)
        logger.info('Generated hourly report at %s', path)
        return path
