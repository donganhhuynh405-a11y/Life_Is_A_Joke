"""
Comprehensive example showing weekly report generation with all key metrics and plots.
"""
import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_example_report():
    """Generate a realistic weekly trading report PDF"""
    os.makedirs('./reports', exist_ok=True)
    filename = f"./reports/Weekly_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=1
    )
    story.append(Paragraph('Weekly Trading Performance Report', title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Date info
    story.append(Paragraph(f'Week of: {datetime.utcnow().strftime("%Y-%m-%d")}', styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Metrics Table
    metrics_data = [
        ['Metric', 'Value', 'Target', 'Status'],
        ['PnL (Total)', '$2,847.50', 'N/A', '✓'],
        ['PnL (Weekly)', '$842.35', '>$666.67', '✓'],
        ['Return %', '10.2%', '10.0%', '✓'],
        ['Sharpe Ratio', '3.24', '>3.0', '✓'],
        ['Win Rate', '72.5%', '>70%', '✓'],
        ['Max Drawdown', '3.8%', '<5%', '✓'],
        ['Profit Factor', '2.15', '>1.5', '✓'],
        ['Trades', '148', 'N/A', 'Active'],
        ['ROI (Annualized)', '62.4%', '>60%', '✓']
    ]
    
    tbl = Table(metrics_data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 0.8*inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.3*inch))
    
    # Generate sample charts
    # Equity curve
    dates = np.arange(0, 7, 0.1)
    equity = 10000 + 100 * np.cumsum(np.random.randn(len(dates)))
    
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(dates, equity, linewidth=2, color='#1f77b4')
    ax.set_title('Equity Curve (7 days)', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)
    ax.get_figure().tight_layout()
    chart_path = './reports/equity_curve.png'
    fig.savefig(chart_path)
    plt.close(fig)
    
    story.append(Paragraph('Equity Curve', styles['Heading2']))
    story.append(Image(chart_path, width=5*inch, height=2.5*inch))
    story.append(Spacer(1, 0.2*inch))
    
    # Drawdown chart
    fig, ax = plt.subplots(figsize=(7, 3))
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    ax.fill_between(dates, 0, drawdown, alpha=0.5, color='#d62728')
    ax.set_title('Drawdown (%)', fontsize=12)
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    ax.get_figure().tight_layout()
    dd_chart = './reports/drawdown.png'
    fig.savefig(dd_chart)
    plt.close(fig)
    
    story.append(PageBreak())
    story.append(Paragraph('Drawdown Analysis', styles['Heading2']))
    story.append(Image(dd_chart, width=5*inch, height=2.5*inch))
    story.append(Spacer(1, 0.3*inch))
    
    # Summary text
    story.append(Paragraph('Summary', styles['Heading2']))
    summary_text = """
    This week the strategy demonstrated strong performance with a 10.2% return, 
    exceeding the 10% weekly target. The Sharpe ratio of 3.24 indicates excellent 
    risk-adjusted returns, with a win rate of 72.5% and profit factor of 2.15. 
    Maximum drawdown remained contained at 3.8%, well below the 5% limit. 
    The system executed 148 trades across multiple symbols (BTCUSDT, ETHUSDT, SOLUSDT) 
    using adaptive risk management and dynamic leverage (1-10x based on volatility).
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    
    doc.build(story)
    print(f'Generated report: {filename}')
    return filename

if __name__ == '__main__':
    path = generate_example_report()
    print(f'✓ Weekly report PDF created at: {path}')
