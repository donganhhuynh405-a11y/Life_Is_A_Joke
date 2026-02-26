import backtrader as bt
import datetime
import logging

logger = logging.getLogger('bot.backtest')

class SampleStrategy(bt.Strategy):
    params = dict()

    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        if not self.position:
            if self.dataclose[0] > self.dataclose[-1]:
                self.buy(size=0.01)
        else:
            if self.dataclose[0] < self.dataclose[-1]:
                self.close()

def run_backtest(datafeed, cerebro=None):
    cerebro = cerebro or bt.Cerebro()
    cerebro.addstrategy(SampleStrategy)
    cerebro.adddata(datafeed)
    cerebro.broker.setcash(10000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)
    results = cerebro.run()
    pnl = cerebro.broker.getvalue() - 10000.0
    logger.info('Backtest PnL: %s', pnl)
    return results, pnl
