import asyncio
import logging
from prometheus_client import start_http_server, Gauge

logger = logging.getLogger('bot.health')

class HealthMonitor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.heartbeat = Gauge('bot_heartbeat', 'Heartbeat timestamp')
        self.latency = Gauge('bot_latency_ms', 'API latency ms')
        self.running = False

    async def start(self):
        self.running = True
        # start prometheus metrics server
        start_http_server(8001)
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self.running = False
        # Cancel the task to prevent "Task was destroyed" warning
        if hasattr(self, '_task') and self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        import time
        while self.running:
            self.heartbeat.set(time.time())
            # measure latency to exchanges or nodes here
            self.latency.set(0)
            await asyncio.sleep(5)
