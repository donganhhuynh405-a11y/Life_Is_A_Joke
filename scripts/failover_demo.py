import asyncio
import time
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('failover_demo')

class Node:
    def __init__(self, name):
        self.name = name
        self.healthy = True
        self.latency = random.uniform(50, 150)

    async def measure_health(self):
        self.latency = random.uniform(30, 200) if random.random()>0.1 else random.uniform(400, 600)
        if self.latency > 500:
            self.healthy = False
        else:
            self.healthy = True
        logger.info('%s: latency %dms, healthy=%s', self.name, int(self.latency), self.healthy)

class FailoverCluster:
    def __init__(self, nodes):
        self.nodes = nodes
        self.active = self.nodes[0]

    async def monitor(self):
        for _ in range(15):
            for n in self.nodes:
                await n.measure_health()
            
            healthy_nodes = [n for n in self.nodes if n.healthy]
            if not self.active.healthy and healthy_nodes:
                old_active = self.active
                self.active = healthy_nodes[0]
                logger.warning('FAILOVER: %s -> %s', old_active.name, self.active.name)
            
            logger.info('Active node: %s', self.active.name)
            await asyncio.sleep(1)

async def run_demo():
    nodes = [Node('Primary-AWS'), Node('Backup-Hetzner'), Node('Local-Host')]
    cluster = FailoverCluster(nodes)
    await cluster.monitor()

if __name__ == '__main__':
    asyncio.run(run_demo())
