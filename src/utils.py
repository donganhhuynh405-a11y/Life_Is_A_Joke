import asyncio
import time
import functools
import logging
from typing import Callable, Any

logger = logging.getLogger('bot.utils')

def retry_async(retries=5, delay=1, backoff=2, exceptions=(Exception,)):
    def deco(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            _delay = delay
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.warning('Retry %s/%s for %s due to %s', i+1, retries, func.__name__, e)
                    await asyncio.sleep(_delay)
                    _delay *= backoff
            return await func(*args, **kwargs)
        return wrapper
    return deco

def retry_sync(retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    def deco(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _delay = delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    time.sleep(_delay)
                    _delay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return deco

class WALLogger:
    def __init__(self, path='wal.log'):
        self.path = path

    def write(self, record: str):
        with open(self.path,'a',encoding='utf8') as f:
            f.write(record + '\n')
