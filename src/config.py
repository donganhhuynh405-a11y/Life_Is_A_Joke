import os
import yaml


class AttrDict(dict):
    """Dictionary that allows attribute access"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        # Recursively convert nested dicts
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)
            elif isinstance(value, list):
                self[key] = [AttrDict(item) if isinstance(item, dict) else item for item in value]


_cfg = None

def load_config(path=None):
    global _cfg
    if _cfg:
        return _cfg
    path = path or os.environ.get('CONFIG_PATH') or os.path.join(os.getcwd(),'config.yaml')
    with open(path,'r',encoding='utf8') as f:
        data = yaml.safe_load(f)
        _cfg = AttrDict(data)
    return _cfg

def get_redis_url():
    cfg = load_config()
    # Safe nested access with defaults
    if isinstance(cfg, dict):
        return cfg.get('database', {}).get('redis', {}).get('url', 'redis://localhost:6379/0')
    # For AttrDict, safely navigate the hierarchy
    database = getattr(cfg, 'database', None)
    if database:
        redis = getattr(database, 'redis', None)
        if redis:
            return getattr(redis, 'url', 'redis://localhost:6379/0')
    return 'redis://localhost:6379/0'
