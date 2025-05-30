import os
import yaml
from pathlib import Path

CONFIG_DIR = Path('/config')

ENV = os.environ.get('ENV', 'dev')

CONFIG_PATH = os.path.join(CONFIG_DIR, f'{ENV}.yaml')

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)
