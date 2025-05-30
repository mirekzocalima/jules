import os
import yaml
from pathlib import Path

ENV = os.environ.get('ENV', 'dev')

CONFIG_PATH = Path('/config')

if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).parent.parent.parent/ 'config'

ENV_PATH = CONFIG_PATH / f'{ENV}.yaml'
RAO_PATH = CONFIG_PATH / 'rao.yaml'


def load_config(file_path: Path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def load_env_config():
    return load_config(ENV_PATH)


def load_rao_config():
    return load_config(RAO_PATH)
