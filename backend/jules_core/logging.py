import datetime
import logging
import sys

from typing import Optional
from pathlib import Path

LOGGER_FORMAT = '{asctime} {levelname} {name}:{lineno}: {message}'


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up logging to both stdout and file (if results_dir is provided).

    Args:
        log_dir: Optional directory for log file. If None, only logs to stdout.

    Returns:
        Root logger configured with the specified format.
    """
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    
    # Setup stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(LOGGER_FORMAT, style='{'))
    stdout_handler.setLevel(level)
    root_logger.addHandler(stdout_handler)
