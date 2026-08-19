import logging
import sys
from . import config

def setup_logging():
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    
    # Avoid duplicate handlers if setup is called multiple times
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(log_level)

def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
