"""logging_config.py -- Structured logging setup."""
import logging
import sys

def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("aiuta_vlmr1")
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger
