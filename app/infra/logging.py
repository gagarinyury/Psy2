import sys
from loguru import logger


def setup_logging():
    """Setup JSON logging with loguru"""
    # Remove default handler
    logger.remove()

    # Add JSON handler for stdout
    logger.add(
        sys.stdout,
        format="{time} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        serialize=True,  # This enables JSON output
        backtrace=True,
        diagnose=True,
    )


def get_logger():
    """Get configured logger instance"""
    return logger
