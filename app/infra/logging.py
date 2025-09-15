import sys

from loguru import logger

from app.core.settings import settings


def setup_logging():
    """Setup JSON logging with loguru"""
    # Remove default handler
    logger.remove()

    # Add JSON handler for stdout
    logger.add(
        sys.stdout,
        format="{time} | {level} | {name}:{function}:{line} | {message}",
        level=settings.LOG_LEVEL,
        serialize=True,  # This enables JSON output
        backtrace=True,
        diagnose=True,
    )


def get_logger():
    """Get configured logger instance"""
    return logger
