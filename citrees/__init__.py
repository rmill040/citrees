import sys

from loguru import logger

from citrees._base import Node  # noqa

# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | <lvl>{level}</lvl> | <lvl>{message}</lvl>",
    level="DEBUG",
)
