"""Logging utilities for the self-improving LLM agent."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with Rich formatting
    if console:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(getattr(logging, level.upper()))
        console_format = logging.Formatter("%(message)s", datefmt="[%X]")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Create default logger
default_logger = setup_logger(
    name="self_improving_agent",
    log_file="./data/logs/agent.log",
    level="INFO",
    console=True,
)


def log_info(message: str) -> None:
    """Log info message."""
    default_logger.info(message)


def log_warning(message: str) -> None:
    """Log warning message."""
    default_logger.warning(message)


def log_error(message: str) -> None:
    """Log error message."""
    default_logger.error(message)


def log_debug(message: str) -> None:
    """Log debug message."""
    default_logger.debug(message)


# import logging

# def setup_logging(name:str)-> logging.Logger:
#     logger = logging.getLogger(name)

#     if logger.handlers:
#         return logger

#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s')
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(formatter)

#     logger.addHandler(console_handler)

#     return logger
