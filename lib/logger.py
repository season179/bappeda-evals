"""
Structured logging system for RAG evaluation

Provides multiple log handlers for console, file, API calls, and errors.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "ragas_eval",
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_api_log: bool = True,
    enable_error_log: bool = True
) -> logging.Logger:
    """
    Setup structured logging with multiple handlers

    Args:
        name: Logger name
        log_dir: Directory for log files
        console_level: Minimum level for console output
        file_level: Minimum level for file output
        enable_api_log: Whether to create separate API call log
        enable_error_log: Whether to create separate error log

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    colored_formatter = ColoredFormatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # 1. Console Handler (with colors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)

    # 2. Main File Handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_log_file = log_path / f"main_{timestamp}.log"
    file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # 3. API Calls Log (separate file for debugging)
    if enable_api_log:
        api_log_file = log_path / "api_calls.log"
        api_handler = logging.FileHandler(api_log_file, mode='a', encoding='utf-8')
        api_handler.setLevel(logging.DEBUG)
        api_handler.setFormatter(detailed_formatter)
        # Only log API-related messages
        api_handler.addFilter(lambda record: 'API' in record.getMessage() or 'api' in record.name.lower())
        logger.addHandler(api_handler)

    # 4. Error Log (ERROR and above only)
    if enable_error_log:
        error_log_file = log_path / "errors.log"
        error_handler = logging.FileHandler(error_log_file, mode='a', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)

    # Log initialization
    logger.info(f"Logging initialized - Main log: {main_log_file}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get existing logger or create new one

    Args:
        name: Logger name (defaults to root ragas_eval logger)

    Returns:
        Logger instance
    """
    if name is None:
        name = "ragas_eval"
    return logging.getLogger(name)
