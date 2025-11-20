"""
RAG Evaluation Library

Modular components for robust, resumable, and observable testset generation.
"""

__version__ = "1.0.0"

from .logger import setup_logger
from .state_manager import CheckpointManager
from .result_writer import IncrementalCSVWriter
from .progress_tracker import DetailedProgressTracker
from .api_validator import APIValidator
from .error_handlers import retry_with_backoff, classify_error

__all__ = [
    "setup_logger",
    "CheckpointManager",
    "IncrementalCSVWriter",
    "DetailedProgressTracker",
    "APIValidator",
    "retry_with_backoff",
    "classify_error",
]
