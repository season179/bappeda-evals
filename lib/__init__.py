"""
RAG Evaluation Library

Modular components for robust, resumable, and observable testset generation.
"""

__version__ = "1.0.0"

from .logger import setup_logger, setup_logger_from_config
from .state_manager import CheckpointManager
from .result_writer import IncrementalCSVWriter
from .progress_tracker import DetailedProgressTracker
from .api_validator import APIValidator
from .metadata_loader import MetadataDocument, MetadataCache

__all__ = [
    "setup_logger",
    "setup_logger_from_config",
    "CheckpointManager",
    "IncrementalCSVWriter",
    "DetailedProgressTracker",
    "APIValidator",
    "MetadataDocument",
    "MetadataCache",
]
