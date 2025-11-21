"""
Detailed progress tracking for testset generation

Provides real-time visibility into generation progress with ETA.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .logger import get_logger


class DetailedProgressTracker:
    """Tracks detailed progress with phase, document, and sample information"""

    def __init__(
        self,
        total_documents: int = 0,
        target_samples: int = 0,
        progress_file: str = "progress_summary.json",
        update_interval: int = 30
    ):
        """
        Initialize progress tracker

        Args:
            total_documents: Total number of documents to process
            target_samples: Target number of samples to generate
            progress_file: Path to progress summary file
            update_interval: Minimum seconds between file updates
        """
        self.total_documents = total_documents
        self.target_samples = target_samples
        self.progress_file = Path(progress_file)
        self.update_interval = update_interval
        self.logger = get_logger(__name__)

        # Internal state
        self.start_time = time.time()
        self.last_update_time = 0
        self.current_phase: Optional[str] = None
        self.current_document: Optional[str] = None
        self.documents_processed = 0
        self.samples_generated = 0
        self.status = "initializing"

        # Initialize progress file
        self._update_progress_file(force=True)

    def start(self) -> None:
        """Mark tracking as started"""
        self.status = "running"
        self.start_time = time.time()
        self._update_progress_file(force=True)
        self.logger.info("Progress tracking started")

    def update_phase(self, phase: str) -> None:
        """
        Update current processing phase

        Args:
            phase: Name of current phase (e.g., "HeadlinesExtractor")
        """
        self.current_phase = phase
        self.logger.info(f"Phase: {phase}")
        self._update_progress_file()

    def update_document(self, document_name: str, index: int) -> None:
        """
        Update current document being processed

        Args:
            document_name: Name of current document
            index: Document index (0-based)
        """
        self.current_document = document_name
        self.documents_processed = index
        self.logger.info(f"Processing document: {document_name} ({index + 1}/{self.total_documents})")
        self._update_progress_file()

    def document_completed(self) -> None:
        """Mark current document as completed"""
        self.documents_processed += 1
        if self.current_document:
            self.logger.info(f"Completed document: {self.current_document} ({self.documents_processed}/{self.total_documents})")
        self._update_progress_file()

    def add_samples(self, count: int) -> None:
        """
        Add generated samples to count

        Args:
            count: Number of samples to add
        """
        self.samples_generated += count
        self.logger.info(f"Generated {count} samples (total: {self.samples_generated}/{self.target_samples})")
        self._update_progress_file()

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time

    def get_estimated_remaining(self) -> Optional[float]:
        """
        Calculate estimated remaining time in seconds

        Returns:
            Estimated seconds remaining, or None if not enough data
        """
        if self.samples_generated == 0:
            return None

        elapsed = self.get_elapsed_time()
        rate = self.samples_generated / elapsed  # samples per second
        remaining_samples = self.target_samples - self.samples_generated

        if rate > 0:
            return remaining_samples / rate
        return None

    def format_time(self, seconds: float) -> str:
        """
        Format seconds as HH:MM:SS or MM:SS

        Args:
            seconds: Seconds to format

        Returns:
            Formatted time string
        """
        if seconds < 0:
            return "00:00"

        delta = timedelta(seconds=int(seconds))
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        secs = delta.seconds % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def get_progress_bar(self, width: int = 50) -> str:
        """
        Generate text progress bar

        Args:
            width: Width of progress bar in characters

        Returns:
            Progress bar string
        """
        if self.target_samples == 0:
            percentage = 0
        else:
            percentage = (self.samples_generated / self.target_samples) * 100

        filled = int(width * percentage / 100)
        bar = '#' * filled + '-' * (width - filled)

        return f"[{bar}] {percentage:.1f}%"

    def get_status_line(self) -> str:
        """
        Get comprehensive status line

        Returns:
            Formatted status line with all key information
        """
        elapsed = self.format_time(self.get_elapsed_time())
        remaining = self.get_estimated_remaining()
        eta = self.format_time(remaining) if remaining else "N/A"

        phase_str = f"Phase: {self.current_phase}" if self.current_phase else "Phase: Initializing"
        doc_str = f"Doc: {self.current_document or 'N/A'} ({self.documents_processed}/{self.total_documents})"
        samples_str = f"Samples: {self.samples_generated}/{self.target_samples}"
        time_str = f"Time: {elapsed} | ETA: ~{eta}"

        return f"[{phase_str}] {doc_str} | {samples_str} | {time_str}"

    def _update_progress_file(self, force: bool = False) -> None:
        """
        Update progress summary file

        Args:
            force: Force update even if interval hasn't elapsed
        """
        current_time = time.time()

        # Check if enough time has passed since last update
        if not force and (current_time - self.last_update_time) < self.update_interval:
            return

        try:
            progress_data = {
                "started_at": datetime.fromtimestamp(self.start_time).isoformat(),
                "last_updated": datetime.now().isoformat(),
                "status": self.status,
                "current_phase": self.current_phase,
                "current_document": self.current_document,
                "documents_processed": self.documents_processed,
                "total_documents": self.total_documents,
                "samples_generated": self.samples_generated,
                "target_samples": self.target_samples,
                "elapsed_seconds": self.get_elapsed_time(),
                "estimated_remaining_seconds": self.get_estimated_remaining(),
                "progress_percentage": (self.samples_generated / self.target_samples * 100) if self.target_samples > 0 else 0
            }

            # Write to temp file first, then rename (atomic)
            temp_file = self.progress_file.with_suffix('.json.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)

            temp_file.replace(self.progress_file)
            self.last_update_time = current_time

        except Exception as e:
            self.logger.debug(f"Failed to update progress file: {e}")

    def complete(self) -> None:
        """Mark tracking as completed"""
        self.status = "completed"
        self.samples_generated = self.target_samples
        self._update_progress_file(force=True)

        elapsed = self.format_time(self.get_elapsed_time())
        self.logger.info(f"Progress tracking completed - Total time: {elapsed}")

    def error(self, error_message: str) -> None:
        """
        Mark tracking as errored

        Args:
            error_message: Error message to record
        """
        self.status = "error"
        self._update_progress_file(force=True)
        self.logger.error(f"Progress tracking stopped due to error: {error_message}")

    def get_summary(self) -> str:
        """
        Get human-readable progress summary

        Returns:
            Formatted summary string
        """
        elapsed = self.format_time(self.get_elapsed_time())
        remaining = self.get_estimated_remaining()
        eta = self.format_time(remaining) if remaining else "N/A"
        progress_bar = self.get_progress_bar(40)

        lines = [
            "Progress Summary:",
            f"  Status: {self.status}",
            f"  Phase: {self.current_phase or 'N/A'}",
            f"  Documents: {self.documents_processed}/{self.total_documents}",
            f"  Samples: {self.samples_generated}/{self.target_samples}",
            f"  {progress_bar}",
            f"  Elapsed: {elapsed}",
            f"  ETA: ~{eta}"
        ]
        return "\n".join(lines)
