"""
Checkpoint and state management for resumable testset generation and evaluation

Enables saving and loading progress to survive errors and crashes.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logger import get_logger


class CheckpointManager:
    """Manages checkpoint state for resumable execution"""

    def __init__(self, checkpoint_file: str = "checkpoint.json"):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.logger = get_logger(__name__)
        self._state: Dict[str, Any] = self._init_state()

    def _init_state(self) -> Dict[str, Any]:
        """Initialize empty state structure"""
        return {
            "version": "1.0",
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "processed_documents": [],
            "generated_samples": [],
            "current_phase": None,
            "current_document_index": 0,
            "samples_count": 0,
            "target_samples": 0,
            "total_documents": 0,
            "elapsed_seconds": 0,
            "metadata": {}
        }

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint file exists"""
        return self.checkpoint_file.exists()

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint from file

        Returns:
            State dictionary

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            json.JSONDecodeError: If checkpoint file is corrupted
        """
        if not self.has_checkpoint():
            self.logger.warning(f"No checkpoint found at {self.checkpoint_file}")
            return self._init_state()

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                self._state = json.load(f)

            self.logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
            self.logger.info(f"  Documents processed: {len(self._state['processed_documents'])}/{self._state['total_documents']}")
            self.logger.info(f"  Samples generated: {self._state['samples_count']}/{self._state['target_samples']}")
            self.logger.info(f"  Last updated: {self._state['last_updated']}")

            return self._state

        except json.JSONDecodeError as e:
            self.logger.error(f"Corrupted checkpoint file: {e}")
            # Backup corrupted file
            backup_path = self.checkpoint_file.with_suffix('.json.corrupted')
            shutil.copy(self.checkpoint_file, backup_path)
            self.logger.warning(f"Backed up corrupted checkpoint to {backup_path}")
            raise

    def save_checkpoint(
        self,
        processed_documents: Optional[List[str]] = None,
        generated_samples: Optional[List[Dict]] = None,
        current_phase: Optional[str] = None,
        current_document_index: Optional[int] = None,
        samples_count: Optional[int] = None,
        target_samples: Optional[int] = None,
        total_documents: Optional[int] = None,
        elapsed_seconds: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save checkpoint to file

        Args:
            processed_documents: List of processed document filenames
            generated_samples: List of generated samples
            current_phase: Current processing phase
            current_document_index: Index of current document
            samples_count: Number of samples generated so far
            target_samples: Target number of samples
            total_documents: Total number of documents
            elapsed_seconds: Total elapsed time
            metadata: Additional metadata
        """
        # Update only provided fields
        if processed_documents is not None:
            self._state['processed_documents'] = processed_documents
        if generated_samples is not None:
            self._state['generated_samples'] = generated_samples
        if current_phase is not None:
            self._state['current_phase'] = current_phase
        if current_document_index is not None:
            self._state['current_document_index'] = current_document_index
        if samples_count is not None:
            self._state['samples_count'] = samples_count
        if target_samples is not None:
            self._state['target_samples'] = target_samples
        if total_documents is not None:
            self._state['total_documents'] = total_documents
        if elapsed_seconds is not None:
            self._state['elapsed_seconds'] = elapsed_seconds
        if metadata is not None:
            self._state['metadata'].update(metadata)

        # Update timestamp
        self._state['last_updated'] = datetime.now().isoformat()

        # Write to temporary file first, then rename (atomic operation)
        temp_file = self.checkpoint_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_file.replace(self.checkpoint_file)

            self.logger.debug(f"Checkpoint saved: {self._state['samples_count']}/{self._state['target_samples']} samples")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise

    def clear_checkpoint(self) -> None:
        """Delete checkpoint file if it exists"""
        if self.has_checkpoint():
            # Backup before deleting
            backup_path = self.checkpoint_file.with_suffix('.json.backup')
            shutil.copy(self.checkpoint_file, backup_path)
            self.checkpoint_file.unlink()
            self.logger.info(f"Checkpoint cleared (backup saved to {backup_path})")
        self._state = self._init_state()

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self._state.copy()

    def mark_document_processed(self, document_name: str) -> None:
        """Mark a document as processed"""
        if document_name not in self._state['processed_documents']:
            self._state['processed_documents'].append(document_name)
            self.logger.info(f"Document marked as processed: {document_name}")

    def is_document_processed(self, document_name: str) -> bool:
        """Check if a document has been processed"""
        return document_name in self._state['processed_documents']

    def add_samples(self, samples: List[Dict]) -> None:
        """Add generated samples to state"""
        self._state['generated_samples'].extend(samples)
        self._state['samples_count'] = len(self._state['generated_samples'])
        self.logger.debug(f"Added {len(samples)} samples (total: {self._state['samples_count']})")

    def get_progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self._state['target_samples'] == 0:
            return 0.0
        return (self._state['samples_count'] / self._state['target_samples']) * 100

    def get_summary(self) -> str:
        """Get human-readable summary of current state"""
        state = self._state
        lines = [
            "Checkpoint Summary:",
            f"  Started: {state['started_at']}",
            f"  Last Updated: {state['last_updated']}",
            f"  Documents: {len(state['processed_documents'])}/{state['total_documents']}",
            f"  Samples: {state['samples_count']}/{state['target_samples']} ({self.get_progress_percentage():.1f}%)",
            f"  Current Phase: {state['current_phase'] or 'Not started'}",
            f"  Elapsed Time: {state['elapsed_seconds']:.1f}s"
        ]
        return "\n".join(lines)


class EvaluationCheckpointManager:
    """Manages checkpoint state for resumable evaluation with per-sample tracking"""

    def __init__(self, checkpoint_file: str = "ragas_checkpoint.json"):
        """
        Initialize evaluation checkpoint manager

        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.logger = get_logger(__name__)
        self._state: Dict[str, Any] = self._init_state()

    def _init_state(self) -> Dict[str, Any]:
        """Initialize empty evaluation state structure"""
        return {
            "version": "1.0",
            "type": "evaluation",
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "current_sample_index": -1,
            "total_samples": 0,
            "processed_samples": {},  # {sample_id: {status, metrics, error, attempts}}
            "failed_sample_ids": [],
            "successful_sample_ids": [],
            "completed_batches": 0,
            "elapsed_seconds": 0,
            "metadata": {}
        }

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint file exists"""
        return self.checkpoint_file.exists()

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint from file

        Returns:
            State dictionary
        """
        if not self.has_checkpoint():
            self.logger.warning(f"No checkpoint found at {self.checkpoint_file}")
            return self._init_state()

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                self._state = json.load(f)

            self.logger.info(f"Loaded evaluation checkpoint from {self.checkpoint_file}")
            self.logger.info(f"  Samples processed: {len(self._state['processed_samples'])}/{self._state['total_samples']}")
            self.logger.info(f"  Successful: {len(self._state['successful_sample_ids'])}")
            self.logger.info(f"  Failed: {len(self._state['failed_sample_ids'])}")
            self.logger.info(f"  Last updated: {self._state['last_updated']}")

            return self._state

        except json.JSONDecodeError as e:
            self.logger.error(f"Corrupted checkpoint file: {e}")
            backup_path = self.checkpoint_file.with_suffix('.json.corrupted')
            shutil.copy(self.checkpoint_file, backup_path)
            self.logger.warning(f"Backed up corrupted checkpoint to {backup_path}")
            raise

    def save_checkpoint(self) -> None:
        """Save checkpoint to file"""
        # Update timestamp
        self._state['last_updated'] = datetime.now().isoformat()

        # Write to temporary file first, then rename (atomic operation)
        temp_file = self.checkpoint_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_file.replace(self.checkpoint_file)

            self.logger.debug(
                f"Checkpoint saved: {len(self._state['processed_samples'])}/{self._state['total_samples']} samples "
                f"({len(self._state['successful_sample_ids'])} success, {len(self._state['failed_sample_ids'])} failed)"
            )

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise

    def clear_checkpoint(self) -> None:
        """Delete checkpoint file if it exists"""
        if self.has_checkpoint():
            backup_path = self.checkpoint_file.with_suffix('.json.backup')
            shutil.copy(self.checkpoint_file, backup_path)
            self.checkpoint_file.unlink()
            self.logger.info(f"Checkpoint cleared (backup saved to {backup_path})")
        self._state = self._init_state()

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self._state.copy()

    def save_sample_result(
        self,
        sample_id: int,
        status: str,
        metrics: Optional[Dict[str, float]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Save result for a single sample

        Args:
            sample_id: Sample ID/index
            status: Status ("success", "failed", "skipped")
            metrics: Dictionary of metric scores (for successful samples)
            error: Error message (for failed samples)
        """
        sample_id_str = str(sample_id)

        # Get existing attempts or initialize
        existing = self._state['processed_samples'].get(sample_id_str, {})
        attempts = existing.get('attempts', 0) + 1

        # Save sample result
        self._state['processed_samples'][sample_id_str] = {
            'status': status,
            'metrics': metrics or {},
            'error': error or '',
            'attempts': attempts,
            'updated_at': datetime.now().isoformat()
        }

        # Update sample index
        if sample_id > self._state['current_sample_index']:
            self._state['current_sample_index'] = sample_id

        # Update status lists
        if status == 'success' and sample_id not in self._state['successful_sample_ids']:
            self._state['successful_sample_ids'].append(sample_id)
            # Remove from failed list if it was there
            if sample_id in self._state['failed_sample_ids']:
                self._state['failed_sample_ids'].remove(sample_id)

        elif status == 'failed' and sample_id not in self._state['failed_sample_ids']:
            self._state['failed_sample_ids'].append(sample_id)
            # Remove from successful list if it was there
            if sample_id in self._state['successful_sample_ids']:
                self._state['successful_sample_ids'].remove(sample_id)

        self.logger.debug(f"Sample {sample_id} marked as {status} (attempt {attempts})")

    def is_sample_processed(self, sample_id: int) -> bool:
        """Check if a sample has been successfully processed"""
        return sample_id in self._state['successful_sample_ids']

    def is_sample_failed(self, sample_id: int) -> bool:
        """Check if a sample has failed"""
        return sample_id in self._state['failed_sample_ids']

    def get_sample_attempts(self, sample_id: int) -> int:
        """Get number of attempts for a sample"""
        sample_id_str = str(sample_id)
        return self._state['processed_samples'].get(sample_id_str, {}).get('attempts', 0)

    def get_failed_samples(self) -> List[int]:
        """Get list of failed sample IDs"""
        return self._state['failed_sample_ids'].copy()

    def get_successful_samples(self) -> List[int]:
        """Get list of successful sample IDs"""
        return self._state['successful_sample_ids'].copy()

    def get_unprocessed_sample_ids(self, total_samples: int) -> List[int]:
        """
        Get list of sample IDs that haven't been successfully processed

        Args:
            total_samples: Total number of samples in dataset

        Returns:
            List of unprocessed sample IDs
        """
        return [
            i for i in range(total_samples)
            if not self.is_sample_processed(i)
        ]

    def mark_batch_completed(self) -> None:
        """Mark a batch as completed"""
        self._state['completed_batches'] += 1

    def set_total_samples(self, total: int) -> None:
        """Set total number of samples"""
        self._state['total_samples'] = total

    def set_elapsed_seconds(self, seconds: float) -> None:
        """Set elapsed time"""
        self._state['elapsed_seconds'] = seconds

    def get_progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self._state['total_samples'] == 0:
            return 0.0
        return (len(self._state['processed_samples']) / self._state['total_samples']) * 100

    def get_summary(self) -> str:
        """Get human-readable summary of current state"""
        state = self._state
        total_processed = len(state['processed_samples'])
        successful = len(state['successful_sample_ids'])
        failed = len(state['failed_sample_ids'])
        progress_pct = self.get_progress_percentage()

        lines = [
            "Evaluation Checkpoint Summary:",
            f"  Started: {state['started_at']}",
            f"  Last Updated: {state['last_updated']}",
            f"  Progress: {total_processed}/{state['total_samples']} ({progress_pct:.1f}%)",
            f"  Successful: {successful}",
            f"  Failed: {failed}",
            f"  Batches Completed: {state['completed_batches']}",
            f"  Elapsed Time: {state['elapsed_seconds']:.1f}s"
        ]
        return "\n".join(lines)
