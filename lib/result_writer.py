"""
Incremental result writing for testset generation and evaluation

Enables saving results as they're generated to prevent data loss.
"""

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .logger import get_logger


class IncrementalCSVWriter:
    """Writes test samples incrementally to CSV file"""

    def __init__(
        self,
        output_file: str = "testset_partial.csv",
        final_file: str = "testset_final.csv",
        backup_enabled: bool = True
    ):
        """
        Initialize incremental CSV writer

        Args:
            output_file: Path to partial results file
            final_file: Path to final consolidated file
            backup_enabled: Whether to backup existing files
        """
        self.output_file = Path(output_file)
        self.final_file = Path(final_file)
        self.backup_enabled = backup_enabled
        self.logger = get_logger(__name__)
        self._samples_written = 0
        self._headers_written = False

    def write_sample(self, sample: Dict[str, Any]) -> None:
        """
        Write a single sample to the partial CSV file

        Args:
            sample: Sample dictionary to write
        """
        try:
            # Determine if file exists and has headers
            file_exists = self.output_file.exists()

            # Open in append mode
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                # Get fieldnames from sample keys
                fieldnames = list(sample.keys())

                writer = csv.DictWriter(f, fieldnames=fieldnames)

                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()
                    self._headers_written = True

                # Write the sample
                writer.writerow(sample)

            self._samples_written += 1
            self.logger.debug(f"Sample written to {self.output_file} (total: {self._samples_written})")

        except Exception as e:
            self.logger.error(f"Failed to write sample: {e}")
            raise

    def write_samples(self, samples: List[Dict[str, Any]]) -> None:
        """
        Write multiple samples to the partial CSV file

        Args:
            samples: List of sample dictionaries to write
        """
        for sample in samples:
            self.write_sample(sample)

    def write_dataframe(self, df: pd.DataFrame) -> None:
        """
        Write entire dataframe to partial CSV file

        Args:
            df: DataFrame to write
        """
        try:
            # Backup existing file if needed
            if self.backup_enabled and self.output_file.exists():
                backup_path = self.output_file.with_suffix('.csv.backup')
                shutil.copy(self.output_file, backup_path)
                self.logger.debug(f"Backed up existing file to {backup_path}")

            # Write dataframe
            df.to_csv(self.output_file, index=False, encoding='utf-8')
            self._samples_written = len(df)
            self._headers_written = True

            self.logger.info(f"Wrote {len(df)} samples to {self.output_file}")

        except Exception as e:
            self.logger.error(f"Failed to write dataframe: {e}")
            raise

    def read_partial_results(self) -> Optional[pd.DataFrame]:
        """
        Read partial results from file

        Returns:
            DataFrame of partial results, or None if file doesn't exist
        """
        if not self.output_file.exists():
            self.logger.debug("No partial results file found")
            return None

        try:
            df = pd.read_csv(self.output_file, encoding='utf-8')
            self.logger.info(f"Loaded {len(df)} partial results from {self.output_file}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to read partial results: {e}")
            return None

    def finalize(self, additional_samples: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Finalize results by consolidating to final file

        Args:
            additional_samples: Optional additional samples to add before finalizing

        Returns:
            DataFrame of final results
        """
        try:
            # Write any additional samples
            if additional_samples:
                self.write_samples(additional_samples)

            # Read all partial results
            if not self.output_file.exists():
                self.logger.warning("No partial results to finalize")
                return pd.DataFrame()

            df = pd.read_csv(self.output_file, encoding='utf-8')

            # Backup existing final file if needed
            if self.backup_enabled and self.final_file.exists():
                backup_path = self.final_file.with_suffix('.csv.backup')
                shutil.copy(self.final_file, backup_path)
                self.logger.debug(f"Backed up existing final file to {backup_path}")

            # Write to final file
            df.to_csv(self.final_file, index=False, encoding='utf-8')

            self.logger.info(f"Finalized {len(df)} samples to {self.final_file}")

            return df

        except Exception as e:
            self.logger.error(f"Failed to finalize results: {e}")
            raise

    def clear_partial(self) -> None:
        """Clear partial results file"""
        if self.output_file.exists():
            # Backup before clearing
            if self.backup_enabled:
                backup_path = self.output_file.with_suffix('.csv.cleared')
                shutil.copy(self.output_file, backup_path)
                self.logger.debug(f"Backed up partial file to {backup_path}")

            self.output_file.unlink()
            self.logger.info("Partial results cleared")

        self._samples_written = 0
        self._headers_written = False

    def get_sample_count(self) -> int:
        """Get number of samples written"""
        return self._samples_written

    def has_partial_results(self) -> bool:
        """Check if partial results file exists"""
        return self.output_file.exists()


class IncrementalJSONLWriter:
    """Writes evaluation results incrementally to JSONL file"""

    def __init__(
        self,
        output_file: str,
        backup_enabled: bool = True
    ):
        """
        Initialize incremental JSONL writer

        Args:
            output_file: Path to JSONL file
            backup_enabled: Whether to backup existing files
        """
        self.output_file = Path(output_file)
        self.backup_enabled = backup_enabled
        self.logger = get_logger(__name__)
        self._samples_written = 0

    def write_sample(self, sample: Dict[str, Any]) -> None:
        """
        Write a single sample to JSONL file

        Args:
            sample: Sample dictionary to write
        """
        try:
            # Open in append mode
            with open(self.output_file, 'a', encoding='utf-8') as f:
                json_line = json.dumps(sample, ensure_ascii=False)
                f.write(json_line + '\n')

            self._samples_written += 1
            self.logger.debug(f"Sample written to {self.output_file} (total: {self._samples_written})")

        except Exception as e:
            self.logger.error(f"Failed to write sample: {e}")
            raise

    def write_samples(self, samples: List[Dict[str, Any]]) -> None:
        """
        Write multiple samples to JSONL file

        Args:
            samples: List of sample dictionaries to write
        """
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                for sample in samples:
                    json_line = json.dumps(sample, ensure_ascii=False)
                    f.write(json_line + '\n')
                    self._samples_written += 1

            self.logger.debug(f"Wrote {len(samples)} samples to {self.output_file}")

        except Exception as e:
            self.logger.error(f"Failed to write samples: {e}")
            raise

    def read_existing(self) -> List[Dict[str, Any]]:
        """
        Read existing results from JSONL file

        Returns:
            List of sample dictionaries, or empty list if file doesn't exist
        """
        if not self.output_file.exists():
            self.logger.debug("No existing JSONL file found")
            return []

        try:
            samples = []
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))

            self._samples_written = len(samples)
            self.logger.info(f"Loaded {len(samples)} existing results from {self.output_file}")
            return samples

        except Exception as e:
            self.logger.error(f"Failed to read existing results: {e}")
            return []

    def clear(self) -> None:
        """Clear JSONL file"""
        if self.output_file.exists():
            # Backup before clearing
            if self.backup_enabled:
                backup_path = self.output_file.with_suffix('.jsonl.backup')
                shutil.copy(self.output_file, backup_path)
                self.logger.debug(f"Backed up file to {backup_path}")

            self.output_file.unlink()
            self.logger.info("JSONL file cleared")

        self._samples_written = 0

    def get_sample_count(self) -> int:
        """Get number of samples written"""
        return self._samples_written

    def exists(self) -> bool:
        """Check if JSONL file exists"""
        return self.output_file.exists()
