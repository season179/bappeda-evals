"""
Data Transformer for Ragas Evaluation

Transforms RAG executor results into Ragas Dataset format.
"""

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset

from .logger import get_logger


class RagasDataTransformer:
    """Transforms executor results to Ragas Dataset format"""

    def __init__(self):
        """Initialize the transformer"""
        self.logger = get_logger(__name__)

    def _parse_string_to_object(self, value: str) -> Any:
        """
        Parse a string value as JSON or Python literal

        Args:
            value: String to parse

        Returns:
            Parsed object, or None if parsing fails
        """
        # Try JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Try Python literal eval
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        return None

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load results from JSONL file

        Args:
            file_path: Path to JSONL file

        Returns:
            List of result dictionaries
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")

        records = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse JSON at line {line_num}: {e}")
                            continue

            self.logger.info(f"Loaded {len(records)} records from {file_path}")
            return records

        except Exception as e:
            self.logger.error(f"Failed to load JSONL file: {e}")
            raise

    def parse_reference_contexts(self, reference_contexts: Any) -> List[str]:
        """
        Parse reference_contexts which may be a string, list, or other format

        Args:
            reference_contexts: Reference contexts in various formats

        Returns:
            List of context strings
        """
        # If already a list, return it
        if isinstance(reference_contexts, list):
            return reference_contexts

        # If string, try to parse as JSON or Python literal
        if isinstance(reference_contexts, str):
            parsed = self._parse_string_to_object(reference_contexts)
            if parsed is not None and isinstance(parsed, list):
                return parsed

            # If parsing fails or result is not a list, treat as single context
            self.logger.warning(f"Could not parse reference_contexts as list, treating as single string")
            return [reference_contexts]

        # If other type, convert to string and wrap in list
        self.logger.warning(f"Unexpected reference_contexts type: {type(reference_contexts)}")
        return [str(reference_contexts)]

    def transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single executor result to Ragas format

        Args:
            record: Executor result record

        Returns:
            Ragas-formatted record
        """
        # Parse reference_contexts if needed
        reference_contexts = self.parse_reference_contexts(
            record.get('reference_contexts', [])
        )

        # Get retrieved contexts (may be empty for failed queries)
        retrieved_contexts = record.get('actual_contexts', [])
        if not isinstance(retrieved_contexts, list):
            retrieved_contexts = []

        # Create Ragas format
        ragas_record = {
            'user_input': record.get('user_input', ''),
            'retrieved_contexts': retrieved_contexts,
            'response': record.get('actual_answer', ''),
            'reference': record.get('reference', ''),
            'reference_contexts': reference_contexts
        }

        # Add metadata for tracking
        ragas_record['_metadata'] = {
            'query_id': record.get('query_id'),
            'status': record.get('status', 'UNKNOWN'),
            'api_latency_ms': record.get('api_latency_ms', 0),
            'error': record.get('error', ''),
            'has_contexts': len(retrieved_contexts) > 0
        }

        return ragas_record

    def transform_to_ragas_dataset(
        self,
        records: List[Dict[str, Any]],
        include_failed: bool = True
    ) -> Dataset:
        """
        Transform list of executor results to Ragas Dataset

        Args:
            records: List of executor result records
            include_failed: Whether to include failed queries (with zero contexts)

        Returns:
            Ragas Dataset
        """
        transformed_records = []

        for record in records:
            # Check if query failed
            status = record.get('status', 'UNKNOWN')
            has_contexts = len(record.get('actual_contexts', [])) > 0

            # Skip failed queries if not including them
            if not include_failed and (status == 'FAILED' or not has_contexts):
                self.logger.debug(
                    f"Skipping failed query {record.get('query_id')}: "
                    f"status={status}, has_contexts={has_contexts}"
                )
                continue

            # Transform record
            try:
                ragas_record = self.transform_record(record)
                transformed_records.append(ragas_record)
            except Exception as e:
                self.logger.error(
                    f"Failed to transform record {record.get('query_id')}: {e}"
                )
                continue

        self.logger.info(
            f"Transformed {len(transformed_records)} records "
            f"(out of {len(records)} total)"
        )

        # Create HuggingFace Dataset
        dataset = Dataset.from_list(transformed_records)
        return dataset

    def load_and_transform(
        self,
        jsonl_path: str,
        include_failed: bool = True
    ) -> Dataset:
        """
        Load JSONL file and transform to Ragas Dataset

        Args:
            jsonl_path: Path to executor results JSONL file
            include_failed: Whether to include failed queries

        Returns:
            Ragas Dataset
        """
        # Load records
        records = self.load_jsonl(jsonl_path)

        # Transform to Ragas format
        dataset = self.transform_to_ragas_dataset(records, include_failed)

        return dataset

    def validate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Validate transformed dataset and return statistics

        Args:
            dataset: Ragas Dataset

        Returns:
            Dictionary with validation statistics
        """
        total_records = len(dataset)

        # Count records with/without contexts
        with_contexts = sum(
            1 for record in dataset
            if len(record['retrieved_contexts']) > 0
        )
        without_contexts = total_records - with_contexts

        # Count by status
        status_counts = {}
        for record in dataset:
            status = record['_metadata']['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        # Average context lengths
        avg_retrieved_contexts = sum(
            len(record['retrieved_contexts']) for record in dataset
        ) / total_records if total_records > 0 else 0

        avg_reference_contexts = sum(
            len(record['reference_contexts']) for record in dataset
        ) / total_records if total_records > 0 else 0

        validation_stats = {
            'total_records': total_records,
            'with_contexts': with_contexts,
            'without_contexts': without_contexts,
            'status_counts': status_counts,
            'avg_retrieved_contexts': avg_retrieved_contexts,
            'avg_reference_contexts': avg_reference_contexts
        }

        self.logger.info("Dataset validation statistics:")
        self.logger.info(f"  Total records: {total_records}")
        self.logger.info(f"  With contexts: {with_contexts}")
        self.logger.info(f"  Without contexts: {without_contexts}")
        self.logger.info(f"  Status counts: {status_counts}")
        self.logger.info(f"  Avg retrieved contexts: {avg_retrieved_contexts:.2f}")
        self.logger.info(f"  Avg reference contexts: {avg_reference_contexts:.2f}")

        return validation_stats
