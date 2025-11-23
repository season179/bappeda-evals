#!/usr/bin/env python3
"""
Standalone Factual Correctness Evaluation Script

Evaluates RAG factual correctness using Ragas FactualCorrectness metric.
Optimized to prevent timeout errors with:
- Single metric focus (no context extraction)
- Low atomicity/coverage (fewer claims)
- Individual sample timeouts
- Length correlation tracking

All dependencies inlined - no external lib/ or config.yaml required.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.outputs import Generation, LLMResult
from langchain_openai import ChatOpenAI
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import FactualCorrectness

# Load environment variables
load_dotenv()


# ============================================================================
# LOGGING UTILITIES (inlined from lib/logger.py)
# ============================================================================

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
        record_copy = logging.makeLogRecord(record.__dict__)
        if record_copy.levelname in self.COLORS:
            record_copy.levelname = f"{self.COLORS[record_copy.levelname]}{record_copy.levelname}{self.RESET}"
        return super().format(record_copy)


def setup_logger(
    name: str = "factual_correctness",
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """Setup structured logging with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)

    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_log_file = log_path / f"factual_correctness_{timestamp}.log"
    file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)

    # Error log
    error_log_file = log_path / "errors.log"
    error_handler = logging.FileHandler(error_log_file, mode='a', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(error_handler)

    logger.info(f"Logging initialized - Main log: {main_log_file}")
    return logger


# ============================================================================
# OPENROUTER CHAT WRAPPER (inlined from lib/openrouter_chat.py)
# ============================================================================

class OpenRouterChatOpenAI(ChatOpenAI):
    """
    Custom ChatOpenAI wrapper for OpenRouter compatibility.

    OpenRouter doesn't support the 'n' parameter for multiple completions.
    This wrapper overrides agenerate_prompt to manually make separate API calls.
    """

    def __init__(self, **kwargs):
        """Initialize with n=1 to ensure single generation per request."""
        super().__init__(**kwargs)
        self.n = 1

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get default parameters, excluding 'n' for OpenRouter compatibility."""
        params = super()._default_params
        if 'n' in params:
            params.pop('n')
        return params

    async def agenerate_prompt(
        self,
        prompts: List[Any],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Process each prompt separately to ensure OpenRouter returns one result per prompt."""
        all_generations: List[List[Generation]] = []
        combined_llm_output: Dict[str, Any] = {}

        for prompt in prompts:
            result = await super().agenerate_prompt(
                [prompt],
                stop=stop,
                callbacks=callbacks,
                **kwargs
            )

            if result.generations:
                all_generations.append(result.generations[0])
            else:
                all_generations.append([])

            # Accumulate token usage
            if result.llm_output:
                for key, value in result.llm_output.items():
                    if key in combined_llm_output:
                        if isinstance(value, (int, float)) and isinstance(combined_llm_output[key], (int, float)):
                            combined_llm_output[key] += value
                    else:
                        combined_llm_output[key] = value

        return LLMResult(
            generations=all_generations,
            llm_output=combined_llm_output if combined_llm_output else None,
        )


# ============================================================================
# JSONL WRITER (inlined from lib/result_writer.py)
# ============================================================================

class IncrementalJSONLWriter:
    """Writes evaluation results incrementally to JSONL file"""

    def __init__(self, output_file: str, backup_enabled: bool = True):
        self.output_file = Path(output_file)
        self.backup_enabled = backup_enabled
        self.logger = logging.getLogger(__name__)
        self._samples_written = 0

    def write_sample(self, sample: Dict[str, Any]) -> None:
        """Write a single sample to JSONL file"""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                json_line = json.dumps(sample, ensure_ascii=False)
                f.write(json_line + '\n')

            self._samples_written += 1
            self.logger.debug(f"Sample written to {self.output_file} (total: {self._samples_written})")

        except Exception as e:
            self.logger.error(f"Failed to write sample: {e}")
            raise

    def clear(self) -> None:
        """Clear JSONL file"""
        if self.output_file.exists():
            if self.backup_enabled:
                backup_path = self.output_file.with_suffix('.jsonl.backup')
                shutil.copy(self.output_file, backup_path)
                self.logger.debug(f"Backed up file to {backup_path}")

            self.output_file.unlink()
            self.logger.info("JSONL file cleared")

        self._samples_written = 0

    def exists(self) -> bool:
        """Check if JSONL file exists"""
        return self.output_file.exists()


# ============================================================================
# CHECKPOINT MANAGER (inlined from lib/state_manager.py)
# ============================================================================

class EvaluationCheckpointManager:
    """Manages checkpoint state for resumable evaluation with per-sample tracking"""

    def __init__(self, checkpoint_file: str = "factual_correctness_checkpoint.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.logger = logging.getLogger(__name__)
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
            "processed_samples": {},
            "failed_sample_ids": [],
            "successful_sample_ids": [],
            "elapsed_seconds": 0,
            "metadata": {}
        }

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint file exists"""
        return self.checkpoint_file.exists()

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint from file"""
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

            return self._state

        except json.JSONDecodeError as e:
            self.logger.error(f"Corrupted checkpoint file: {e}")
            backup_path = self.checkpoint_file.with_suffix('.json.corrupted')
            shutil.copy(self.checkpoint_file, backup_path)
            raise

    def save_checkpoint(self) -> None:
        """Save checkpoint to file"""
        self._state['last_updated'] = datetime.now().isoformat()

        temp_file = self.checkpoint_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)

            temp_file.replace(self.checkpoint_file)

            self.logger.debug(
                f"Checkpoint saved: {len(self._state['processed_samples'])}/{self._state['total_samples']} samples"
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

    def save_sample_result(
        self,
        sample_id: int,
        status: str,
        metrics: Optional[Dict[str, float]] = None,
        error: Optional[str] = None
    ) -> None:
        """Save result for a single sample"""
        sample_id_str = str(sample_id)

        existing = self._state['processed_samples'].get(sample_id_str, {})
        attempts = existing.get('attempts', 0) + 1

        self._state['processed_samples'][sample_id_str] = {
            'status': status,
            'metrics': metrics or {},
            'error': error or '',
            'attempts': attempts,
            'updated_at': datetime.now().isoformat()
        }

        if sample_id > self._state['current_sample_index']:
            self._state['current_sample_index'] = sample_id

        if status == 'success' and sample_id not in self._state['successful_sample_ids']:
            self._state['successful_sample_ids'].append(sample_id)
            if sample_id in self._state['failed_sample_ids']:
                self._state['failed_sample_ids'].remove(sample_id)

        elif status == 'failed' and sample_id not in self._state['failed_sample_ids']:
            self._state['failed_sample_ids'].append(sample_id)
            if sample_id in self._state['successful_sample_ids']:
                self._state['successful_sample_ids'].remove(sample_id)

    def get_failed_samples(self) -> List[int]:
        """Get list of failed sample IDs"""
        return self._state['failed_sample_ids'].copy()

    def get_successful_samples(self) -> List[int]:
        """Get list of successful sample IDs"""
        return self._state['successful_sample_ids'].copy()

    def set_total_samples(self, total: int) -> None:
        """Set total number of samples"""
        self._state['total_samples'] = total

    def set_elapsed_seconds(self, seconds: float) -> None:
        """Set elapsed time"""
        self._state['elapsed_seconds'] = seconds


# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG factual correctness using Ragas (standalone version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s --input results/rag_execution_250122_143022.jsonl

  # Faster evaluation (precision-only mode)
  %(prog)s --input results/rag_execution.jsonl --mode precision

  # Resume from checkpoint
  %(prog)s --input results/rag_execution.jsonl --resume

  # Retry only failed/timeout samples
  %(prog)s --input results/rag_execution.jsonl --resume --retry-failed
"""
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file from RAG executor (required)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory (default: ./results)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (default: from OPENROUTER_API_KEY env var)"
    )

    parser.add_argument(
        "--api-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="OpenRouter API base URL (default: https://openrouter.ai/api/v1)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5.1-codex-mini",
        help="LLM model to use (default: openai/gpt-5.1-codex-mini)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["F1", "precision", "recall"],
        default="F1",
        help="Scoring mode: F1 (both directions), precision (response→reference only), "
             "or recall (reference→response only). Default: F1"
    )

    parser.add_argument(
        "--atomicity",
        type=str,
        choices=["low", "high"],
        default="low",
        help="Claim granularity: low (fewer claims, faster) or high (more granular). Default: low"
    )

    parser.add_argument(
        "--coverage",
        type=str,
        choices=["low", "high"],
        default="low",
        help="Claim coverage: low (main points only) or high (all details). Default: low"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per sample in seconds (default: 600)"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=15,
        help="Maximum API retries for rate limits (default: 15)"
    )

    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=600,
        help="LLM request timeout in seconds (default: 600)"
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum retry attempts per sample (default: 3)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only failed/timeout samples from previous run (requires --resume)"
    )

    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include queries without responses (default: skip them)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose console logging"
    )

    return parser.parse_args()


def load_executor_results(jsonl_path: str, logger) -> List[dict]:
    """Load executor results from JSONL file"""
    results = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                result = json.loads(line)
                results.append(result)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(results)} results from {jsonl_path}")
    return results


def transform_to_dataset(
    records: List[dict],
    include_failed: bool,
    logger
) -> EvaluationDataset:
    """Transform executor results to Ragas EvaluationDataset format"""
    samples = []
    skipped_count = 0

    for record in records:
        user_input = record.get('user_input', '').strip()
        response = record.get('actual_answer', '').strip()
        reference = record.get('reference', '').strip()

        # Validate required fields
        if not user_input:
            logger.warning(f"Skipping query_id {record.get('query_id')}: missing user_input")
            skipped_count += 1
            continue

        if not reference:
            logger.warning(f"Skipping query_id {record.get('query_id')}: missing reference")
            skipped_count += 1
            continue

        # Handle failed queries (no response)
        if not response:
            if include_failed:
                logger.warning(f"Query {record.get('query_id')} has no response - will get zero score")
                response = ""  # Empty response will get zero score
            else:
                logger.warning(f"Skipping query_id {record.get('query_id')}: no response (use --include-failed to evaluate)")
                skipped_count += 1
                continue

        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            reference=reference
        )

        samples.append(sample)

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} invalid samples")

    dataset = EvaluationDataset(samples=samples)
    return dataset


def initialize_metric(
    mode: str,
    atomicity: str,
    coverage: str,
    llm,
    logger
) -> FactualCorrectness:
    """Initialize FactualCorrectness metric with specified parameters"""
    logger.info(f"Initializing FactualCorrectness metric:")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Atomicity: {atomicity}")
    logger.info(f"  Coverage: {coverage}")

    metric = FactualCorrectness(
        mode=mode,
        atomicity=atomicity,
        coverage=coverage,
        llm=llm
    )

    return metric


async def evaluate_single_sample(
    sample: SingleTurnSample,
    metric: FactualCorrectness,
    timeout_seconds: int,
    logger
) -> Dict:
    """Evaluate a single sample with timeout"""
    start_time = time.time()

    try:
        # Run evaluation with timeout
        result = await asyncio.wait_for(
            metric.single_turn_ascore(sample),
            timeout=timeout_seconds
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Extract scores
        return {
            "factual_correctness": float(result),
            "response_length": len(sample.response),
            "reference_length": len(sample.reference),
            "processing_time_ms": elapsed_ms,
            "status": "SUCCESS",
            "error": ""
        }

    except asyncio.TimeoutError:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Timeout after {elapsed_ms}ms (limit: {timeout_seconds}s)")

        return {
            "factual_correctness": None,
            "response_length": len(sample.response),
            "reference_length": len(sample.reference),
            "processing_time_ms": elapsed_ms,
            "status": "TIMEOUT",
            "error": f"Evaluation exceeded {timeout_seconds}s timeout"
        }

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Evaluation failed: {e}")

        return {
            "factual_correctness": None,
            "response_length": len(sample.response),
            "reference_length": len(sample.reference),
            "processing_time_ms": elapsed_ms,
            "status": "FAILED",
            "error": str(e)
        }


def analyze_timeout_correlation(results: List[dict], logger) -> Dict:
    """Analyze correlation between text length and timeouts"""
    if not results:
        return {}

    df = pd.DataFrame(results)

    # Calculate combined length
    df['combined_length'] = df['response_length'] + df['reference_length']

    # Separate by status
    success_df = df[df['status'] == 'SUCCESS']
    timeout_df = df[df['status'] == 'TIMEOUT']
    failed_df = df[df['status'] == 'FAILED']

    stats = {
        'total_samples': len(df),
        'success_count': len(success_df),
        'timeout_count': len(timeout_df),
        'failed_count': len(failed_df),
        'timeout_rate': len(timeout_df) / len(df) if len(df) > 0 else 0
    }

    if len(success_df) > 0:
        stats['success_avg_length'] = success_df['combined_length'].mean()
        stats['success_avg_time_ms'] = success_df['processing_time_ms'].mean()
        stats['success_max_time_ms'] = success_df['processing_time_ms'].max()

    if len(timeout_df) > 0:
        stats['timeout_avg_length'] = timeout_df['combined_length'].mean()
        stats['timeout_min_length'] = timeout_df['combined_length'].min()
        stats['timeout_max_length'] = timeout_df['combined_length'].max()

    # Find length threshold where timeouts become common
    if len(timeout_df) > 0 and len(success_df) > 0:
        threshold = timeout_df['combined_length'].min()
        stats['suggested_length_limit'] = int(threshold * 0.8)  # 20% safety margin
        logger.info(f"Timeout threshold detected: {threshold} chars")
        logger.info(f"Suggested length limit: {stats['suggested_length_limit']} chars")

    return stats


def generate_report(
    results: List[dict],
    correlation_stats: Dict,
    config: Dict,
    output_path: str,
    logger
):
    """Generate markdown report with scores and timeout analysis"""
    # Calculate score statistics (only successful samples)
    success_results = [r for r in results if r['status'] == 'SUCCESS' and r['factual_correctness'] is not None]

    if success_results:
        scores = [r['factual_correctness'] for r in success_results]
        score_stats = {
            'mean': sum(scores) / len(scores),
            'median': sorted(scores)[len(scores) // 2],
            'min': min(scores),
            'max': max(scores),
            'std': (sum((x - sum(scores) / len(scores)) ** 2 for x in scores) / len(scores)) ** 0.5
        }
    else:
        score_stats = None

    # Generate markdown
    lines = [
        "# Factual Correctness Evaluation Report",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- **LLM Model**: {config.get('llm_model', 'N/A')}",
        f"- **Mode**: {config.get('mode', 'N/A')}",
        f"- **Atomicity**: {config.get('atomicity', 'N/A')}",
        f"- **Coverage**: {config.get('coverage', 'N/A')}",
        f"- **Timeout**: {config.get('timeout_seconds', 'N/A')}s per sample",
        "",
        "## Overall Results",
        "",
        f"- **Total Samples**: {correlation_stats.get('total_samples', 0)}",
        f"- **Successful**: {correlation_stats.get('success_count', 0)} ({correlation_stats.get('success_count', 0) / max(correlation_stats.get('total_samples', 1), 1) * 100:.1f}%)",
        f"- **Timeout**: {correlation_stats.get('timeout_count', 0)} ({correlation_stats.get('timeout_rate', 0) * 100:.1f}%)",
        f"- **Failed**: {correlation_stats.get('failed_count', 0)}",
        "",
    ]

    if score_stats:
        lines.extend([
            "## Factual Correctness Scores",
            "",
            f"- **Mean**: {score_stats['mean']:.4f}",
            f"- **Median**: {score_stats['median']:.4f}",
            f"- **Std Dev**: {score_stats['std']:.4f}",
            f"- **Min**: {score_stats['min']:.4f}",
            f"- **Max**: {score_stats['max']:.4f}",
            "",
        ])
    else:
        lines.extend([
            "## Factual Correctness Scores",
            "",
            "_No successful evaluations to report_",
            "",
        ])

    # Processing time analysis
    if correlation_stats.get('success_count', 0) > 0:
        lines.extend([
            "## Processing Time Analysis",
            "",
            f"- **Average Time (Success)**: {correlation_stats.get('success_avg_time_ms', 0) / 1000:.1f}s",
            f"- **Max Time (Success)**: {correlation_stats.get('success_max_time_ms', 0) / 1000:.1f}s",
            f"- **Average Length (Success)**: {correlation_stats.get('success_avg_length', 0):.0f} chars",
            "",
        ])

    # Timeout correlation analysis
    if correlation_stats.get('timeout_count', 0) > 0:
        lines.extend([
            "## Timeout Correlation Analysis",
            "",
            f"- **Timeout Rate**: {correlation_stats.get('timeout_rate', 0) * 100:.1f}%",
            f"- **Average Length (Timeout)**: {correlation_stats.get('timeout_avg_length', 0):.0f} chars",
            f"- **Min Length (Timeout)**: {correlation_stats.get('timeout_min_length', 0):.0f} chars",
            f"- **Max Length (Timeout)**: {correlation_stats.get('timeout_max_length', 0):.0f} chars",
            "",
        ])

        if 'suggested_length_limit' in correlation_stats:
            lines.extend([
                "### Recommendations",
                "",
                f"**Suggested Length Limit**: {correlation_stats['suggested_length_limit']} characters (combined response + reference)",
                "",
                "Samples exceeding this limit have high timeout probability. Consider:",
                "- Truncating long responses/references before evaluation",
                "- Increasing timeout duration for long samples",
                "- Using lower atomicity/coverage settings",
                "",
            ])

    # Write report
    report_content = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"Report saved to {output_path}")


def main():
    """Main execution function"""
    args = parse_args()

    # Setup logging
    console_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(
        name="factual_correctness",
        log_dir="logs",
        console_level=console_level,
        file_level=logging.DEBUG
    )

    logger.info("=" * 80)
    logger.info("Factual Correctness Evaluation (Standalone)")
    logger.info("=" * 80)

    # Configuration
    input_file = args.input
    output_dir = args.output_dir
    detailed_results_file = "factual_correctness_detailed.jsonl"
    report_file = "factual_correctness_report.md"
    checkpoint_file = "factual_correctness_checkpoint.json"

    detailed_results_path = Path(output_dir) / detailed_results_file
    report_path = Path(output_dir) / report_file

    # Get API key
    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found. Use --api-key or set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    logger.info(f"Configuration:")
    logger.info(f"  Input file: {input_file}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Detailed results: {detailed_results_path}")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  LLM Model: {args.model}")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Atomicity: {args.atomicity}")
    logger.info(f"  Coverage: {args.coverage}")
    logger.info(f"  Timeout per sample: {args.timeout}s")
    logger.info(f"  Max attempts per sample: {args.max_attempts}")
    logger.info(f"  Include failed queries: {args.include_failed}")

    # Pre-flight checks
    logger.info("\n" + "=" * 80)
    logger.info("PRE-FLIGHT CHECKS")
    logger.info("=" * 80)

    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    logger.info(f"✓ Input file exists: {input_file}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory ready: {output_dir}")

    # Load and transform data
    logger.info("\n" + "=" * 80)
    logger.info("LOADING AND TRANSFORMING DATA")
    logger.info("=" * 80)

    try:
        executor_results = load_executor_results(input_file, logger)
        dataset = transform_to_dataset(executor_results, args.include_failed, logger)
        logger.info(f"✓ Transformed {len(dataset)} samples for evaluation")

    except Exception as e:
        logger.error(f"Failed to load and transform data: {e}")
        sys.exit(1)

    # Initialize LLM
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING LLM")
    logger.info("=" * 80)

    try:
        llm = OpenRouterChatOpenAI(
            api_key=api_key,
            base_url=args.api_base_url,
            model=args.model,
            temperature=args.temperature,
            timeout=args.llm_timeout,
            max_retries=args.max_retries
        )
        logger.info(f"✓ LLM initialized: {args.model}")

    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        sys.exit(1)

    # Initialize metric
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING METRIC")
    logger.info("=" * 80)

    try:
        metric = initialize_metric(args.mode, args.atomicity, args.coverage, llm, logger)
        logger.info("✓ FactualCorrectness metric initialized")

    except Exception as e:
        logger.error(f"Failed to initialize metric: {e}")
        sys.exit(1)

    # Initialize checkpoint manager
    checkpoint_manager = None
    skip_sample_ids = []
    retry_sample_ids = []

    if args.resume:
        checkpoint_manager = EvaluationCheckpointManager(checkpoint_file)
        checkpoint_data = checkpoint_manager.load_checkpoint()
        checkpoint_manager.set_total_samples(len(dataset))

        if args.retry_failed:
            retry_sample_ids = checkpoint_manager.get_failed_samples()
            logger.info(f"✓ Retry mode: {len(retry_sample_ids)} failed samples to retry")
        else:
            skip_sample_ids = checkpoint_manager.get_successful_samples()
            logger.info(f"✓ Resume mode: {len(skip_sample_ids)} samples already completed")
    else:
        checkpoint_manager = EvaluationCheckpointManager(checkpoint_file)
        checkpoint_manager.clear_checkpoint()
        checkpoint_manager.set_total_samples(len(dataset))

    # Initialize writer
    writer = IncrementalJSONLWriter(
        output_file=str(detailed_results_path),
        backup_enabled=True
    )

    if not args.resume and writer.exists():
        writer.clear()

    # Run evaluation
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING EVALUATION")
    logger.info("=" * 80)

    start_time = time.time()
    all_results = []
    successful_count = 0
    timeout_count = 0
    failed_count = 0

    try:
        for idx, sample in enumerate(dataset):
            sample_id = idx

            # Skip if already processed (resume mode)
            if skip_sample_ids and sample_id in skip_sample_ids:
                continue

            # Skip if not in retry list (retry mode)
            if retry_sample_ids and sample_id not in retry_sample_ids:
                continue

            logger.info(f"\nEvaluating sample {sample_id + 1}/{len(dataset)}")
            logger.info(f"  Response length: {len(sample.response)} chars")
            logger.info(f"  Reference length: {len(sample.reference)} chars")

            # Evaluate with retries
            attempt = 0
            eval_result = None

            while attempt < args.max_attempts:
                attempt += 1

                if attempt > 1:
                    logger.info(f"  Retry attempt {attempt}/{args.max_attempts}")

                eval_result = asyncio.run(
                    evaluate_single_sample(sample, metric, args.timeout, logger)
                )

                if eval_result['status'] == 'SUCCESS':
                    break
                elif eval_result['status'] == 'TIMEOUT':
                    logger.warning(f"  Timeout on attempt {attempt}")
                    if attempt < args.max_attempts:
                        logger.info(f"  Waiting 5s before retry...")
                        time.sleep(5)
                else:
                    logger.error(f"  Error on attempt {attempt}: {eval_result['error']}")
                    break

            # Build result record
            result_record = {
                "query_id": sample_id,
                "user_input": sample.user_input,
                "response": sample.response,
                "reference": sample.reference,
                **eval_result,
                "num_attempts": attempt,
                "evaluation_config": {
                    "mode": args.mode,
                    "atomicity": args.atomicity,
                    "coverage": args.coverage
                }
            }

            # Update counters
            if eval_result['status'] == 'SUCCESS':
                successful_count += 1
            elif eval_result['status'] == 'TIMEOUT':
                timeout_count += 1
            else:
                failed_count += 1

            # Save to checkpoint
            if eval_result['status'] == 'SUCCESS':
                checkpoint_manager.save_sample_result(
                    sample_id=sample_id,
                    status="success",
                    metrics={"factual_correctness": eval_result['factual_correctness']}
                )
            else:
                checkpoint_manager.save_sample_result(
                    sample_id=sample_id,
                    status="failed",
                    error=eval_result['error']
                )

            # Write immediately
            writer.write_sample(result_record)
            all_results.append(result_record)

            # Save checkpoint
            checkpoint_manager.set_elapsed_seconds(time.time() - start_time)
            checkpoint_manager.save_checkpoint()

            logger.info(f"  Status: {eval_result['status']}")
            if eval_result['status'] == 'SUCCESS':
                logger.info(f"  Score: {eval_result['factual_correctness']:.4f}")
                logger.info(f"  Time: {eval_result['processing_time_ms'] / 1000:.1f}s")

            logger.info(f"\nProgress: {successful_count + timeout_count + failed_count}/{len(dataset)} samples")
            logger.info(f"  Success: {successful_count}, Timeout: {timeout_count}, Failed: {failed_count}")

        elapsed_time = time.time() - start_time

        logger.info(f"\n✓ Evaluation completed in {elapsed_time:.1f}s")
        logger.info(f"Successful: {successful_count}")
        logger.info(f"Timeout: {timeout_count}")
        logger.info(f"Failed: {failed_count}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.info(f"Partial results saved to {detailed_results_path}")
        logger.info(f"Checkpoint saved to {checkpoint_file}")
        logger.info("Use --resume to continue from where you left off")
        sys.exit(1)

    # Generate report
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING REPORT")
    logger.info("=" * 80)

    try:
        correlation_stats = analyze_timeout_correlation(all_results, logger)

        report_config = {
            'llm_model': args.model,
            'mode': args.mode,
            'atomicity': args.atomicity,
            'coverage': args.coverage,
            'timeout_seconds': args.timeout
        }

        generate_report(all_results, correlation_stats, report_config, str(report_path), logger)
        logger.info(f"✓ Report generated: {report_path}")

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Timeout: {timeout_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Total time: {elapsed_time:.1f}s")
    logger.info(f"Detailed results: {detailed_results_path}")
    logger.info(f"Report: {report_path}")

    # Calculate average score
    if successful_count > 0:
        success_results = [r for r in all_results if r['status'] == 'SUCCESS']
        avg_score = sum(r['factual_correctness'] for r in success_results) / len(success_results)
        logger.info(f"\nAverage Factual Correctness: {avg_score:.4f}")

    # Clear checkpoint on success
    if timeout_count == 0 and failed_count == 0:
        checkpoint_manager.clear_checkpoint()
        logger.info("\n✓ All samples evaluated successfully!")
    elif args.retry_failed:
        logger.info(f"\n⚠ {timeout_count + failed_count} samples still failing after retry")
    else:
        logger.info(f"\n⚠ {timeout_count + failed_count} samples failed - use --resume --retry-failed to retry")


if __name__ == "__main__":
    main()
