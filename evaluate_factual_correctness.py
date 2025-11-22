#!/usr/bin/env python3
"""
Factual Correctness Evaluation Script

Evaluates RAG factual correctness using Ragas FactualCorrectness metric.
Optimized to prevent timeout errors with:
- Single metric focus (no context extraction)
- Low atomicity/coverage (fewer claims)
- Individual sample timeouts
- Length correlation tracking
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import FactualCorrectness
from ragas.run_config import RunConfig

from lib import setup_logger_from_config
from lib.openrouter_chat import OpenRouterChatOpenAI
from lib.result_writer import IncrementalJSONLWriter
from lib.state_manager import EvaluationCheckpointManager

# Load environment variables
load_dotenv()


def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG factual correctness using Ragas",
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
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
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
        default=None,
        help="Output directory (default: from config)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["F1", "precision", "recall"],
        default=None,
        help="Scoring mode: F1 (both directions), precision (response→reference only), "
             "or recall (reference→response only). Default from config."
    )

    parser.add_argument(
        "--atomicity",
        type=str,
        choices=["low", "high"],
        default=None,
        help="Claim granularity: low (fewer claims, faster) or high (more granular). "
             "Default from config."
    )

    parser.add_argument(
        "--coverage",
        type=str,
        choices=["low", "high"],
        default=None,
        help="Claim coverage: low (main points only) or high (all details). "
             "Default from config."
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout per sample in seconds (default: from config)"
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
        "--verbose",
        action="store_true",
        help="Enable verbose console logging"
    )

    return parser.parse_args()


def load_executor_results(jsonl_path: str, logger) -> List[dict]:
    """
    Load executor results from JSONL file

    Args:
        jsonl_path: Path to executor output JSONL file
        logger: Logger instance

    Returns:
        List of result dictionaries
    """
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
    """
    Transform executor results to Ragas EvaluationDataset format

    Only extracts 3 fields: user_input, response (from actual_answer), reference
    Skips all context extraction (not needed for factual correctness)

    Args:
        records: List of executor result dictionaries
        include_failed: Whether to include failed queries (without responses)
        logger: Logger instance

    Returns:
        EvaluationDataset with SingleTurnSample objects
    """
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
                logger.warning(f"Skipping query_id {record.get('query_id')}: no response (use include_failed to evaluate)")
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
    """
    Initialize FactualCorrectness metric with specified parameters

    Args:
        mode: F1, precision, or recall
        atomicity: low (fewer claims) or high (granular)
        coverage: low (main points) or high (all details)
        llm: Language model for evaluation
        logger: Logger instance

    Returns:
        Configured FactualCorrectness metric
    """
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
    """
    Evaluate a single sample with timeout

    Args:
        sample: SingleTurnSample to evaluate
        metric: FactualCorrectness metric
        timeout_seconds: Timeout in seconds
        logger: Logger instance

    Returns:
        Dictionary with scores, timing, and status
    """
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
    """
    Analyze correlation between text length and timeouts

    Args:
        results: List of evaluation results
        logger: Logger instance

    Returns:
        Dictionary with correlation statistics
    """
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
    """
    Generate markdown report with scores and timeout analysis

    Args:
        results: List of evaluation results
        correlation_stats: Timeout correlation statistics
        config: Configuration dictionary
        output_path: Path to save report
        logger: Logger instance
    """
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

    # Load configuration
    config = load_config(args.config)
    fc_config = config.get('factual_correctness_evaluation', {})
    api_config = config.get('api', {})
    llm_config = config.get('llm', {})

    # Setup logging
    logger = setup_logger_from_config(
        name="factual_correctness",
        config=config,
        verbose=args.verbose,
        title="Factual Correctness Evaluation"
    )

    # Configuration (CLI args override config file)
    input_file = args.input
    output_dir = args.output_dir or fc_config.get('output_dir', './results')
    detailed_results_file = fc_config.get('detailed_results_file', 'factual_correctness_detailed.jsonl')
    report_file = fc_config.get('report_file', 'factual_correctness_report.md')
    checkpoint_file = fc_config.get('checkpoint_file', 'factual_correctness_checkpoint.json')

    mode = args.mode or fc_config.get('mode', 'F1')
    atomicity = args.atomicity or fc_config.get('atomicity', 'low')
    coverage = args.coverage or fc_config.get('coverage', 'low')
    timeout_seconds = args.timeout or fc_config.get('sample_timeout_seconds', 600)
    max_sample_attempts = fc_config.get('max_sample_attempts', 3)
    include_failed = not fc_config.get('skip_failed_queries', False)

    detailed_results_path = Path(output_dir) / detailed_results_file
    report_path = Path(output_dir) / report_file

    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables")
        sys.exit(1)

    # Get LLM config
    fc_llm_config = fc_config.get('llm', {})
    llm_model = fc_llm_config.get('model', llm_config.get('model'))
    llm_temperature = fc_llm_config.get('temperature', llm_config.get('temperature', 0.7))
    llm_timeout = fc_config.get('llm_timeout_seconds', 600)
    max_retries = fc_config.get('max_retries', 15)
    max_wait = fc_config.get('max_wait', 90)

    logger.info(f"Configuration:")
    logger.info(f"  Input file: {input_file}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Detailed results: {detailed_results_path}")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  LLM Model: {llm_model}")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Atomicity: {atomicity}")
    logger.info(f"  Coverage: {coverage}")
    logger.info(f"  Timeout per sample: {timeout_seconds}s")
    logger.info(f"  Include failed queries: {include_failed}")

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
        dataset = transform_to_dataset(executor_results, include_failed, logger)
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
            base_url=api_config.get('base_url'),
            model=llm_model,
            temperature=llm_temperature,
            timeout=llm_timeout,
            max_retries=max_retries
        )
        logger.info(f"✓ LLM initialized: {llm_model}")

    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        sys.exit(1)

    # Initialize metric
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING METRIC")
    logger.info("=" * 80)

    try:
        metric = initialize_metric(mode, atomicity, coverage, llm, logger)
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

            while attempt < max_sample_attempts:
                attempt += 1

                if attempt > 1:
                    logger.info(f"  Retry attempt {attempt}/{max_sample_attempts}")

                eval_result = asyncio.run(
                    evaluate_single_sample(sample, metric, timeout_seconds, logger)
                )

                if eval_result['status'] == 'SUCCESS':
                    break
                elif eval_result['status'] == 'TIMEOUT':
                    logger.warning(f"  Timeout on attempt {attempt}")
                    if attempt < max_sample_attempts:
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
                    "mode": mode,
                    "atomicity": atomicity,
                    "coverage": coverage
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
            'llm_model': llm_model,
            'mode': mode,
            'atomicity': atomicity,
            'coverage': coverage,
            'timeout_seconds': timeout_seconds
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
