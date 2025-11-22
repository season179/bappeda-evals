#!/usr/bin/env python3
"""
Ragas Evaluation Script

Evaluates RAG executor results using Ragas metrics and generates comprehensive reports.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

from lib import setup_logger_from_config
from lib.data_transformer import RagasDataTransformer
from lib.ragas_evaluator import RagasEvaluator
from lib.report_generator import RagasReportGenerator
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
        description="Run Ragas evaluation on RAG executor results",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        "--metric",
        type=str,
        required=True,
        help="Single metric to evaluate (required). Available: context_precision, context_recall, "
             "context_entity_recall, answer_relevancy, faithfulness, answer_correctness, "
             "answer_similarity, context_utilization"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose console logging"
    )

    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip queries without retrieved contexts (default: assign zero scores)"
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only failed samples from previous run (requires --resume)"
    )

    return parser.parse_args()


def create_output_directory(output_dir: str):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def generate_metadata(row):
    """
    Generate metadata dict for each row

    Args:
        row: DataFrame row containing query results

    Returns:
        Dictionary with metadata fields
    """
    retrieved_contexts = row.get('retrieved_contexts', [])
    has_contexts = isinstance(retrieved_contexts, list) and len(retrieved_contexts) > 0

    return {
        'has_contexts': has_contexts,
        'query_id': row.name,
        'error': '',
        'api_latency_ms': 0
    }


def main():
    """Main execution function"""
    args = parse_args()

    # Validate metric before proceeding
    AVAILABLE_METRICS = [
        'context_precision', 'context_recall', 'context_entity_recall',
        'answer_relevancy', 'faithfulness', 'answer_correctness',
        'answer_similarity', 'context_utilization'
    ]

    if args.metric not in AVAILABLE_METRICS:
        print(f"\nError: Invalid metric '{args.metric}'")
        print(f"\nAvailable metrics:")
        for metric in AVAILABLE_METRICS:
            print(f"  - {metric}")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)
    eval_config = config.get('ragas_evaluation', {})
    api_config = config.get('api', {})
    llm_config = config.get('llm', {})
    embeddings_config = config.get('embeddings', {})
    # Setup logging
    logger = setup_logger_from_config(
        name="ragas_evaluation",
        config=config,
        verbose=args.verbose,
        title="Ragas Evaluation"
    )

    # Configuration
    input_file = args.input  # Required argument
    output_dir = args.output_dir or eval_config.get('output_dir', './results')
    detailed_results_file = eval_config.get('detailed_results_file', 'ragas_eval_detailed.jsonl')
    report_file = eval_config.get('report_file', 'ragas_eval_report.md')
    metrics = [args.metric]  # Single metric as list for evaluator
    batch_size = eval_config.get('batch_size', 10)
    checkpoint_enabled = eval_config.get('checkpoint_enabled', True)
    checkpoint_file = eval_config.get('checkpoint_file', 'ragas_checkpoint.json')
    max_sample_attempts = eval_config.get('max_sample_attempts', 3)
    continue_on_batch_error = eval_config.get('continue_on_batch_error', True)
    include_failed = not args.skip_failed

    detailed_results_path = Path(output_dir) / detailed_results_file
    report_path = Path(output_dir) / report_file

    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables")
        sys.exit(1)

    # Get Ragas evaluation LLM config for logging (falls back to main LLM config)
    eval_llm_config = eval_config.get('llm', {})
    ragas_llm_model = eval_llm_config.get('model', llm_config.get('model'))
    ragas_llm_temperature = eval_llm_config.get('temperature', llm_config.get('temperature', 0.7))

    logger.info(f"Configuration:")
    logger.info(f"  Input file: {input_file}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Detailed results: {detailed_results_path}")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Ragas LLM Model: {ragas_llm_model}")
    logger.info(f"  Ragas LLM Temperature: {ragas_llm_temperature}")
    logger.info(f"  Embedding Model: {embeddings_config.get('model')}")
    logger.info(f"  Metric: {args.metric}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Include failed queries: {include_failed}")

    # Pre-flight checks
    logger.info("\n" + "=" * 80)
    logger.info("PRE-FLIGHT CHECKS")
    logger.info("=" * 80)

    # Check if input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    logger.info(f"✓ Input file exists: {input_file}")

    # Create output directory
    create_output_directory(output_dir)
    logger.info(f"✓ Output directory ready: {output_dir}")

    # Load and transform data
    logger.info("\n" + "=" * 80)
    logger.info("LOADING AND TRANSFORMING DATA")
    logger.info("=" * 80)

    try:
        transformer = RagasDataTransformer()
        dataset = transformer.load_and_transform(
            jsonl_path=input_file,
            include_failed=include_failed
        )

        logger.info(f"✓ Loaded and transformed {len(dataset)} records")

        # Validate dataset
        validation_stats = transformer.validate_dataset(dataset)

    except Exception as e:
        logger.error(f"Failed to load and transform data: {e}")
        sys.exit(1)

    # Initialize Ragas evaluator
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING RAGAS EVALUATOR")
    logger.info("=" * 80)

    # Get timeout configurations
    llm_timeout = eval_config.get('llm_timeout_seconds', 300)
    embeddings_timeout = eval_config.get('embeddings_timeout_seconds', 180)

    try:
        evaluator = RagasEvaluator(
            api_key=api_key,
            api_base_url=api_config.get('base_url'),
            llm_model=ragas_llm_model,
            embedding_model=embeddings_config.get('model'),
            llm_temperature=ragas_llm_temperature,
            timeout=llm_timeout,
            embeddings_timeout=embeddings_timeout,
            max_retries=3  # Internal retry for API calls
        )
        logger.info("✓ Ragas evaluator initialized")

    except Exception as e:
        logger.error(f"Failed to initialize Ragas evaluator: {e}")
        sys.exit(1)

    # Initialize checkpoint manager
    checkpoint_manager = None
    skip_sample_ids = []
    retry_sample_ids = []

    if checkpoint_enabled:
        checkpoint_manager = EvaluationCheckpointManager(checkpoint_file)

        if args.resume:
            checkpoint_data = checkpoint_manager.load_checkpoint()
            checkpoint_manager.set_total_samples(len(dataset))

            if args.retry_failed:
                # Retry failed samples only
                retry_sample_ids = checkpoint_manager.get_failed_samples()
                logger.info(f"✓ Retry mode: {len(retry_sample_ids)} failed samples to retry")
            else:
                # Skip successful samples
                skip_sample_ids = checkpoint_manager.get_successful_samples()
                logger.info(f"✓ Resume mode: {len(skip_sample_ids)} samples already completed")
        else:
            # Clear checkpoint for fresh run
            checkpoint_manager.clear_checkpoint()
            checkpoint_manager.set_total_samples(len(dataset))

    # Run evaluation
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING RAGAS EVALUATION (INCREMENTAL)")
    logger.info("=" * 80)

    start_time = time.time()

    # Initialize incremental writer
    writer = IncrementalJSONLWriter(
        output_file=str(detailed_results_path),
        backup_enabled=True
    )

    # Clear output file if starting fresh (not resuming)
    if not args.resume and writer.exists():
        writer.clear()

    # Track aggregated metrics
    all_metric_scores = {}
    successful_sample_count = 0
    failed_sample_count = 0

    try:
        # Run incremental evaluation
        for batch_results, batch_sample_ids, batch_status in evaluator.evaluate_incremental(
            dataset=dataset,
            metric_names=metrics,
            batch_size=batch_size,
            show_progress=True,
            skip_sample_ids=skip_sample_ids,
            retry_sample_ids=retry_sample_ids if args.retry_failed else None,
            max_sample_attempts=max_sample_attempts
        ):
            # Process batch results
            for idx, sample_result in enumerate(batch_results):
                sample_id = batch_sample_ids[idx]

                # Extract metrics for this sample
                sample_metrics = {
                    k: v for k, v in sample_result.items()
                    if k not in ['user_input', 'reference_contexts', 'reference', 'retrieved_contexts', 'response']
                }

                # Check if sample has valid metrics (non-zero)
                has_valid_metrics = any(v > 0 for v in sample_metrics.values())

                if batch_status == "success" and has_valid_metrics:
                    # Save as successful
                    if checkpoint_manager:
                        checkpoint_manager.save_sample_result(
                            sample_id=sample_id,
                            status="success",
                            metrics=sample_metrics
                        )

                    # Accumulate metrics for averaging
                    for metric_name, score in sample_metrics.items():
                        if metric_name not in all_metric_scores:
                            all_metric_scores[metric_name] = []
                        all_metric_scores[metric_name].append(score)

                    successful_sample_count += 1

                else:
                    # Save as failed
                    if checkpoint_manager:
                        attempts = checkpoint_manager.get_sample_attempts(sample_id)
                        checkpoint_manager.save_sample_result(
                            sample_id=sample_id,
                            status="failed",
                            error=f"Batch status: {batch_status}"
                        )

                    failed_sample_count += 1

                # Write result to JSONL immediately
                writer.write_sample(sample_result)

            # Mark batch as completed
            if checkpoint_manager:
                checkpoint_manager.mark_batch_completed()
                checkpoint_manager.set_elapsed_seconds(time.time() - start_time)
                checkpoint_manager.save_checkpoint()

            logger.info(f"Progress: {successful_sample_count + failed_sample_count}/{len(dataset)} samples processed")

        elapsed_time = time.time() - start_time

        logger.info(f"\n✓ Evaluation completed in {elapsed_time:.1f}s")
        logger.info(f"Successful samples: {successful_sample_count}")
        logger.info(f"Failed samples: {failed_sample_count}")
        if successful_sample_count > 0:
            logger.info(f"Average time per successful sample: {elapsed_time / successful_sample_count:.2f}s")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.info(f"Partial results saved to {detailed_results_path}")
        if checkpoint_manager:
            logger.info(f"Checkpoint saved to {checkpoint_file}")
            logger.info("Use --resume to continue from where you left off")
        sys.exit(1)

    # Detailed results already saved incrementally
    logger.info("\n" + "=" * 80)
    logger.info("PREPARING SUMMARY")
    logger.info("=" * 80)

    # Calculate average metrics from accumulated scores
    results_dict = {
        metric_name: sum(scores) / len(scores)
        for metric_name, scores in all_metric_scores.items()
        if len(scores) > 0
    }

    logger.info(f"✓ Detailed results saved to {detailed_results_path}")
    logger.info(f"  Total samples written: {writer.get_sample_count()}")

    # Generate report
    try:
        report_generator = RagasReportGenerator()

        # Read detailed results from JSONL for report
        detailed_samples = writer.read_existing()
        detailed_df = pd.DataFrame(detailed_samples)

        # Generate _metadata column for report generator
        if not detailed_df.empty:
            detailed_df['_metadata'] = detailed_df.apply(generate_metadata, axis=1)
            logger.info(f"Generated _metadata for {len(detailed_df)} queries")
        else:
            logger.warning("No detailed results found for report generation")

        # Add config context for report
        report_config = {
            'llm_model': ragas_llm_model,
            'embedding_model': embeddings_config.get('model'),
            'api_base_url': api_config.get('base_url')
        }

        report_generator.generate_report(
            results=results_dict,
            detailed_results=detailed_df,
            output_path=str(report_path),
            config=report_config
        )

        logger.info(f"✓ Report generated: {report_path}")

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        # Don't exit - evaluation is complete

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total samples in dataset: {len(dataset)}")
    logger.info(f"Successful evaluations: {successful_sample_count}")
    logger.info(f"Failed evaluations: {failed_sample_count}")
    logger.info(f"Total time: {elapsed_time:.1f}s")
    logger.info(f"Detailed results: {detailed_results_path}")
    logger.info(f"Report: {report_path}")

    # Log final metrics (averaged over successful samples)
    if results_dict:
        logger.info("\nFinal Metrics (averaged over successful samples):")
        for metric_name, score in results_dict.items():
            logger.info(f"  {metric_name}: {score:.4f}")
    else:
        logger.warning("No metrics calculated (no successful samples)")

    # Clear checkpoint on successful completion
    if checkpoint_manager:
        checkpoint_manager.clear_checkpoint()

    logger.info("\n✓ Ragas evaluation completed successfully!")


if __name__ == "__main__":
    main()
