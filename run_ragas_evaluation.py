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

from lib import setup_logger
from lib.data_transformer import RagasDataTransformer
from lib.ragas_evaluator import RagasEvaluator
from lib.report_generator import RagasReportGenerator
from lib.state_manager import CheckpointManager

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
        default=None,
        help="Input JSONL file (default: from config)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config)"
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        default=None,
        help="Metrics to evaluate (default: all from config)"
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

    return parser.parse_args()


def create_output_directory(output_dir: str):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def save_detailed_results(
    dataset,
    results: Dict,
    output_path: str,
    logger
):
    """
    Save detailed per-query results to JSONL file

    Args:
        dataset: Evaluated Ragas dataset
        results: Ragas evaluation results
        output_path: Path to save JSONL file
        logger: Logger instance
    """
    logger.info(f"Saving detailed results to {output_path}")

    try:
        # Convert dataset to pandas DataFrame
        df = dataset.to_pandas()

        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # Convert row to dict
                record = row.to_dict()
                # Write as JSON line
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        logger.info(f"✓ Saved {len(df)} detailed results to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")
        raise


def main():
    """Main execution function"""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    eval_config = config.get('ragas_evaluation', {})
    api_config = config.get('api', {})
    llm_config = config.get('llm', {})
    embeddings_config = config.get('embeddings', {})
    logging_config = config.get('logging', {})

    # Setup logging
    import logging as log_module
    console_level_str = logging_config.get('console_level', 'INFO')
    if args.verbose:
        console_level_str = 'DEBUG'

    console_level = getattr(log_module, console_level_str)
    file_level = getattr(log_module, logging_config.get('file_level', 'DEBUG'))

    logger = setup_logger(
        name="ragas_evaluation",
        log_dir=logging_config.get('directory', 'logs'),
        console_level=console_level,
        file_level=file_level,
        enable_api_log=logging_config.get('api_log_enabled', True),
        enable_error_log=logging_config.get('error_log_enabled', True)
    )

    logger.info("=" * 80)
    logger.info("Ragas Evaluation")
    logger.info("=" * 80)

    # Configuration
    input_file = args.input or eval_config.get('input_file', './results/eval_results.jsonl')
    output_dir = args.output_dir or eval_config.get('output_dir', './results')
    detailed_results_file = eval_config.get('detailed_results_file', 'ragas_eval_detailed.jsonl')
    report_file = eval_config.get('report_file', 'ragas_eval_report.md')
    metrics = args.metrics or eval_config.get('metrics', None)
    batch_size = eval_config.get('batch_size', 10)
    checkpoint_enabled = eval_config.get('checkpoint_enabled', True)
    checkpoint_file = eval_config.get('checkpoint_file', 'ragas_checkpoint.json')
    checkpoint_interval = eval_config.get('checkpoint_interval', 10)
    max_retries = eval_config.get('max_retries', 3)
    retry_delay = eval_config.get('retry_delay_seconds', 5)
    include_failed = not args.skip_failed

    detailed_results_path = Path(output_dir) / detailed_results_file
    report_path = Path(output_dir) / report_file

    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables")
        sys.exit(1)

    logger.info(f"Configuration:")
    logger.info(f"  Input file: {input_file}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Detailed results: {detailed_results_path}")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  LLM Model: {llm_config.get('model')}")
    logger.info(f"  Embedding Model: {embeddings_config.get('model')}")
    logger.info(f"  Metrics: {metrics or 'all'}")
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
            llm_model=llm_config.get('model'),
            embedding_model=embeddings_config.get('model'),
            llm_temperature=llm_config.get('temperature', 0.7),
            timeout=llm_timeout,
            embeddings_timeout=embeddings_timeout,
            max_retries=max_retries
        )
        logger.info("✓ Ragas evaluator initialized")

    except Exception as e:
        logger.error(f"Failed to initialize Ragas evaluator: {e}")
        sys.exit(1)

    # Check for checkpoint
    checkpoint_manager = None
    start_index = 0

    if checkpoint_enabled:
        checkpoint_manager = CheckpointManager(checkpoint_file)

        if args.resume:
            checkpoint_data = checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                start_index = checkpoint_data.get('last_processed_index', 0) + 1
                logger.info(f"✓ Resuming from sample {start_index}")
        else:
            # Clear checkpoint for fresh run
            checkpoint_manager.clear_checkpoint()

    # Run evaluation
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING RAGAS EVALUATION")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        # Run evaluation with retry logic
        results_dict, evaluation_result = evaluator.evaluate_with_retry(
            dataset=dataset,
            metric_names=metrics,
            batch_size=batch_size,
            show_progress=True,
            max_retries=max_retries,
            retry_delay=retry_delay
        )

        elapsed_time = time.time() - start_time

        logger.info(f"\n✓ Evaluation completed in {elapsed_time:.1f}s")
        logger.info(f"Average time per sample: {elapsed_time / len(dataset):.2f}s")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

    # Save detailed results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    try:
        # Use the evaluation_result (EvaluationResult object) for detailed results
        save_detailed_results(evaluation_result.dataset, results_dict, detailed_results_path, logger)
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")
        # Continue to report generation

    # Generate report
    try:
        report_generator = RagasReportGenerator()

        # Convert evaluation dataset to DataFrame for report
        detailed_df = evaluation_result.dataset.to_pandas()

        # Add config context for report
        report_config = {
            'llm_model': llm_config.get('model'),
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
    logger.info(f"Total samples evaluated: {len(dataset)}")
    logger.info(f"Total time: {elapsed_time:.1f}s")
    logger.info(f"Detailed results: {detailed_results_path}")
    logger.info(f"Report: {report_path}")

    # Log final metrics
    logger.info("\nFinal Metrics:")
    for metric_name, score in results_dict.items():
        logger.info(f"  {metric_name}: {score:.4f}")

    # Clear checkpoint on successful completion
    if checkpoint_manager:
        checkpoint_manager.clear_checkpoint()

    logger.info("\n✓ Ragas evaluation completed successfully!")


if __name__ == "__main__":
    main()
