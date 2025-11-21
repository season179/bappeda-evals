#!/usr/bin/env python3
"""
RAG Executor for Evaluation
Runs test queries from multihop_translated.csv against SmartKnowledge RAG app
and captures results for Ragas evaluation
"""

import argparse
import csv
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
from lib.rag_client import RAGClient
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
        description="Execute RAG queries for evaluation",
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
        default="multihop_translated.csv",
        help="Input test dataset CSV file (default: multihop_translated.csv)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process (for testing)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose console logging"
    )

    return parser.parse_args()


def load_test_dataset(file_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load test dataset from CSV file

    Args:
        file_path: Path to CSV file
        limit: Optional limit on number of rows to load

    Returns:
        DataFrame with test queries
    """
    df = pd.read_csv(file_path)

    # Validate required columns
    required_columns = ['user_input', 'reference_contexts', 'reference']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Apply limit if specified
    if limit is not None:
        df = df.head(limit)

    return df


def create_results_directory(results_dir: str):
    """Create results directory if it doesn't exist"""
    Path(results_dir).mkdir(parents=True, exist_ok=True)


def initialize_results_file(output_path: str):
    """
    Initialize CSV file with headers

    Args:
        output_path: Path to output CSV file
    """
    headers = [
        'query_id',
        'user_input',
        'reference_contexts',
        'reference',
        'actual_contexts',
        'actual_answer',
        'tool_calls_json',
        'api_latency_ms',
        'status',
        'error'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def append_result(
    output_path: str,
    query_id: int,
    user_input: str,
    reference_contexts: str,
    reference: str,
    actual_contexts: List[str],
    actual_answer: str,
    tool_calls_json: str,
    api_latency_ms: int,
    status: str,
    error: str = ""
):
    """
    Append a single result to the CSV file

    Args:
        output_path: Path to output CSV file
        query_id: Query index
        user_input: Test query
        reference_contexts: Ground truth contexts
        reference: Ground truth answer
        actual_contexts: Retrieved contexts from RAG
        actual_answer: Generated answer from RAG
        tool_calls_json: JSON string of tool executions
        api_latency_ms: API response time
        status: SUCCESS or FAILED
        error: Error message if FAILED
    """
    # Convert actual_contexts list to JSON string
    actual_contexts_json = json.dumps(actual_contexts, ensure_ascii=False)

    row = [
        query_id,
        user_input,
        reference_contexts,
        reference,
        actual_contexts_json,
        actual_answer,
        tool_calls_json,
        api_latency_ms,
        status,
        error
    ]

    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def execute_single_query(
    client: RAGClient,
    query: str,
    logger
) -> Dict:
    """
    Execute a single query against the RAG API

    Args:
        client: RAG API client
        query: User query
        logger: Logger instance

    Returns:
        Dictionary with:
            - actual_contexts: List of retrieved contexts
            - actual_answer: Generated answer
            - tool_calls_json: JSON string of tool executions
            - api_latency_ms: API latency
            - status: SUCCESS or FAILED
            - error: Error message if FAILED
    """
    try:
        # Call the RAG API
        response = client.call_chat_api(query)

        # Check if response has error status
        if response.get('status') == 'error':
            error_msg = response.get('error', 'Unknown error')
            return {
                'actual_contexts': [],
                'actual_answer': '',
                'tool_calls_json': '',
                'api_latency_ms': client.get_latency(response),
                'status': 'FAILED',
                'error': f"API error: {error_msg}"
            }

        # Extract data from response
        actual_contexts = client.parse_tool_executions(response)
        actual_answer = client.get_answer(response)
        tool_calls_json = client.get_tool_calls_json(response)
        api_latency_ms = client.get_latency(response)

        return {
            'actual_contexts': actual_contexts,
            'actual_answer': actual_answer,
            'tool_calls_json': tool_calls_json,
            'api_latency_ms': api_latency_ms,
            'status': 'SUCCESS',
            'error': ''
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Query execution failed: {error_msg}")

        return {
            'actual_contexts': [],
            'actual_answer': '',
            'tool_calls_json': '',
            'api_latency_ms': 0,
            'status': 'FAILED',
            'error': error_msg
        }


def main():
    """Main execution function"""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    executor_config = config.get('rag_executor', {})
    logging_config = config.get('logging', {})

    # Setup logging
    import logging as log_module
    console_level_str = logging_config.get('console_level', 'INFO')
    if args.verbose:
        console_level_str = 'DEBUG'

    console_level = getattr(log_module, console_level_str)
    file_level = getattr(log_module, logging_config.get('file_level', 'DEBUG'))

    logger = setup_logger(
        name="rag_executor",
        log_dir=logging_config.get('directory', 'logs'),
        console_level=console_level,
        file_level=file_level,
        enable_api_log=logging_config.get('api_log_enabled', True),
        enable_error_log=logging_config.get('error_log_enabled', True)
    )

    logger.info("=" * 80)
    logger.info("RAG Executor for Evaluation")
    logger.info("=" * 80)

    # Configuration
    api_base_url = executor_config.get('api_base_url', 'http://localhost:3000')
    timeout = executor_config.get('timeout_seconds', 120)
    max_retries = executor_config.get('max_retries', 3)
    retry_delay = executor_config.get('retry_delay_seconds', 2)
    results_dir = executor_config.get('results_dir', './results')
    output_file = executor_config.get('output_file', 'eval_results.csv')
    checkpoint_enabled = executor_config.get('checkpoint_enabled', True)
    checkpoint_file = executor_config.get('checkpoint_file', 'executor_checkpoint.json')
    checkpoint_interval = executor_config.get('checkpoint_interval', 10)

    output_path = Path(results_dir) / output_file

    logger.info(f"Configuration:")
    logger.info(f"  API Base URL: {api_base_url}")
    logger.info(f"  Input file: {args.input}")
    logger.info(f"  Output file: {output_path}")
    logger.info(f"  Timeout: {timeout}s")
    logger.info(f"  Max retries: {max_retries}")
    logger.info(f"  Limit: {args.limit if args.limit else 'None (all queries)'}")

    # Pre-flight checks
    logger.info("\n" + "=" * 80)
    logger.info("PRE-FLIGHT CHECKS")
    logger.info("=" * 80)

    # Check if input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    logger.info(f"✓ Input file exists: {args.input}")

    # Create results directory
    create_results_directory(results_dir)
    logger.info(f"✓ Results directory ready: {results_dir}")

    # Initialize RAG client
    logger.info(f"\nInitializing RAG client...")
    client = RAGClient(
        base_url=api_base_url,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    # Health check
    logger.info(f"Checking API health at {api_base_url}...")
    if not client.health_check():
        logger.error(f"✗ API health check failed!")
        logger.error(f"  Please ensure SmartKnowledge is running at {api_base_url}")
        sys.exit(1)
    logger.info(f"✓ API is accessible")

    # Load test dataset
    logger.info(f"\nLoading test dataset...")
    try:
        df = load_test_dataset(args.input, limit=args.limit)
        logger.info(f"✓ Loaded {len(df)} test queries")
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")
        sys.exit(1)

    # Initialize checkpoint manager
    checkpoint_manager = None
    start_index = 0

    if checkpoint_enabled:
        checkpoint_manager = CheckpointManager(checkpoint_file)

        if args.resume:
            checkpoint_data = checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                start_index = checkpoint_data.get('last_processed_index', 0) + 1
                logger.info(f"✓ Resuming from query {start_index}")
        else:
            # Clear checkpoint for fresh run
            checkpoint_manager.clear_checkpoint()

    # Initialize output file if starting fresh
    if start_index == 0:
        initialize_results_file(output_path)
        logger.info(f"✓ Initialized output file: {output_path}")

    # Execute queries
    logger.info("\n" + "=" * 80)
    logger.info("EXECUTING RAG QUERIES")
    logger.info("=" * 80)

    total_queries = len(df)
    success_count = 0
    failed_count = 0
    total_latency = 0

    start_time = time.time()

    for idx in range(start_index, total_queries):
        row = df.iloc[idx]
        query_id = idx
        user_input = row['user_input']
        reference_contexts = row['reference_contexts']
        reference = row['reference']

        logger.info(f"\n[{idx + 1}/{total_queries}] Processing query {query_id}")
        logger.info(f"  Query: {user_input[:100]}..." if len(user_input) > 100 else f"  Query: {user_input}")

        # Execute query
        result = execute_single_query(client, user_input, logger)

        # Log result
        if result['status'] == 'SUCCESS':
            success_count += 1
            total_latency += result['api_latency_ms']
            logger.info(f"  ✓ Success (latency: {result['api_latency_ms']}ms)")
            logger.info(f"  Retrieved {len(result['actual_contexts'])} contexts")
            logger.info(f"  Answer length: {len(result['actual_answer'])} chars")
        else:
            failed_count += 1
            logger.error(f"  ✗ Failed: {result['error']}")

        # Append to results file
        append_result(
            output_path,
            query_id,
            user_input,
            reference_contexts,
            reference,
            result['actual_contexts'],
            result['actual_answer'],
            result['tool_calls_json'],
            result['api_latency_ms'],
            result['status'],
            result['error']
        )

        # Save checkpoint
        if checkpoint_manager and (idx + 1) % checkpoint_interval == 0:
            checkpoint_manager.save_checkpoint(
                metadata={
                    'last_processed_index': idx,
                    'total_processed': idx + 1,
                    'success_count': success_count,
                    'failed_count': failed_count
                }
            )
            logger.info(f"  Checkpoint saved at query {idx}")

    # Final statistics
    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_latency = total_latency / success_count if success_count > 0 else 0

    logger.info("\n" + "=" * 80)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total queries: {total_queries}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Average latency: {avg_latency:.0f}ms")
    logger.info(f"Total time: {elapsed_time:.1f}s")
    logger.info(f"Output file: {output_path}")

    # Clear checkpoint on successful completion
    if checkpoint_manager:
        checkpoint_manager.clear_checkpoint()

    # Close client
    client.close()

    logger.info("\n✓ RAG Executor completed successfully!")


if __name__ == "__main__":
    main()
