#!/usr/bin/env python3
"""
Synthetic Test Dataset Generation for RAG Evaluation using Ragas

This script generates a comprehensive synthetic evaluation dataset for RAG applications
using the Ragas framework with OpenRouter models.

Features:
- Checkpoint/resume capability
- Incremental result saving
- Detailed progress tracking
- API validation
- Robust error handling with retries

Models used:
- LLM: x-ai/grok-code-fast-1 (via OpenRouter)
- Embeddings: qwen/qwen3-embedding-8b (via OpenRouter)
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution

# Import our custom library
from lib import (
    APIValidator,
    CheckpointManager,
    DetailedProgressTracker,
    IncrementalCSVWriter,
    setup_logger,
)
from lib.error_handlers import format_error_message

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
        description="Generate synthetic testset for RAG evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear checkpoint and start fresh"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: use 1 document and generate 5 samples"
    )

    parser.add_argument(
        "--validate-api",
        action="store_true",
        help="Validate API connectivity and exit"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging to console"
    )

    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Custom output file path"
    )

    return parser.parse_args()


def validate_api(config: dict, api_key: str, logger) -> bool:
    """
    Run API validation checks

    Returns:
        True if validation passed
    """
    logger.info("=" * 80)
    logger.info("API Validation")
    logger.info("=" * 80)

    validator = APIValidator(
        api_key=api_key,
        base_url=config['api']['base_url'],
        llm_model=config['llm']['model'],
        embedding_model=config['embeddings']['model']
    )

    logger.info(validator.get_validation_summary())
    logger.info("")

    success, errors = validator.validate_all()

    if not success:
        logger.error("")
        logger.error("=" * 80)
        logger.error("API Validation Failed!")
        logger.error("=" * 80)
        for service, error in errors.items():
            logger.error(f"\n{service.upper()}:")
            logger.error(error)
        return False

    logger.info("")
    logger.info("=" * 80)
    logger.info("API Validation Successful!")
    logger.info("=" * 80)
    return True


def main():
    """Main function to generate synthetic testset"""

    # Parse arguments
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Setup logging
    console_level = "DEBUG" if args.verbose else config['logging']['console_level']
    logger = setup_logger(
        log_dir=config['logging']['directory'],
        console_level=getattr(__import__('logging'), console_level),
        file_level=getattr(__import__('logging'), config['logging']['file_level']),
        enable_api_log=config['logging']['api_log_enabled'],
        enable_error_log=config['logging']['error_log_enabled']
    )

    logger.info("=" * 80)
    logger.info("RAG Evaluation - Synthetic Testset Generation")
    logger.info("=" * 80)
    logger.info("")

    # Validate API key
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        logger.error(
            "OPENROUTER_API_KEY not found. Please:\n"
            "1. Copy .env.example to .env\n"
            "2. Add your OpenRouter API key to .env\n"
            "3. Get your key from: https://openrouter.ai/settings/keys"
        )
        sys.exit(1)

    # Run API validation if requested or enabled in config
    if args.validate_api or config['validation']['enabled']:
        if not validate_api(config, OPENROUTER_API_KEY, logger):
            if args.validate_api or config['validation']['fail_fast']:
                sys.exit(1)
            logger.warning("Proceeding despite validation warnings...")

    # Exit if only validation was requested
    if args.validate_api:
        sys.exit(0)

    # Initialize components
    checkpoint_enabled = config['checkpoint']['enabled'] and not args.no_checkpoint
    checkpoint_file = config['checkpoint']['file']

    checkpoint_manager = CheckpointManager(checkpoint_file) if checkpoint_enabled else None

    # Handle reset flag
    if args.reset and checkpoint_manager:
        logger.info("Clearing checkpoint...")
        checkpoint_manager.clear_checkpoint()

    # Handle resume flag
    if args.resume and checkpoint_manager and checkpoint_manager.has_checkpoint():
        logger.info("=" * 80)
        logger.info("Resuming from Checkpoint")
        logger.info("=" * 80)
        state = checkpoint_manager.load_checkpoint()
        logger.info(checkpoint_manager.get_summary())
        logger.info("")
    elif checkpoint_manager and checkpoint_manager.has_checkpoint() and not args.reset:
        logger.warning("=" * 80)
        logger.warning("Checkpoint found! Use --resume to continue or --reset to start fresh")
        logger.warning("=" * 80)
        logger.info(checkpoint_manager.get_summary())
        response = input("\nContinue anyway and overwrite checkpoint? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Exiting. Use --resume or --reset flag to proceed.")
            sys.exit(0)
        checkpoint_manager.clear_checkpoint()

    # Determine test size
    if args.test:
        test_size = config['test_mode']['sample_limit']
        doc_limit = config['test_mode']['document_limit']
        logger.info(f"🧪 Test mode enabled: {doc_limit} document(s), {test_size} samples")
        logger.info("")
    else:
        test_size = config['generation']['test_size']
        doc_limit = None

    # Setup output files
    output_file = args.output if args.output else config['output']['final_file']
    result_writer = IncrementalCSVWriter(
        output_file=config['output']['partial_file'],
        final_file=output_file,
        backup_enabled=config['output']['backup_enabled']
    )

    # Step 1: Load documents
    logger.info(f"[1/6] Loading documents from {config['knowledge_base']['directory']}...")
    loader = DirectoryLoader(
        config['knowledge_base']['directory'],
        glob=config['knowledge_base']['glob_pattern'],
        loader_cls=TextLoader,
        show_progress=True
    )
    documents = loader.load()

    # Ensure metadata includes filename
    for doc in documents:
        if 'source' in doc.metadata and 'filename' not in doc.metadata:
            doc.metadata['filename'] = doc.metadata['source']

    # Limit documents in test mode
    if doc_limit and len(documents) > doc_limit:
        documents = documents[:doc_limit]
        logger.info(f"      Limited to {doc_limit} document(s) for testing")

    logger.info(f"      Loaded {len(documents)} document(s)")
    logger.info("")

    # Step 2: Configure LLM
    logger.info(f"[2/6] Configuring LLM: {config['llm']['model']}...")
    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config['llm']['model'],
            api_key=OPENROUTER_API_KEY,
            base_url=config['api']['base_url'],
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens']
        )
    )
    logger.info("      LLM configured successfully")
    logger.info("")

    # Step 3: Configure Embeddings
    logger.info(f"[3/6] Configuring Embeddings: {config['embeddings']['model']}...")
    openai_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=config['api']['base_url']
    )
    generator_embeddings = OpenAIEmbeddings(
        client=openai_client,
        model=config['embeddings']['model']
    )
    logger.info("      Embeddings configured successfully")
    logger.info("")

    # Step 4: Initialize TestsetGenerator
    logger.info("[4/6] Initializing TestsetGenerator...")
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )
    logger.info("      TestsetGenerator initialized")
    logger.info("")

    # Step 5: Create custom query distribution
    logger.info("[5/6] Creating query distribution...")
    dist_config = config['generation']['distribution']
    logger.info("      Distribution strategy:")
    logger.info(f"        - SingleHopSpecificQuerySynthesizer: {dist_config['single_hop']*100:.0f}%")
    logger.info(f"        - MultiHopAbstractQuerySynthesizer: {dist_config['multi_hop_abstract']*100:.0f}%")
    logger.info(f"        - MultiHopSpecificQuerySynthesizer: {dist_config['multi_hop_specific']*100:.0f}%")

    query_distribution = default_query_distribution(generator_llm)
    custom_distribution = [
        (query_distribution[0][0], dist_config['single_hop']),
        (query_distribution[1][0], dist_config['multi_hop_abstract']),
        (query_distribution[2][0], dist_config['multi_hop_specific']),
    ]

    logger.info(f"      Configured for {test_size} samples")
    logger.info("")

    # Step 6: Generate testset
    logger.info(f"[6/6] Generating synthetic testset ({test_size} samples)...")
    logger.info("      This may take several minutes...")

    # Initialize progress tracker
    progress_tracker = DetailedProgressTracker(
        total_documents=len(documents),
        target_samples=test_size,
        progress_file=config['progress']['file'],
        update_interval=config['progress']['update_interval']
    ) if config['progress']['enabled'] else None

    if progress_tracker:
        progress_tracker.start()

    logger.info("")

    try:
        # Generate testset
        dataset = generator.generate_with_langchain_docs(
            documents,
            testset_size=test_size,
            query_distribution=custom_distribution
        )

        logger.info("      Testset generation completed successfully!")
        logger.info("")

        # Convert to pandas DataFrame
        df = dataset.to_pandas()

        # Save final results
        logger.info(f"[OUTPUT] Finalizing results to {output_file}...")
        result_writer.write_dataframe(df)
        logger.info(f"         Saved {len(df)} samples")
        logger.info("")

        # Mark progress as complete
        if progress_tracker:
            progress_tracker.complete()

        # Clear checkpoint on success
        if checkpoint_manager:
            checkpoint_manager.clear_checkpoint()

        # Display summary
        logger.info("=" * 80)
        logger.info("Generation Complete!")
        logger.info("=" * 80)
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Output file: {output_file}")
        if progress_tracker:
            logger.info(f"Elapsed time: {progress_tracker.format_time(progress_tracker.get_elapsed_time())}")
        logger.info("")

        # Display first few samples
        logger.info("Sample preview:")
        logger.info("-" * 80)
        logger.info(df.head(3).to_string())
        logger.info("")

    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("=" * 80)
        logger.warning("Generation interrupted by user")
        logger.warning("=" * 80)

        # Save partial results
        partial_df = result_writer.read_partial_results()
        if partial_df is not None and len(partial_df) > 0:
            logger.info(f"Saved {len(partial_df)} partial results to {config['output']['partial_file']}")
            logger.info("Use --resume to continue from checkpoint")

        if progress_tracker:
            progress_tracker.error("Interrupted by user")

        sys.exit(1)

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("Testset generation failed!")
        logger.error("=" * 80)
        logger.error(format_error_message(e))

        # Save partial results if any
        partial_df = result_writer.read_partial_results()
        if partial_df is not None and len(partial_df) > 0:
            logger.info(f"Saved {len(partial_df)} partial results to {config['output']['partial_file']}")

        if progress_tracker:
            progress_tracker.error(str(e))

        raise


if __name__ == "__main__":
    main()
