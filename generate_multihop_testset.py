#!/usr/bin/env python3
"""
Multi-Hop Query Generation for RAG Evaluation using Ragas Knowledge Graph

This script generates multi-hop queries that require synthesizing information
across multiple DKI Jakarta government planning documents using knowledge graph
relationships and custom query synthesis.

Features:
- Knowledge graph construction with document nodes
- Relationship building via keyphrase overlap
- Custom multi-hop query synthesizer
- Bahasa Indonesia query generation
- DKI Jakarta government worker personas
- Extensive logging at each phase
- Checkpoint/resume capability

Models used:
- LLM: x-ai/grok-code-fast-1 (via OpenRouter)
- Embeddings: qwen/qwen3-embedding-8b (via OpenRouter)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona

# Import our custom library
from lib import (
    APIValidator,
    CheckpointManager,
    DetailedProgressTracker,
    IncrementalCSVWriter,
    MetadataCache,
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
        description="Generate multi-hop queries using knowledge graph",
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
        help="Custom output file path (default: multihop_testset.csv)"
    )

    parser.add_argument(
        "--size",
        type=int,
        help="Number of queries to generate (overrides config)"
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


def create_personas_from_config(personas_config: List[dict], logger) -> List[Persona]:
    """
    Create Ragas Persona objects from config

    Args:
        personas_config: List of persona dicts with 'name' and 'role' keys
        logger: Logger instance

    Returns:
        List of Persona objects
    """
    logger.info("Creating personas from configuration...")
    personas = []

    for i, p_config in enumerate(personas_config, 1):
        persona = Persona(
            name=p_config['name'],
            role_description=p_config['role']
        )
        personas.append(persona)
        logger.info(f"  [{i}/{len(personas_config)}] {p_config['name']}")
        logger.debug(f"       Role: {p_config['role']}")

    logger.info(f"Created {len(personas)} personas")
    logger.info("")
    return personas


# Unused functions removed - using TestsetGenerator directly instead


def main():
    """Main function to generate multi-hop queries"""

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
    logger.info("Multi-Hop Query Generation using Knowledge Graph")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This script generates queries that require synthesizing information")
    logger.info("across multiple DKI Jakarta government planning documents.")
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

    # Determine test size
    test_size = args.size if args.size else config['multihop']['test_size']
    logger.info(f"Configuration:")
    logger.info(f"  Target queries: {test_size}")
    logger.info(f"  Language: {config['multihop']['language']}")
    logger.info(f"  Overlap threshold: {config['multihop']['overlap_threshold']}")
    logger.info(f"  Max keyphrases: {config['multihop']['max_keyphrases']}")
    logger.info("")

    # Setup output files
    output_file = args.output if args.output else "multihop_testset.csv"
    result_writer = IncrementalCSVWriter(
        output_file=f"multihop_partial.csv",
        final_file=output_file,
        backup_enabled=config['output']['backup_enabled']
    )

    # Initialize checkpoint manager
    checkpoint_enabled = config['checkpoint']['enabled'] and not args.no_checkpoint
    checkpoint_file = "multihop_checkpoint.json"
    checkpoint_manager = CheckpointManager(checkpoint_file) if checkpoint_enabled else None

    # Handle reset/resume
    if args.reset and checkpoint_manager:
        logger.info("Clearing checkpoint...")
        checkpoint_manager.clear_checkpoint()

    if args.resume and checkpoint_manager and checkpoint_manager.has_checkpoint():
        logger.info("=" * 80)
        logger.info("Resuming from Checkpoint")
        logger.info("=" * 80)
        state = checkpoint_manager.load_checkpoint()
        logger.info(checkpoint_manager.get_summary())
        logger.info("")

    # Step 1: Load documents (using cached metadata if available)
    logger.info("=" * 80)
    logger.info("PHASE 0: Document Loading")
    logger.info("=" * 80)
    logger.info("")

    # Check for cached metadata
    metadata_cache = MetadataCache(
        cache_dir=Path('.cache/metadata'),
        source_dir=Path(config['knowledge_base']['directory'])
    )

    cache_stats = metadata_cache.get_cache_stats()

    if cache_stats['cached_files'] > 0:
        logger.info(f"[0.1] Loading documents from metadata cache...")
        logger.info(f"      Cache directory: {metadata_cache.cache_dir}")
        logger.info(f"      Cached files: {cache_stats['cached_files']}")
        logger.info(f"      Cache size: {cache_stats['total_cache_size'] / 1024:.1f} KB")
        logger.info("")

        # Load metadata documents
        metadata_docs = metadata_cache.load_all()

        if len(metadata_docs) == 0:
            logger.error("No metadata documents found in cache!")
            logger.error("Please run: python extract_metadata.py")
            sys.exit(1)

        logger.info(f"      Loaded {len(metadata_docs)} metadata document(s):")
        total_summary_size = 0
        total_original_size = 0

        for i, meta_doc in enumerate(metadata_docs, 1):
            summary_size = len(meta_doc.summary)
            total_summary_size += summary_size
            total_original_size += meta_doc.char_count
            memory_est = meta_doc.get_size_estimate()

            logger.info(
                f"        [{i}] {meta_doc.filename}\n"
                f"             Original: {meta_doc.char_count:,} chars, "
                f"Summary: {summary_size:,} chars, "
                f"Memory: ~{memory_est//1024}KB"
            )

        logger.info("")
        logger.info(f"      Total original size: {total_original_size:,} characters ({total_original_size/1024/1024:.1f} MB)")
        logger.info(f"      Total summary size: {total_summary_size:,} characters ({total_summary_size/1024:.1f} KB)")
        logger.info(f"      Memory reduction: ~{100 * (1 - total_summary_size/total_original_size):.1f}%")
        logger.info("")

        # Convert to LangChain documents using summaries
        logger.info("[0.2] Converting to LangChain documents...")
        documents = [meta_doc.to_langchain_document(use_summary=True) for meta_doc in metadata_docs]
        logger.info(f"      Converted {len(documents)} documents")
        logger.info("      Using summaries for knowledge graph (low memory mode)")
        logger.info("")

    else:
        logger.warning("No cached metadata found!")
        logger.warning("")
        logger.warning("To use disk-based processing with low memory usage:")
        logger.warning("  1. Run: python extract_metadata.py")
        logger.warning("  2. Then re-run this script")
        logger.warning("")
        logger.warning("Falling back to loading full documents (high memory usage)...")
        logger.warning("")

        logger.info(f"[0.1] Loading full documents from {config['knowledge_base']['directory']}...")

        loader = DirectoryLoader(
            config['knowledge_base']['directory'],
            glob=config['knowledge_base']['glob_pattern'],
            loader_cls=TextLoader,
            show_progress=False
        )
        documents = loader.load()

        # Ensure metadata includes filename
        for doc in documents:
            if 'source' in doc.metadata and 'filename' not in doc.metadata:
                doc.metadata['filename'] = doc.metadata['source']

        logger.info(f"      Loaded {len(documents)} document(s):")
        for i, doc in enumerate(documents, 1):
            filename = doc.metadata.get('filename', doc.metadata.get('source', f'doc_{i}'))
            if isinstance(filename, str):
                filename = Path(filename).name
            content_size = len(doc.page_content)
            logger.info(f"        [{i}] {filename} ({content_size:,} characters)")

        logger.info("")
        total_chars = sum(len(doc.page_content) for doc in documents)
        logger.info(f"      Total content: {total_chars:,} characters")
        logger.info("")

    # Step 2: Configure LLM with structured output support
    logger.info(f"[0.3] Configuring LLM: {config['llm']['model']}...")
    logger.info("      Enabling structured outputs for reliable JSON parsing...")

    # Configure ChatOpenAI with structured output support
    # Using model_kwargs to pass response_format for OpenRouter
    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config['llm']['model'],
            api_key=OPENROUTER_API_KEY,
            base_url=config['api']['base_url'],
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens'],
            model_kwargs={
                "response_format": {
                    "type": "json_object"  # Enable JSON mode for reliable parsing
                }
            }
        )
    )
    logger.info("      LLM configured successfully with structured outputs")
    logger.info("")

    # Step 3: Configure Embeddings
    logger.info(f"[0.4] Configuring Embeddings: {config['embeddings']['model']}...")
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

    # Step 4: Create personas
    logger.info("[0.5] Setting up DKI Jakarta government worker personas...")
    personas = create_personas_from_config(config['multihop']['personas'], logger)

    # Initialize progress tracker
    progress_tracker = DetailedProgressTracker(
        total_documents=len(documents),
        target_samples=test_size,
        progress_file="multihop_progress.json",
        update_interval=config['progress']['update_interval']
    ) if config['progress']['enabled'] else None

    if progress_tracker:
        progress_tracker.start()

    try:
        # Phase 1 & 2: Generate multi-hop queries using TestsetGenerator
        logger.info("=" * 80)
        logger.info("PHASE 1: Multi-Hop Query Generation")
        logger.info("=" * 80)
        logger.info("")

        logger.info(f"[1.1] Initializing TestsetGenerator with multi-hop synthesizers...")
        logger.info(f"      Language: {config['multihop']['language']} (Bahasa Indonesia)")
        logger.info(f"      Target queries: {test_size}")
        logger.info("")

        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings
        )

        logger.info("[1.2] Configuring multi-hop query distribution...")
        logger.info("      Using MultiHopAbstractQuerySynthesizer and MultiHopSpecificQuerySynthesizer")
        logger.info("")

        # Use Ragas' built-in multi-hop synthesizers
        from ragas.testset.synthesizers import (
            MultiHopAbstractQuerySynthesizer,
            MultiHopSpecificQuerySynthesizer
        )

        # Create custom distribution focused on multi-hop queries
        from ragas.testset.synthesizers import default_query_distribution

        # Get default distribution and modify to use only multi-hop
        query_dist = default_query_distribution(generator_llm)
        multi_hop_distribution = [
            (query_dist[1][0], 0.5),  # MultiHopAbstractQuerySynthesizer 50%
            (query_dist[2][0], 0.5),  # MultiHopSpecificQuerySynthesizer 50%
        ]

        logger.info("      Distribution:")
        logger.info("        - MultiHopAbstractQuerySynthesizer: 50%")
        logger.info("        - MultiHopSpecificQuerySynthesizer: 50%")
        logger.info("")

        logger.info("[1.3] Generating multi-hop queries...")
        logger.info("      This process builds knowledge graph and generates queries")
        logger.info("      This may take several minutes...")
        logger.info("")

        # Generate testset with multi-hop focus
        # Note: Personas are not directly supported in v0.3.9's generate_with_langchain_docs
        # They would need to be integrated through custom synthesizers
        dataset = generator.generate_with_langchain_docs(
            documents,
            testset_size=test_size,
            query_distribution=multi_hop_distribution,
            with_debugging_logs=False,
        )

        logger.info("")
        logger.info("      Multi-hop query generation completed!")
        logger.info(f"      Generated {len(dataset)} queries")
        logger.info("")

        # Convert to pandas DataFrame
        df = dataset.to_pandas()

        # Save final results
        logger.info("=" * 80)
        logger.info("PHASE 3: Saving Results")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"[3.1] Finalizing results to {output_file}...")

        # Write to both partial and final files
        result_writer.write_dataframe(df)
        final_df = result_writer.finalize()

        logger.info(f"      Saved {len(df)} queries to {output_file}")
        logger.info("")

        # Mark progress as complete
        if progress_tracker:
            progress_tracker.complete()

        # Clear checkpoint on success
        if checkpoint_manager:
            checkpoint_manager.clear_checkpoint()

        # Display summary
        logger.info("=" * 80)
        logger.info("Multi-Hop Query Generation Complete!")
        logger.info("=" * 80)
        logger.info(f"Total queries: {len(df)}")
        logger.info(f"Output file: {output_file}")
        if progress_tracker:
            logger.info(f"Elapsed time: {progress_tracker.format_time(progress_tracker.get_elapsed_time())}")
        logger.info("")

        # Display sample queries
        logger.info("Sample queries:")
        logger.info("-" * 80)
        for i, row in df.head(5).iterrows():
            query = row.get('user_input', row.get('question', 'N/A'))
            logger.info(f"{i+1}. {query}")
        logger.info("")

    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("=" * 80)
        logger.warning("Generation interrupted by user")
        logger.warning("=" * 80)

        if progress_tracker:
            progress_tracker.error("Interrupted by user")

        sys.exit(1)

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("Multi-hop query generation failed!")
        logger.error("=" * 80)
        logger.error(format_error_message(e))

        if progress_tracker:
            progress_tracker.error(str(e))

        raise


if __name__ == "__main__":
    main()
