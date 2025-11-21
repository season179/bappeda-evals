#!/usr/bin/env python3
"""
Metadata Extraction for Disk-Based Processing

This script extracts and caches document metadata (headlines, keyphrases, summaries)
to minimize memory usage during knowledge graph construction.

Features:
- Processes documents one-at-a-time to minimize memory
- Extracts headlines, keyphrases, and summaries using LLM
- Caches results to disk (.cache/metadata/)
- Checkpoint/resume capability
- Progress tracking

Usage:
    python extract_metadata.py [--force] [--doc FILENAME]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lib import setup_logger, CheckpointManager

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


def extract_headlines(text: str, llm: ChatOpenAI, max_retries: int = 3) -> List[str]:
    """
    Extract headlines/section titles from document using LLM.

    Args:
        text: Document text
        llm: Language model instance
        max_retries: Maximum retry attempts on failure

    Returns:
        List of extracted headlines
    """
    prompt = f"""Extract the main section headlines and titles from this document.
Return ONLY a JSON array of strings, one per headline. Do not include any other text.

Document:
{text[:5000]}...

Example response format:
["Headline 1", "Headline 2", "Headline 3"]

JSON array of headlines:"""

    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Try to parse JSON response
            if content.startswith('[') and content.endswith(']'):
                headlines = json.loads(content)
                if isinstance(headlines, list):
                    return headlines

            # Fallback: split by newlines
            headlines = [line.strip('- "\'') for line in content.split('\n') if line.strip()]
            return headlines[:20]  # Limit to top 20 headlines

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Warning: Failed to extract headlines after {max_retries} attempts: {e}")
                return []
            print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")

    return []


def extract_keyphrases(text: str, llm: ChatOpenAI, max_phrases: int = 15) -> List[str]:
    """
    Extract keyphrases for relationship building.

    Args:
        text: Document text
        llm: Language model instance
        max_phrases: Maximum number of keyphrases to extract

    Returns:
        List of keyphrases
    """
    prompt = f"""Extract {max_phrases} key phrases/topics from this document that could be used to find relationships with other documents.
Focus on: programs, policies, locations, departments, themes, objectives.
Return ONLY a JSON array of strings.

Document:
{text[:5000]}...

Example response format:
["phrase 1", "phrase 2", "phrase 3"]

JSON array of keyphrases:"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        if content.startswith('[') and content.endswith(']'):
            keyphrases = json.loads(content)
            if isinstance(keyphrases, list):
                return keyphrases[:max_phrases]

        # Fallback
        keyphrases = [line.strip('- "\'') for line in content.split('\n') if line.strip()]
        return keyphrases[:max_phrases]

    except Exception as e:
        print(f"  Warning: Failed to extract keyphrases: {e}")
        return []


def create_summary(text: str, llm: ChatOpenAI, target_length: int = 500) -> str:
    """
    Create a brief summary of the document.

    Args:
        text: Full document text
        llm: Language model instance
        target_length: Target summary length in words

    Returns:
        Document summary
    """
    # For very large documents, use only first part for summary
    text_sample = text[:10000] if len(text) > 10000 else text

    prompt = f"""Create a concise summary of this document in approximately {target_length} words.
Focus on main topics, objectives, and key information.
Write in Bahasa Indonesia if the document is in Indonesian, otherwise in the source language.

Document:
{text_sample}

Summary:"""

    try:
        response = llm.invoke(prompt)
        summary = response.content.strip()
        return summary

    except Exception as e:
        print(f"  Warning: Failed to create summary: {e}")
        # Fallback: use first N characters
        return text[:3000] + "..."


def create_chunk_mappings(text: str, chunk_size: int = 50000, overlap: int = 2000) -> List[Dict[str, Any]]:
    """
    Create mappings from chunks to original document offsets.

    Args:
        text: Full document text
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks

    Returns:
        List of chunk mapping dictionaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)
    mappings = []
    current_pos = 0

    for i, chunk in enumerate(chunks):
        # Find chunk position in original text
        chunk_start = text.find(chunk[:100], current_pos)
        if chunk_start == -1:
            chunk_start = current_pos

        chunk_end = chunk_start + len(chunk)

        # Create summary snippet for this chunk
        chunk_summary = chunk[:200] + "..." if len(chunk) > 200 else chunk

        mappings.append({
            'chunk_id': f'chunk_{i}',
            'start': chunk_start,
            'end': chunk_end,
            'summary': chunk_summary
        })

        current_pos = chunk_end

    return mappings


def process_document(
    doc_path: Path,
    llm: ChatOpenAI,
    cache_dir: Path,
    logger
) -> bool:
    """
    Process a single document and cache its metadata.

    Args:
        doc_path: Path to document
        llm: Language model instance
        cache_dir: Directory for cached metadata
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    try:
        filename = doc_path.name
        cache_file = cache_dir / f"{doc_path.stem}.json"

        # Check if already cached
        if cache_file.exists():
            logger.info(f"  Already cached: {filename}")
            return True

        logger.info(f"  Processing: {filename}")

        # Load document
        loader = TextLoader(str(doc_path), encoding='utf-8')
        doc = loader.load()[0]
        text = doc.page_content
        char_count = len(text)

        logger.info(f"    Size: {char_count:,} characters ({char_count/1024/1024:.2f} MB)")

        # Extract metadata
        logger.info("    Extracting headlines...")
        headlines = extract_headlines(text, llm)
        logger.info(f"    Extracted {len(headlines)} headlines")

        logger.info("    Extracting keyphrases...")
        keyphrases = extract_keyphrases(text, llm)
        logger.info(f"    Extracted {len(keyphrases)} keyphrases")

        logger.info("    Creating summary...")
        summary = create_summary(text, llm)
        logger.info(f"    Created summary ({len(summary)} characters)")

        logger.info("    Creating chunk mappings...")
        chunk_mappings = create_chunk_mappings(text)
        logger.info(f"    Created {len(chunk_mappings)} chunk mappings")

        # Save to cache
        metadata = {
            'filename': filename,
            'metadata': {
                'source': str(doc_path),
                'original_size': char_count
            },
            'headlines': headlines,
            'keyphrases': keyphrases,
            'summary': summary,
            'char_count': char_count,
            'chunk_mappings': chunk_mappings
        }

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"    Cached to: {cache_file}")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"    Failed to process {doc_path.name}: {e}")
        return False


def main():
    """Main function to extract metadata from documents"""

    parser = argparse.ArgumentParser(
        description="Extract and cache document metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cached"
    )

    parser.add_argument(
        "--doc",
        type=str,
        help="Process only a specific document (filename)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Setup logging
    logger = setup_logger(
        log_dir=config['logging']['directory'],
        console_level=getattr(__import__('logging'), 'INFO'),
        file_level=getattr(__import__('logging'), config['logging']['file_level'])
    )

    logger.info("=" * 80)
    logger.info("Metadata Extraction for Disk-Based Processing")
    logger.info("=" * 80)
    logger.info("")

    # Validate API key
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        logger.error(
            "OPENROUTER_API_KEY not found. Please:\n"
            "1. Copy .env.example to .env\n"
            "2. Add your OpenRouter API key to .env"
        )
        sys.exit(1)

    # Setup cache directory
    cache_dir = Path('.cache/metadata')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Configure LLM
    logger.info(f"Configuring LLM: {config['llm']['model']}...")
    llm = ChatOpenAI(
        model=config['llm']['model'],
        api_key=OPENROUTER_API_KEY,
        base_url=config['api']['base_url'],
        temperature=0.3,  # Lower temperature for more consistent extraction
        max_tokens=2000
    )
    logger.info("LLM configured")
    logger.info("")

    # Get list of documents to process
    knowledge_dir = Path(config['knowledge_base']['directory'])
    glob_pattern = config['knowledge_base']['glob_pattern']

    if args.doc:
        # Process specific document
        doc_path = knowledge_dir / args.doc
        if not doc_path.exists():
            logger.error(f"Document not found: {doc_path}")
            sys.exit(1)
        documents_to_process = [doc_path]
    else:
        # Process all documents
        documents_to_process = sorted(knowledge_dir.glob(glob_pattern))

    logger.info(f"Found {len(documents_to_process)} document(s) to process")
    logger.info("")

    # Clear cache if force flag set
    if args.force:
        logger.info("Force flag set - clearing existing cache...")
        for cache_file in cache_dir.glob('*.json'):
            cache_file.unlink()
        logger.info("Cache cleared")
        logger.info("")

    # Process documents
    logger.info("=" * 80)
    logger.info("Processing Documents")
    logger.info("=" * 80)
    logger.info("")

    successful = 0
    failed = 0

    for i, doc_path in enumerate(documents_to_process, 1):
        logger.info(f"[{i}/{len(documents_to_process)}] {doc_path.name}")

        if process_document(doc_path, llm, cache_dir, logger):
            successful += 1
        else:
            failed += 1

    # Summary
    logger.info("=" * 80)
    logger.info("Metadata Extraction Complete")
    logger.info("=" * 80)
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info("")

    # Cache statistics
    cache_files = list(cache_dir.glob('*.json'))
    total_cache_size = sum(f.stat().st_size for f in cache_files)
    logger.info(f"Cached files: {len(cache_files)}")
    logger.info(f"Total cache size: {total_cache_size / 1024:.1f} KB")
    logger.info("")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
