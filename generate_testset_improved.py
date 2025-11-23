#!/usr/bin/env python3
"""
Improved Testset Generation for Large Government Documents.

This script handles large markdown documents (like LKPJ 2024.md with 28k+ lines)
using proper chunking strategies and generates both single-hop and multi-hop questions.

Key improvements:
1. Proper handling of large documents via chunking
2. Clear control over single-hop vs multi-hop question generation
3. Domain-specific transformations for government documents
4. Configurable chunk sizes optimized for structured reports
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.outputs import Generation, LLMResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import transformers for handling large documents
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
    OverlapScoreBuilder
)

# Import synthesizers for different question types
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer
)
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer
)
from ragas.testset.synthesizers.multi_hop.abstract import (
    MultiHopAbstractQuerySynthesizer
)

load_dotenv()


# ============================================================================
# OPENROUTER CHAT WRAPPER (same as in evaluate_factual_correctness_standalone.py)
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


def split_large_documents(
    docs: List,
    max_chars: int = 20000,
    chunk_size: int = 12000,
    chunk_overlap: int = 1000,
    verbose: bool = False
) -> List:
    """
    Split large documents into manageable chunks before processing.

    Uses RecursiveCharacterTextSplitter with markdown-aware separators to
    intelligently split documents that exceed the max_chars threshold.

    Args:
        docs: List of LangChain documents
        max_chars: Maximum characters before splitting (default: 20000)
        chunk_size: Target chunk size in characters (default: 12000)
        chunk_overlap: Overlap between chunks for context (default: 1000)
        verbose: Print detailed splitting information

    Returns:
        List of documents (large ones split into chunks, small ones unchanged)
    """
    if verbose:
        print(f"\n[0] Pre-processing large documents...")
        print(f"  → Splitting threshold: {max_chars:,} characters")
        print(f"  → Target chunk size: {chunk_size:,} characters")
        print(f"  → Chunk overlap: {chunk_overlap:,} characters")

    result_docs = []
    total_input_docs = len(docs)
    docs_split = 0
    total_chunks_created = 0

    # Markdown-aware separators (try headers first, then paragraphs, then sentences)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n# ",      # Markdown H1
            "\n## ",     # Markdown H2
            "\n### ",    # Markdown H3
            "\n\n",      # Paragraph breaks
            "\n",        # Line breaks
            ". ",        # Sentences
            " ",         # Words
            ""           # Characters (fallback)
        ],
        length_function=len,
        is_separator_regex=False
    )

    for doc in docs:
        doc_length = len(doc.page_content)
        source = doc.metadata.get('source', 'unknown')

        # Check if document needs splitting
        if doc_length > max_chars:
            if verbose:
                print(f"  → Splitting '{Path(source).name}': {doc_length:,} chars")

            # Split the document
            chunks = text_splitter.split_documents([doc])

            # Add metadata to track chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'original_source': source,
                    'original_length': doc_length
                })

            result_docs.extend(chunks)
            docs_split += 1
            total_chunks_created += len(chunks)

            if verbose:
                print(f"     ✓ Created {len(chunks)} chunks (avg {doc_length // len(chunks):,} chars/chunk)")
        else:
            # Document is small enough, keep as-is
            result_docs.append(doc)

    if verbose:
        print(f"\n✓ Pre-processing complete:")
        print(f"  - Input documents: {total_input_docs}")
        print(f"  - Documents split: {docs_split}")
        print(f"  - Total chunks created: {total_chunks_created}")
        print(f"  - Final document count: {len(result_docs)}")

    return result_docs


def create_knowledge_graph_from_docs(
    docs: List,
    llm,
    chunk_min_tokens: int = 300,
    chunk_max_tokens: int = 1000,
    max_headlines: int = 30,
    max_keyphrases: int = 15,
    doc_max_chars: int = 20000,
    doc_chunk_size: int = 12000,
    doc_chunk_overlap: int = 1000,
    verbose: bool = False
) -> KnowledgeGraph:
    """
    Create a knowledge graph from documents with proper chunking for large files.

    Args:
        docs: List of LangChain documents
        llm: LLM for extraction tasks
        chunk_min_tokens: Minimum chunk size (default: 300)
        chunk_max_tokens: Maximum chunk size (default: 1000)
        max_headlines: Max headlines to extract per document (default: 30)
        max_keyphrases: Max keyphrases to extract per chunk (default: 15)
        doc_max_chars: Max characters before splitting document (default: 20000)
        doc_chunk_size: Target chunk size when splitting (default: 12000)
        doc_chunk_overlap: Overlap between chunks (default: 1000)
        verbose: Print progress information

    Returns:
        KnowledgeGraph with enriched nodes and relationships
    """
    print(f"\n{'=' * 80}")
    print("BUILDING KNOWLEDGE GRAPH")
    print(f"{'=' * 80}")

    # Step 0: Pre-process large documents (split if needed)
    docs = split_large_documents(
        docs=docs,
        max_chars=doc_max_chars,
        chunk_size=doc_chunk_size,
        chunk_overlap=doc_chunk_overlap,
        verbose=verbose
    )

    # Step 1: Create initial knowledge graph
    if verbose:
        print("\n[1/6] Creating initial knowledge graph from documents...")

    kg = KnowledgeGraph()

    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata
                }
            )
        )

    print(f"✓ Created {len(kg.nodes)} document nodes")

    # Step 2: Extract headlines (sections/headings)
    # This is crucial for large government documents with structured sections
    if verbose:
        print(f"\n[2/6] Extracting headlines (max {max_headlines} per document)...")
        print("  → This identifies sections like 'BAB I', 'BAB II', etc.")

    headline_extractor = HeadlinesExtractor(
        llm=llm,
        max_num=max_headlines,  # For large docs, extract more headlines
        max_token_limit=8000  # Limit to first 8000 tokens to avoid timeouts on huge docs
    )

    # Step 3: Split documents by headlines into chunks
    if verbose:
        print(f"\n[3/6] Splitting documents into chunks ({chunk_min_tokens}-{chunk_max_tokens} tokens)...")
        print("  → This breaks large documents into manageable sections")

    headline_splitter = HeadlineSplitter(
        min_tokens=chunk_min_tokens,
        max_tokens=chunk_max_tokens
    )

    # Step 4: Extract keyphrases for relationship building
    if verbose:
        print(f"\n[4/6] Extracting keyphrases (max {max_keyphrases} per chunk)...")
        print("  → This identifies key concepts for connecting related chunks")

    keyphrase_extractor = KeyphrasesExtractor(
        llm=llm,
        max_num=max_keyphrases
    )

    # Step 5: Build relationships between chunks
    # This enables multi-hop question generation
    if verbose:
        print("\n[5/6] Building relationships between chunks...")
        print("  → This connects related sections (enables multi-hop questions)")

    overlap_builder = OverlapScoreBuilder(
        property_name="keyphrases"  # Connect chunks with similar keyphrases
    )

    # Step 6: Apply all transformations in sequence
    transforms = [
        headline_extractor,
        headline_splitter,
        keyphrase_extractor,
        overlap_builder
    ]

    if verbose:
        print("\n[6/6] Applying transformations to knowledge graph...")
    else:
        print("\nApplying transformations to knowledge graph...")
    apply_transforms(kg, transforms=transforms)

    # Count final nodes
    chunk_nodes = [n for n in kg.nodes if n.type == NodeType.CHUNK]
    print(f"\n✓ Knowledge graph built:")
    print(f"  - Total nodes: {len(kg.nodes)}")
    print(f"  - Chunk nodes: {len(chunk_nodes)}")
    print(f"  - Relationships: {len(kg.relationships)}")

    return kg


def generate_testset_with_hops(
    docs_path: str,
    testset_size: int = 50,
    output_file: str = "test_questions.jsonl",
    llm_model: str = "openai/gpt-5.1-codex-mini",
    embedding_model: str = "qwen/qwen3-embedding-8b",
    temperature: float = 0.0,
    api_base_url: str = "https://openrouter.ai/api/v1",
    # Question distribution
    single_hop_ratio: float = 0.5,
    multi_hop_specific_ratio: float = 0.25,
    multi_hop_abstract_ratio: float = 0.25,
    # Chunking parameters
    chunk_min_tokens: int = 300,
    chunk_max_tokens: int = 1000,
    # Document pre-processing (for very large files)
    doc_max_chars: int = 20000,
    doc_chunk_size: int = 12000,
    doc_chunk_overlap: int = 1000,
    verbose: bool = False
) -> List[Dict]:
    """
    Generate testset with explicit control over single-hop and multi-hop questions.

    Args:
        docs_path: Path to documents directory
        testset_size: Total number of questions to generate
        output_file: Output JSONL file path
        llm_model: LLM model for generation (OpenRouter format, e.g. openai/gpt-5.1-codex-mini)
        embedding_model: Embedding model (OpenRouter format, e.g. qwen/qwen3-embedding-8b)
        temperature: LLM temperature (0.0 = deterministic)
        api_base_url: OpenRouter API base URL
        single_hop_ratio: Ratio of single-hop questions (0.0-1.0)
        multi_hop_specific_ratio: Ratio of multi-hop specific questions
        multi_hop_abstract_ratio: Ratio of multi-hop abstract questions
        chunk_min_tokens: Minimum chunk size for document splitting
        chunk_max_tokens: Maximum chunk size for document splitting
        doc_max_chars: Max characters before splitting document (default: 20000)
        doc_chunk_size: Target chunk size when splitting (default: 12000)
        doc_chunk_overlap: Overlap between chunks (default: 1000)
        verbose: Print detailed progress

    Returns:
        List of generated test questions
    """
    # Validate ratios sum to 1.0
    total_ratio = single_hop_ratio + multi_hop_specific_ratio + multi_hop_abstract_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point error
        raise ValueError(f"Question ratios must sum to 1.0 (got {total_ratio})")

    print(f"\n{'=' * 80}")
    print("TESTSET GENERATION - LARGE DOCUMENT MODE")
    print(f"{'=' * 80}")
    print(f"\nConfiguration:")
    print(f"  Documents path: {docs_path}")
    print(f"  Testset size: {testset_size}")
    print(f"  LLM model: {llm_model}")
    print(f"  Embedding model: {embedding_model}")
    print(f"\nQuestion Distribution:")
    print(f"  Single-hop: {single_hop_ratio * 100:.0f}% (~{int(testset_size * single_hop_ratio)} questions)")
    print(f"    → Questions answerable from ONE document chunk")
    print(f"  Multi-hop (specific): {multi_hop_specific_ratio * 100:.0f}% (~{int(testset_size * multi_hop_specific_ratio)} questions)")
    print(f"    → Fact-based questions requiring MULTIPLE chunks")
    print(f"  Multi-hop (abstract): {multi_hop_abstract_ratio * 100:.0f}% (~{int(testset_size * multi_hop_abstract_ratio)} questions)")
    print(f"    → Interpretive questions synthesizing across chunks")
    print(f"\nChunking Strategy:")
    print(f"  Min tokens per chunk: {chunk_min_tokens}")
    print(f"  Max tokens per chunk: {chunk_max_tokens}")
    print(f"  → Optimized for large government documents")

    # Load documents
    print(f"\n{'=' * 80}")
    print("LOADING DOCUMENTS")
    print(f"{'=' * 80}")

    loader = DirectoryLoader(
        docs_path,
        glob="**/*.md",  # Only markdown files
        loader_cls=TextLoader,  # Use TextLoader to avoid unstructured dependency
        show_progress=True
    )
    docs = loader.load()

    if not docs:
        raise ValueError(f"No markdown documents found in {docs_path}")

    print(f"\n✓ Loaded {len(docs)} documents:")
    for doc in docs:
        filename = doc.metadata.get('source', 'unknown')
        content_length = len(doc.page_content)
        print(f"  - {Path(filename).name}: {content_length:,} characters")

    # Initialize LLM and embeddings
    print(f"\n{'=' * 80}")
    print("INITIALIZING MODELS")
    print(f"{'=' * 80}")

    # Get API configuration
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Set it in your .env file.")

    # Initialize OpenRouter-compatible LLM
    openrouter_llm = OpenRouterChatOpenAI(
        api_key=api_key,
        base_url=api_base_url,
        model=llm_model,
        temperature=temperature
    )

    # Wrap for Ragas compatibility
    generator_llm = LangchainLLMWrapper(openrouter_llm)
    print(f"✓ LLM initialized: {llm_model} (via OpenRouter)")

    # Initialize embeddings via OpenRouter
    # IMPORTANT: Ragas testset generation REQUIRES embeddings for building relationships
    # between chunks. OpenRouter now supports embeddings!
    generator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,  # Use same OpenRouter API key
            base_url=api_base_url  # Point to OpenRouter
        )
    )
    print(f"✓ Embeddings initialized: {embedding_model} (via OpenRouter)")

    # Build knowledge graph with chunking
    kg = create_knowledge_graph_from_docs(
        docs=docs,
        llm=generator_llm,
        chunk_min_tokens=chunk_min_tokens,
        chunk_max_tokens=chunk_max_tokens,
        doc_max_chars=doc_max_chars,
        doc_chunk_size=doc_chunk_size,
        doc_chunk_overlap=doc_chunk_overlap,
        verbose=verbose
    )

    # Configure query synthesizers
    print(f"\n{'=' * 80}")
    print("CONFIGURING QUESTION SYNTHESIZERS")
    print(f"{'=' * 80}")

    query_distribution = []

    # Single-hop synthesizer
    if single_hop_ratio > 0:
        single_hop = SingleHopSpecificQuerySynthesizer(
            llm=generator_llm,
            property_name="keyphrases"  # Use keyphrases to generate focused questions
        )
        query_distribution.append((single_hop, single_hop_ratio))
        print(f"✓ Single-hop synthesizer: {single_hop_ratio * 100:.0f}%")

    # Multi-hop specific synthesizer
    if multi_hop_specific_ratio > 0:
        multi_hop_specific = MultiHopSpecificQuerySynthesizer(
            llm=generator_llm
        )
        query_distribution.append((multi_hop_specific, multi_hop_specific_ratio))
        print(f"✓ Multi-hop (specific) synthesizer: {multi_hop_specific_ratio * 100:.0f}%")

    # Multi-hop abstract synthesizer
    if multi_hop_abstract_ratio > 0:
        multi_hop_abstract = MultiHopAbstractQuerySynthesizer(
            llm=generator_llm
        )
        query_distribution.append((multi_hop_abstract, multi_hop_abstract_ratio))
        print(f"✓ Multi-hop (abstract) synthesizer: {multi_hop_abstract_ratio * 100:.0f}%")

    # Initialize generator
    print(f"\n{'=' * 80}")
    print("GENERATING TESTSET")
    print(f"{'=' * 80}")
    print("This may take several minutes for large documents...\n")

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg
    )

    # Generate testset
    testset = generator.generate(
        testset_size=testset_size,
        query_distribution=query_distribution
    )

    print(f"\n✓ Generated {len(testset.samples)} test questions")

    # Convert to output format
    test_questions = []
    for i, sample in enumerate(testset.samples):
        test_questions.append({
            "query_id": i,
            "user_input": sample.user_input,
            "reference": sample.reference,
            "reference_contexts": sample.reference_contexts
        })

    # Save to JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for q in test_questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print(f"\n{'=' * 80}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"✓ Saved {len(test_questions)} questions to {output_path}")

    # Print examples
    print(f"\n{'=' * 80}")
    print("SAMPLE QUESTIONS")
    print(f"{'=' * 80}")

    for i, q in enumerate(test_questions[:3], 1):
        print(f"\nExample {i}:")
        print(f"  Question: {q['user_input']}")
        print(f"  Reference: {q['reference'][:200]}..." if len(q['reference']) > 200 else f"  Reference: {q['reference']}")

    return test_questions


def main():
    parser = argparse.ArgumentParser(
        description="Generate test questions from large government documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with balanced distribution
  %(prog)s knowledge-files/ --size 50

  # More single-hop questions (easier)
  %(prog)s knowledge-files/ --size 50 --single-hop 0.7 --multi-hop-specific 0.2 --multi-hop-abstract 0.1

  # Larger chunks for very long documents
  %(prog)s knowledge-files/ --size 100 --chunk-max 1500

  # Verbose mode to see detailed progress
  %(prog)s knowledge-files/ --size 50 --verbose
"""
    )

    parser.add_argument(
        "docs_path",
        type=str,
        help="Path to documents directory (markdown files)"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=50,
        help="Number of test questions to generate (default: 50)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="test_questions.jsonl",
        help="Output JSONL file (default: test_questions.jsonl)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5.1-codex-mini",
        help="LLM model for generation (default: openai/gpt-5.1-codex-mini)"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="qwen/qwen3-embedding-8b",
        help="Embedding model via OpenRouter (default: qwen/qwen3-embedding-8b)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0 for deterministic generation)"
    )

    parser.add_argument(
        "--api-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="OpenRouter API base URL (default: https://openrouter.ai/api/v1)"
    )

    # Question distribution
    parser.add_argument(
        "--single-hop",
        type=float,
        default=0.5,
        help="Ratio of single-hop questions (default: 0.5)"
    )

    parser.add_argument(
        "--multi-hop-specific",
        type=float,
        default=0.25,
        help="Ratio of multi-hop specific questions (default: 0.25)"
    )

    parser.add_argument(
        "--multi-hop-abstract",
        type=float,
        default=0.25,
        help="Ratio of multi-hop abstract questions (default: 0.25)"
    )

    # Chunking parameters
    parser.add_argument(
        "--chunk-min",
        type=int,
        default=300,
        help="Minimum tokens per chunk (default: 300)"
    )

    parser.add_argument(
        "--chunk-max",
        type=int,
        default=1000,
        help="Maximum tokens per chunk (default: 1000)"
    )

    # Document pre-processing parameters (for very large files)
    parser.add_argument(
        "--doc-max-chars",
        type=int,
        default=20000,
        help="Max characters before splitting document (default: 20000)"
    )

    parser.add_argument(
        "--doc-chunk-size",
        type=int,
        default=12000,
        help="Target chunk size when splitting large docs (default: 12000)"
    )

    parser.add_argument(
        "--doc-chunk-overlap",
        type=int,
        default=1000,
        help="Overlap between chunks for context (default: 1000)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    args = parser.parse_args()

    try:
        generate_testset_with_hops(
            docs_path=args.docs_path,
            testset_size=args.size,
            output_file=args.output,
            llm_model=args.model,
            embedding_model=args.embedding_model,
            temperature=args.temperature,
            api_base_url=args.api_base_url,
            single_hop_ratio=args.single_hop,
            multi_hop_specific_ratio=args.multi_hop_specific,
            multi_hop_abstract_ratio=args.multi_hop_abstract,
            chunk_min_tokens=args.chunk_min,
            chunk_max_tokens=args.chunk_max,
            doc_max_chars=args.doc_max_chars,
            doc_chunk_size=args.doc_chunk_size,
            doc_chunk_overlap=args.doc_chunk_overlap,
            verbose=args.verbose
        )

        print(f"\n{'=' * 80}")
        print("NEXT STEPS")
        print(f"{'=' * 80}")
        print("1. Validate the generated questions:")
        print(f"   python validate_data.py {args.output} --skip-response-check")
        print("\n2. Implement your RAG system in execute_rag.py")
        print("\n3. Execute questions through your RAG:")
        print(f"   python execute_rag.py {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
