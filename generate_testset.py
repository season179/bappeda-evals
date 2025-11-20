#!/usr/bin/env python3
"""
Synthetic Test Dataset Generation for RAG Evaluation using Ragas v0.2

This script generates a comprehensive synthetic evaluation dataset (Golden Quad format)
for RAG applications using the Ragas framework with OpenRouter models.

Models used:
- LLM: x-ai/grok-code-fast-1 (via OpenRouter)
- Embeddings: qwen/qwen3-embedding-8b (via OpenRouter)
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "x-ai/grok-code-fast-1"
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
KNOWLEDGE_DIR = "./knowledge-files"
TEST_SIZE = 100
OUTPUT_FILE = "synthetic_testset.csv"

def main():
    """Main function to generate synthetic testset"""

    print("=" * 80)
    print("Ragas v0.2 Synthetic Testset Generation for RAG Evaluation")
    print("=" * 80)
    print()

    # Validate API key
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        raise ValueError(
            "OPENROUTER_API_KEY not found. Please:\n"
            "1. Copy .env.example to .env\n"
            "2. Add your OpenRouter API key to .env\n"
            "3. Get your key from: https://openrouter.ai/settings/keys"
        )

    # Step 1: Load documents
    print(f"[1/6] Loading documents from {KNOWLEDGE_DIR}...")
    loader = DirectoryLoader(
        KNOWLEDGE_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True
    )
    documents = loader.load()

    # Ensure metadata includes filename for better traceability
    for doc in documents:
        if 'source' in doc.metadata and 'filename' not in doc.metadata:
            doc.metadata['filename'] = doc.metadata['source']

    print(f"      Loaded {len(documents)} markdown documents")
    print()

    # Step 2: Configure Generator LLM via OpenRouter
    print(f"[2/6] Configuring LLM (Generator): {LLM_MODEL}...")
    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.7,
        )
    )
    print(f"      LLM configured successfully")
    print()

    # Step 3: Configure Embeddings via OpenRouter
    print(f"[3/6] Configuring Embeddings: {EMBEDDING_MODEL}...")
    generator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    )
    print(f"      Embeddings configured successfully")
    print()

    # Step 4: Initialize TestsetGenerator
    print("[4/6] Initializing TestsetGenerator...")
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )
    print("      TestsetGenerator initialized")
    print()

    # Step 5: Create custom query distribution
    print("[5/6] Creating custom query distribution...")
    print("      Distribution strategy:")
    print("        - SingleHopSpecificQuerySynthesizer: 40% (simple queries)")
    print("        - MultiHopAbstractQuerySynthesizer: 35% (reasoning queries)")
    print("        - MultiHopSpecificQuerySynthesizer: 25% (multi-context queries)")
    print()

    # Get default distribution and customize weights
    query_distribution = default_query_distribution(generator_llm)

    # Customize the distribution weights
    # default_query_distribution returns: [(Synthesizer, weight), ...]
    # Default is typically: SingleHop=0.5, MultiHopAbstract=0.25, MultiHopSpecific=0.25
    # We want: SingleHop=0.4, MultiHopAbstract=0.35, MultiHopSpecific=0.25
    custom_distribution = [
        (query_distribution[0][0], 0.4),   # SingleHopSpecificQuerySynthesizer
        (query_distribution[1][0], 0.35),  # MultiHopAbstractQuerySynthesizer
        (query_distribution[2][0], 0.25),  # MultiHopSpecificQuerySynthesizer
    ]

    print(f"      Query distribution configured for {TEST_SIZE} samples")
    print()

    # Step 6: Generate testset
    print(f"[6/6] Generating synthetic testset ({TEST_SIZE} samples)...")
    print("      This may take several minutes depending on document size...")
    print()

    try:
        dataset = generator.generate_with_langchain_docs(
            documents,
            testset_size=TEST_SIZE,
            query_distribution=custom_distribution
        )

        print("      Testset generation completed successfully!")
        print()

        # Convert to pandas DataFrame
        df = dataset.to_pandas()

        # Save to CSV
        print(f"[OUTPUT] Saving testset to {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"         Saved {len(df)} samples to {OUTPUT_FILE}")
        print()

        # Display summary statistics
        print("=" * 80)
        print("Dataset Summary")
        print("=" * 80)
        print(f"Total samples generated: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print()

        # Display first 5 rows
        print("First 5 samples:")
        print("-" * 80)
        print(df.head().to_string())
        print()

        print("=" * 80)
        print("Generation Complete!")
        print("=" * 80)
        print(f"Dataset saved to: {OUTPUT_FILE}")
        print()

    except Exception as e:
        print(f"ERROR: Testset generation failed!")
        print(f"Error details: {str(e)}")
        raise

if __name__ == "__main__":
    main()
