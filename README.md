# RAG Evaluation with Ragas v0.2

Synthetic test dataset generation for RAG (Retrieval-Augmented Generation) applications using the Ragas framework and OpenRouter models.

## Overview

This project generates a comprehensive synthetic evaluation dataset (Golden Quad format) for testing and evaluating RAG systems. It uses:

- **Ragas v0.2**: Advanced testset generation framework
- **OpenRouter**: Unified API for accessing multiple LLM models
- **Grok Code Fast 1**: High-performance LLM for generation (x-ai/grok-code-fast-1)
- **Qwen3 Embedding 8B**: State-of-the-art embeddings (qwen/qwen3-embedding-8b)

## Features

- **Multiple Query Types**: Generates 100 synthetic test samples with balanced distribution:
  - 40% Single-hop specific queries (simple retrieval)
  - 35% Multi-hop abstract queries (reasoning)
  - 25% Multi-hop specific queries (multi-context retrieval)

- **Golden Quad Format**: Each sample includes:
  - `user_input`: The generated question
  - `reference`: Ground truth answer
  - `retrieved_contexts`: Relevant context passages

- **Indonesian Language Support**: Optimized for Bappeda DKI Jakarta documents

## Project Structure

```
rag_eval_ragas/
├── knowledge-files/          # Source documents (8 Markdown files)
│   ├── basis_pengetahuan_bappeda_dki_jakarta.md
│   ├── PERDA NO.1 TAHUN 2025.md
│   ├── PERDA NO.2 TAHUN 2025.md
│   ├── PERDA NO.3 TAHUN 2025.md
│   ├── RKPD 2025.md
│   ├── RPJMD 2025-2029.md
│   ├── RPJPD 2025-2045.md
│   └── LKPJ 2024.md
├── generate_testset.py       # Main generation script
├── pyproject.toml            # Project dependencies
├── .env.example              # Environment variable template
└── README.md                 # This file
```

## Setup

### 1. Install Dependencies

Using `uv` (recommended):
```bash
uv sync
```

Using `pip`:
```bash
pip install -e .
```

### 2. Configure API Key

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Get your OpenRouter API key from: https://openrouter.ai/settings/keys

3. Edit `.env` and add your API key:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

## Usage

### Generate Testset

Run the generation script:

```bash
python generate_testset.py
```

The script will:
1. Load all Markdown documents from `knowledge-files/`
2. Configure OpenRouter models (Grok LLM + Qwen3 embeddings)
3. Initialize Ragas TestsetGenerator
4. Generate 100 synthetic test samples with custom distribution
5. Save results to `synthetic_testset.csv`
6. Display first 5 samples in the console

### Expected Output

```
================================================================================
Ragas v0.2 Synthetic Testset Generation for RAG Evaluation
================================================================================

[1/6] Loading documents from ./knowledge-files...
      Loaded 8 markdown documents

[2/6] Configuring LLM (Generator): x-ai/grok-code-fast-1...
      LLM configured successfully

[3/6] Configuring Embeddings: qwen/qwen3-embedding-8b...
      Embeddings configured successfully

[4/6] Initializing TestsetGenerator...
      TestsetGenerator initialized

[5/6] Creating custom query distribution...
      Distribution strategy:
        - SingleHopSpecificQuerySynthesizer: 40% (simple queries)
        - MultiHopAbstractQuerySynthesizer: 35% (reasoning queries)
        - MultiHopSpecificQuerySynthesizer: 25% (multi-context queries)

[6/6] Generating synthetic testset (100 samples)...
      This may take several minutes depending on document size...

      Testset generation completed successfully!

[OUTPUT] Saving testset to synthetic_testset.csv...
         Saved 100 samples to synthetic_testset.csv

================================================================================
Dataset Summary
================================================================================
Total samples generated: 100
Columns: ['user_input', 'reference', 'retrieved_contexts']

First 5 samples:
[Sample data displayed here...]

================================================================================
Generation Complete!
================================================================================
Dataset saved to: synthetic_testset.csv
```

## Configuration

You can customize the generation by modifying these variables in `generate_testset.py`:

```python
TEST_SIZE = 100                              # Number of samples to generate
OUTPUT_FILE = "synthetic_testset.csv"        # Output filename
KNOWLEDGE_DIR = "./knowledge-files"          # Source documents directory
```

To adjust query distribution, modify the weights in the `custom_distribution` list:

```python
custom_distribution = [
    (query_distribution[0][0], 0.4),   # Single-hop: 40%
    (query_distribution[1][0], 0.35),  # Multi-hop abstract: 35%
    (query_distribution[2][0], 0.25),  # Multi-hop specific: 25%
]
```

## Output Format

The generated CSV file contains:

- **user_input**: Synthetic question generated from the documents
- **reference**: Ground truth answer/response
- **retrieved_contexts**: List of relevant context passages

This format is ideal for evaluating RAG systems using Ragas evaluation metrics.

## Models

### LLM: Grok Code Fast 1
- **Provider**: xAI (via OpenRouter)
- **Model ID**: `x-ai/grok-code-fast-1`
- **Context Window**: 256K tokens
- **Pricing**: $0.20 per million tokens
- **Use Case**: Question and answer generation

### Embeddings: Qwen3 Embedding 8B
- **Provider**: Qwen (via OpenRouter)
- **Model ID**: `qwen/qwen3-embedding-8b`
- **Performance**: #1 on MTEB multilingual leaderboard (score: 70.58)
- **Use Case**: Document embeddings and semantic search

## Troubleshooting

### API Key Error
```
ValueError: OPENROUTER_API_KEY not found
```
**Solution**: Ensure you've created `.env` file with a valid OpenRouter API key.

### Import Error
```
ModuleNotFoundError: No module named 'ragas'
```
**Solution**: Install dependencies using `uv sync` or `pip install -e .`

### Generation Takes Too Long
With 8 documents (>50K lines total), generation may take 10-30 minutes depending on:
- API rate limits
- Network speed
- Model availability

**Tip**: Start with a smaller `TEST_SIZE` (e.g., 10) to test the setup.

## Resources

- [Ragas Documentation](https://docs.ragas.io/)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Grok Model Information](https://openrouter.ai/x-ai/grok-code-fast-1)
- [Qwen3 Embeddings](https://openrouter.ai/qwen/qwen3-embedding-8b)

## License

This project is for evaluating RAG systems with Bappeda DKI Jakarta planning and budgeting documents.
