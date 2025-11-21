# RAG Evaluation Testset Generator

Multi-hop query generation and API credit management tools for RAG (Retrieval-Augmented Generation) evaluation using Ragas framework and OpenRouter.

## Overview

This project provides tools for generating synthetic evaluation datasets for RAG systems focused on DKI Jakarta government planning documents. It uses:

- **Ragas v0.3.9**: Advanced testset generation framework with multi-hop query support
- **OpenRouter**: Unified API for accessing multiple LLM models
- **Gemini 2.5 Flash**: High-performance LLM for generation (google/gemini-2.5-flash)
- **Qwen3 Embedding 8B**: State-of-the-art embeddings (qwen/qwen3-embedding-8b)

## Features

### Multi-Hop Query Generation
- **Knowledge Graph Construction**: Builds relationships between documents
- **Multi-Hop Synthesizers**: Generates queries requiring cross-document reasoning
  - MultiHopAbstractQuerySynthesizer (50%)
  - MultiHopSpecificQuerySynthesizer (50%)
- **DKI Jakarta Personas**: Government worker roles for contextual queries
- **Bahasa Indonesia**: Native language support
- **Checkpoint/Resume**: Robust progress tracking and recovery
- **Comprehensive Logging**: API calls, progress, and error tracking

### Credit Management
- Check OpenRouter account balance
- Monitor API usage and remaining credits
- Validate API connectivity before generation

## Project Structure

```
rag_eval_ragas/
├── generate_multihop_testset.py  # Main testset generation script
├── check_credits.py               # OpenRouter credit checker
├── config.yaml                    # Configuration file
├── knowledge-files/               # Source documents (3 Perda files)
│   ├── PERDA NO.1 TAHUN 2025.md
│   ├── PERDA NO.2 TAHUN 2025.md
│   └── PERDA NO.3 TAHUN 2025.md
├── lib/                           # Utility modules
│   ├── __init__.py
│   ├── api_validator.py          # API connectivity validation
│   ├── error_handlers.py         # Error formatting
│   ├── logger.py                 # Multi-file logging setup
│   ├── progress_tracker.py       # Real-time progress tracking
│   ├── result_writer.py          # Incremental CSV writing
│   └── state_manager.py          # Checkpoint management
├── pyproject.toml                # Dependencies
├── .env.example                  # Environment variable template
└── README.md                     # This file
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

### Check API Credits

Before generating queries, check your OpenRouter credit balance:

```bash
python check_credits.py
```

**Output:**
```
==================================================
OpenRouter Credit Balance
==================================================
Total Credits: $10.00
Total Usage: $2.34
Remaining Balance: $7.66
==================================================
```

### Generate Multi-Hop Testset

Run the generation script:

```bash
python generate_multihop_testset.py
```

**Options:**
```bash
# Validate API without generating
python generate_multihop_testset.py --validate-api

# Resume from last checkpoint
python generate_multihop_testset.py --resume

# Clear checkpoint and start fresh
python generate_multihop_testset.py --reset

# Custom configuration file
python generate_multihop_testset.py --config custom_config.yaml

# Generate specific number of queries (overrides config)
python generate_multihop_testset.py --size 50

# Custom output file
python generate_multihop_testset.py --output my_testset.csv

# Enable verbose console logging
python generate_multihop_testset.py --verbose

# Disable checkpointing
python generate_multihop_testset.py --no-checkpoint
```

### Expected Output

```
================================================================================
Multi-Hop Query Generation using Knowledge Graph
================================================================================

This script generates queries that require synthesizing information
across multiple DKI Jakarta government planning documents.

Configuration:
  Target queries: 20
  Language: id
  Overlap threshold: 0.6
  Max keyphrases: 10

================================================================================
PHASE 0: Document Loading
================================================================================

[0.1] Loading documents from ./knowledge-files...
      Loaded 3 document(s):
        [1] PERDA NO.1 TAHUN 2025.md (16,051 characters)
        [2] PERDA NO.2 TAHUN 2025.md (13,320 characters)
        [3] PERDA NO.3 TAHUN 2025.md (23,874 characters)

      Total content: 53,245 characters

[0.2] Configuring LLM: google/gemini-2.5-flash...
      Enabling structured outputs for reliable JSON parsing...
      LLM configured successfully with structured outputs

[0.3] Configuring Embeddings: qwen/qwen3-embedding-8b...
      Embeddings configured successfully

[0.4] Setting up DKI Jakarta government worker personas...
      Created 5 personas

================================================================================
PHASE 1: Multi-Hop Query Generation
================================================================================

[1.1] Initializing TestsetGenerator with multi-hop synthesizers...
      Language: id (Bahasa Indonesia)
      Target queries: 20

[1.2] Configuring multi-hop query distribution...
      Using MultiHopAbstractQuerySynthesizer and MultiHopSpecificQuerySynthesizer

      Distribution:
        - MultiHopAbstractQuerySynthesizer: 50%
        - MultiHopSpecificQuerySynthesizer: 50%

[1.3] Generating multi-hop queries...
      This process builds knowledge graph and generates queries
      This may take several minutes...

      Multi-hop query generation completed!
      Generated 20 queries

================================================================================
PHASE 3: Saving Results
================================================================================

[3.1] Finalizing results to multihop_testset.csv...
      Saved 20 queries

================================================================================
Multi-Hop Query Generation Complete!
================================================================================
Total queries: 20
Output file: multihop_testset.csv
Elapsed time: 5m 23s

Sample queries:
--------------------------------------------------------------------------------
1. Bagaimana ketentuan tentang pengelolaan keuangan daerah...
2. Apa hubungan antara RPJMD dan pelaksanaan program pembangunan...
...
```

## Configuration

Edit `config.yaml` to customize:

### API Settings
```yaml
api:
  base_url: "https://openrouter.ai/api/v1"
```

### Model Configuration
```yaml
llm:
  model: "google/gemini-2.5-flash"
  temperature: 0.7
  max_tokens: null

embeddings:
  model: "qwen/qwen3-embedding-8b"
```

### Generation Parameters
```yaml
multihop:
  test_size: 20               # Number of queries to generate
  language: "id"              # Bahasa Indonesia
  overlap_threshold: 0.6      # Keyphrase overlap for relationships
  max_keyphrases: 10          # Keyphrases per document
```

### Output & Logging
```yaml
output:
  partial_file: "testset_partial.csv"
  final_file: "testset_final.csv"
  backup_enabled: true

logging:
  directory: "logs"
  console_level: "INFO"
  file_level: "DEBUG"
  api_log_enabled: true
  error_log_enabled: true
```

### Progress Tracking
```yaml
checkpoint:
  enabled: true
  file: "checkpoint.json"

progress:
  enabled: true
  file: "progress_summary.json"
  update_interval: 30  # seconds
```

## Output Format

The generated CSV file (`multihop_testset.csv`) contains:

- **user_input**: The generated question in Bahasa Indonesia
- **reference**: Ground truth answer
- **retrieved_contexts**: List of relevant context passages from documents

This format is compatible with Ragas evaluation metrics for RAG systems.

## DKI Jakarta Government Personas

The generator uses 5 authentic government worker personas:

1. **Perencana Pembangunan Daerah** - Regional development planner at Bappeda
2. **Analis Anggaran Daerah** - Budget analyst at financial management agency
3. **Kepala Seksi Perencanaan** - Planning section head at Bappeda
4. **Peneliti Kebijakan Publik** - Public policy researcher
5. **Koordinator Program Pembangunan** - Development program coordinator

These personas ensure queries are contextually relevant to actual government planning workflows.

## Logging

The system generates multiple log files in the `logs/` directory:

- `main_YYYYMMDD_HHMMSS.log` - Main application log (DEBUG level)
- `api_YYYYMMDD_HHMMSS.log` - All API calls and responses
- `errors_YYYYMMDD_HHMMSS.log` - Errors and exceptions only

Progress is also tracked in `multihop_progress.json` with real-time updates.

## Troubleshooting

### API Key Error
```
OPENROUTER_API_KEY not found
```
**Solution**: Ensure `.env` file exists with valid OpenRouter API key.

### Model Not Found (404)
```
NotFoundError: The specified model was not found
```
**Solution**:
1. Verify model name in `config.yaml`
2. Check model availability at https://openrouter.ai/models
3. Ensure your API key has access to the model

### Rate Limit Error (429)
```
RateLimitError: Rate limit exceeded
```
**Solution**:
1. Wait a few minutes before retrying
2. Reduce `test_size` in config.yaml
3. Consider upgrading OpenRouter plan

### Insufficient Credits
```
AuthenticationError: Insufficient credits
```
**Solution**:
1. Run `python check_credits.py` to check balance
2. Add credits at https://openrouter.ai/credits
3. Or use a different API key

### Generation Interrupted
If generation is interrupted, use `--resume` to continue from checkpoint:
```bash
python generate_multihop_testset.py --resume
```

## Models

### LLM: Google Gemini 2.5 Flash
- **Model ID**: `google/gemini-2.5-flash`
- **Context Window**: 1M tokens
- **Pricing**: Competitive rates via OpenRouter
- **Use Case**: Question and answer generation with structured output support

### Embeddings: Qwen3 Embedding 8B
- **Model ID**: `qwen/qwen3-embedding-8b`
- **Performance**: #1 on MTEB multilingual leaderboard (score: 70.58)
- **Dimensions**: 8192
- **Use Case**: Document embeddings and semantic similarity

## Resources

- [Ragas Documentation](https://docs.ragas.io/)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Gemini Models](https://openrouter.ai/google/gemini-2.5-flash)
- [Qwen3 Embeddings](https://openrouter.ai/qwen/qwen3-embedding-8b)
- [OpenRouter API Keys](https://openrouter.ai/settings/keys)
- [Check Credits](https://openrouter.ai/credits)

## License

This project is for evaluating RAG systems with Bappeda DKI Jakarta planning and regulatory documents (Peraturan Daerah).
