# RAG Evaluation Testset Generator

Multi-hop query generation and API credit management tools for RAG (Retrieval-Augmented Generation) evaluation using Ragas framework and OpenRouter.

## Overview

This project provides tools for generating synthetic evaluation datasets for RAG systems focused on DKI Jakarta government planning documents. It uses:

- **Ragas v0.3.9**: Advanced testset generation framework with multi-hop query support
- **OpenRouter**: Unified API for accessing multiple LLM models
- **Gemini 2.5 Flash**: High-performance LLM for generation (google/gemini-2.5-flash)
- **Qwen3 Embedding 8B**: State-of-the-art embeddings (qwen/qwen3-embedding-8b)

## Architecture

**Two-Phase Disk-Based Processing** for handling large documents efficiently:

1. **Phase 1**: Extract metadata (headlines, keyphrases, summaries) from documents one-at-a-time and cache to disk
2. **Phase 2**: Build knowledge graph from lightweight summaries (28KB vs 12MB), preserving all cross-document relationships

This approach achieves **99.8% memory reduction** while maintaining full knowledge graph capabilities.

## Features

### Multi-Hop Query Generation
- **Disk-Based Processing**: 99.8% memory reduction (28KB vs 12MB) for large documents
- **Two-Phase Architecture**: Metadata extraction + knowledge graph generation
- **Knowledge Graph Construction**: Builds relationships across all documents
- **Multi-Hop Synthesizers**: Generates queries requiring cross-document reasoning
  - MultiHopAbstractQuerySynthesizer (50%)
  - MultiHopSpecificQuerySynthesizer (50%)
- **DKI Jakarta Personas**: Government worker roles for contextual queries
- **Bahasa Indonesia**: Native language support
- **Checkpoint/Resume**: Robust progress tracking and recovery
- **Comprehensive Logging**: API calls, progress, and error tracking

### Translation (Optional)
- Translate English queries to Bahasa Indonesia
- Preserves technical terms and regulatory names
- Resume support with progress caching
- Smart question-only translation (prevents LLM from answering)

### Account Status Monitoring
- Check OpenRouter credit balance and limits
- Monitor rate limits and usage statistics
- Track daily, weekly, and monthly API usage
- View BYOK (Bring Your Own Key) usage
- Validate account tier and limit reset schedules

## Project Structure

```
rag_eval_ragas/
├── extract_metadata.py            # Phase 1: Extract & cache metadata
├── generate_multihop_testset.py  # Phase 2: Generate queries from cache
├── translate_user_input.py        # Optional: Translate queries to Indonesian
├── check_openrouter_status.py     # OpenRouter account status (credits, rate limits, usage)
├── config.yaml                    # Configuration file
├── knowledge-files/               # Source documents (8 documents)
│   ├── LKPJ 2024.md              # 8.8MB - Large document
│   ├── RKPD 2025.md
│   ├── RPJMD 2025-2029.md
│   └── ...
├── .cache/metadata/               # Cached metadata (139KB total)
│   ├── LKPJ 2024.json            # Headlines, keyphrases, summaries
│   └── ...
├── lib/                           # Utility modules
│   ├── __init__.py
│   ├── metadata_loader.py        # Disk-based document loading
│   ├── api_validator.py          # API connectivity validation
│   ├── logger.py                 # Multi-file logging setup
│   ├── progress_tracker.py       # Real-time progress tracking
│   ├── result_writer.py          # Incremental CSV writing
│   └── state_manager.py          # Checkpoint management
├── pyproject.toml                # Dependencies
└── .env.example                  # Environment variable template
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

### Quick Start

```bash
# 1. Check your account status (optional)
uv run python check_openrouter_status.py

# 2. Extract metadata from documents (run once, ~3 minutes)
uv run python extract_metadata.py

# 3. Generate multi-hop queries
uv run python generate_multihop_testset.py --size 10

# 4. Optional: Translate queries to Indonesian (if needed)
uv run python translate_user_input.py --input multihop_testset.csv
```

### Step 1: Check Account Status (Optional)

```bash
uv run python check_openrouter_status.py
```

**Output:**
```
============================================================
OpenRouter Account Status
============================================================

📋 Account Overview
------------------------------------------------------------
API Key Label: sk-or-v1-...
Account Type: Paid Account

💳 Credit Limits & Balance
------------------------------------------------------------
Credit Cap: Unlimited
Credits Remaining: Unlimited
Limit Reset: Never

📊 Usage Statistics
------------------------------------------------------------
All-Time Usage: $23.01
Daily Usage: $2.94
Weekly Usage: $8.95
Monthly Usage: $9.87

============================================================
```

### Step 2: Extract Metadata (Run Once)

Extract and cache metadata from all documents:

```bash
uv run python extract_metadata.py
```

**What this does:**
- Processes documents one-at-a-time (low memory)
- Extracts headlines, keyphrases, summaries
- Creates chunk mappings for large documents
- Caches to `.cache/metadata/` (~139KB for 8 documents)
- **Run time**: ~3 minutes for 12MB of documents

**Options:**
```bash
# Process specific document only
uv run python extract_metadata.py --doc "LKPJ 2024.md"

# Force re-extraction (clear cache)
uv run python extract_metadata.py --force
```

### Step 3: Generate Multi-Hop Queries

Generate queries using cached metadata:

```bash
uv run python generate_multihop_testset.py --size 10
```

**Options:**
```bash
# Validate API without generating
uv run python generate_multihop_testset.py --validate-api

# Resume from last checkpoint
uv run python generate_multihop_testset.py --resume

# Generate specific number of queries (overrides config)
uv run python generate_multihop_testset.py --size 50

# Custom output file
uv run python generate_multihop_testset.py --output my_testset.csv

# Enable verbose console logging
uv run python generate_multihop_testset.py --verbose
```

### Optional: Translate Queries to Indonesian

If the generated queries are in English (or mixed language), translate them to proper Bahasa Indonesia:

```bash
uv run python translate_user_input.py --input multihop_testset.csv --output multihop_translated.csv
```

**What this does:**
- Translates `user_input` column from English to Indonesian
- Preserves technical terms (RPJMD, RKPD, APBD, etc.)
- Maintains question format and structure
- Uses checkpoint/resume for interrupted translations
- Caches translations in `translation_progress.json`

**Features:**
- **Smart translation**: Preserves regulatory names and technical terms
- **Resume support**: Automatically resumes from last translated row
- **Progress tracking**: Shows translation progress with caching
- **Error recovery**: Falls back to original text on translation errors

**Example translation:**
```
Input:  "What does RPJMD say about regional development?"
Output: "Apa yang diatur dalam RPJMD tentang pembangunan daerah?"
```

**Note**: The script prevents the LLM from answering questions - it only translates the question structure.

### Expected Output

**Phase 1 - Metadata Extraction:**
```
[1/8] LKPJ 2024.md
  Processing: LKPJ 2024.md
    Size: 8,828,706 characters (8.42 MB)
    Extracting headlines...
    ✓ Extracted 6 headlines
    Extracting keyphrases...
    ✓ Extracted 15 keyphrases
    Creating summary...
    ✓ Created summary (3,520 characters)
    Creating chunk mappings...
    ✓ Created 241 chunk mappings
    ✓ Cached to: .cache/metadata/LKPJ 2024.json

================================================================================
Metadata Extraction Complete
================================================================================
Successful: 8
Cache size: 139.1 KB
```

**Phase 2 - Query Generation:**
```
================================================================================
PHASE 0: Document Loading
================================================================================

[0.1] Loading documents from metadata cache...
      Cache directory: .cache/metadata
      Cached files: 8
      Cache size: 139.1 KB

      Total original size: 12,594,595 characters (12.0 MB)
      Total summary size: 28,594 characters (27.9 KB)
      Memory reduction: ~99.8%

[0.2] Converting to LangChain documents...
      ✓ Converted 8 documents
      Using summaries for knowledge graph (low memory mode)

================================================================================
PHASE 1: Multi-Hop Query Generation
================================================================================

      Multi-hop query generation completed!
      Generated 10 queries

================================================================================
Multi-Hop Query Generation Complete!
================================================================================
Total queries: 10
Output file: multihop_testset.csv
Elapsed time: 3m 16s
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

translation:
  model: "x-ai/grok-code-fast-1"  # Model for translating queries
  temperature: 0.3                # Lower = more consistent translations
  max_tokens: 500                 # Max tokens for translation
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

### Missing Metadata Cache
```
⚠ No cached metadata found!
```
**Solution**: Run metadata extraction first:
```bash
uv run python extract_metadata.py
```

### TypeError: 'NoneType' object is not iterable
This error occurred in older versions when processing large documents. The new disk-based architecture fixes this by:
- Extracting metadata one document at a time
- Using summaries instead of full documents for knowledge graph
- Reducing memory usage by 99.8%

**Solution**: Use the two-phase workflow (extract metadata → generate queries).

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
1. Run `python check_openrouter_status.py` to check balance and usage
2. Add credits at https://openrouter.ai/credits
3. Or use a different API key

### Generation Interrupted
If query generation is interrupted, use `--resume`:
```bash
uv run python generate_multihop_testset.py --resume
```

**Note**: Metadata extraction does not support resume. If interrupted, simply re-run - it will skip already cached files.

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
