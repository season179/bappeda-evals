# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG evaluation system for DKI Jakarta government planning documents using Ragas framework. The project consists of three main workflows:

1. **Testset Generation**: Generate synthetic multi-hop queries from knowledge documents
2. **RAG Execution**: Execute queries against SmartKnowledge RAG application and capture results
3. **Ragas Evaluation**: Evaluate RAG performance using multiple Ragas metrics

## Common Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Check OpenRouter account status (credits, rate limits, usage)
uv run python check_openrouter_status.py
```

### Testset Generation Workflow
```bash
# Phase 1: Extract and cache metadata from documents (run once)
uv run python extract_metadata.py
uv run python extract_metadata.py --force  # Re-extract all
uv run python extract_metadata.py --doc "LKPJ 2024.md"  # Specific document

# Phase 2: Generate multi-hop queries
uv run python generate_multihop_testset.py --size 10
uv run python generate_multihop_testset.py --resume  # Resume from checkpoint
uv run python generate_multihop_testset.py --validate-api  # Test API without generating

# Optional: Translate queries to Bahasa Indonesia
uv run python translate_user_input.py --input multihop_testset.csv
```

### RAG Execution and Evaluation Workflow
```bash
# Execute queries against SmartKnowledge API
uv run python run_rag_executor.py --input multihop_translated.csv
uv run python run_rag_executor.py --resume  # Resume from checkpoint
uv run python run_rag_executor.py --limit 10  # Test with limited queries

# Run Ragas evaluation (--metric and --input are required)
uv run python run_ragas_evaluation.py --metric faithfulness --input results/rag_execution_250122_143022.jsonl
uv run python run_ragas_evaluation.py --metric context_precision --input results/rag_execution_250122_143022.jsonl --resume  # Resume from checkpoint
uv run python run_ragas_evaluation.py --metric answer_relevancy --input results/rag_execution_250122_143022.jsonl --skip-failed  # Exclude queries without contexts

# Run Factual Correctness evaluation (optimized, timeout-resistant)
uv run python evaluate_factual_correctness.py --input results/rag_execution_250122_143022.jsonl
uv run python evaluate_factual_correctness.py --input results/rag_execution_250122_143022.jsonl --mode precision  # Faster: only check response claims
uv run python evaluate_factual_correctness.py --input results/rag_execution_250122_143022.jsonl --resume  # Resume from checkpoint
uv run python evaluate_factual_correctness.py --input results/rag_execution_250122_143022.jsonl --resume --retry-failed  # Retry timeout/failed samples
uv run python evaluate_factual_correctness.py --input results/rag_execution_250122_143022.jsonl --timeout 300  # Custom timeout (5 min)
```

## Architecture

### Two-Phase Disk-Based Processing

The testset generation uses a two-phase architecture for memory efficiency:

**Phase 1: Metadata Extraction** (`extract_metadata.py`)
- Processes documents one-at-a-time (low memory footprint)
- Extracts headlines, keyphrases, summaries, and chunk mappings
- Caches metadata to `.cache/metadata/` (~139KB for 12MB of documents)
- Achieves 99.8% memory reduction

**Phase 2: Query Generation** (`generate_multihop_testset.py`)
- Loads lightweight summaries from metadata cache
- Builds knowledge graph from summaries (preserves cross-document relationships)
- Generates multi-hop queries using:
  - `MultiHopAbstractQuerySynthesizer` (50%)
  - `MultiHopSpecificQuerySynthesizer` (50%)

### RAG Execution Pipeline

**Executor** (`run_rag_executor.py`)
- Calls SmartKnowledge API at `http://localhost:3000/api/chat-complete`
- Extracts retrieved contexts from tool executions (search tools only)
- Outputs to JSONL format (one JSON object per line)
- Supports checkpoint/resume for interrupted executions

**Expected Tool Names**:
- `hybrid_search`
- `vector_search`
- `full_text_search`
- `get_contextual_chunks`

### Ragas Evaluation System

**Multi-Metric Evaluator** (`run_ragas_evaluation.py`)
- Transforms executor JSONL to Ragas Dataset format
- Evaluates using a single metric per run (specified via `--metric` argument)
- Generates detailed results (JSONL) and markdown report
- Supports all Ragas metrics including context-based metrics

**Available Metrics**:
- `context_precision`: Precision of retrieved contexts
- `context_recall`: Recall of reference contexts
- `context_entity_recall`: Entity-level recall
- `answer_relevancy`: Relevance of answer to question
- `faithfulness`: Answer grounded in contexts
- `answer_correctness`: Correctness vs reference answer
- `answer_similarity`: Semantic similarity to reference
- `context_utilization`: How well contexts are used

**Factual Correctness Evaluator** (`evaluate_factual_correctness.py`)
- **Optimized single-metric script** for evaluating factual accuracy
- **Timeout-resistant design**: Individual 600s timeout per sample, low atomicity/coverage
- **Minimal data requirements**: Only needs `user_input`, `response`, `reference` (no contexts)
- **Performance features**:
  - One-by-one sample processing (isolates failures)
  - Low atomicity/coverage settings (30-50% fewer claims)
  - Timeout correlation analysis (identifies problematic sample lengths)
  - Incremental JSONL writing (immediate save after each sample)
- **Three scoring modes**: F1 (default), precision-only (50% faster), recall-only
- **Output**: Precision, recall, F1 scores + processing time + timeout analysis
- **Use when**: Experiencing timeout issues with multi-metric evaluation

### OpenRouter Integration

**Custom LLM Wrapper** (`lib/openrouter_chat.py`)
- Extends `ChatOpenAI` to handle OpenRouter-specific requirements
- Overrides `_generate()` to bypass the `n` parameter (unsupported by OpenRouter)
- Used with `LangchainLLMWrapper(bypass_n=True)` for Ragas compatibility

**Models Used**:
- LLM: `x-ai/grok-code-fast-1` (default) or `google/gemini-2.5-flash`
- Embeddings: `qwen/qwen3-embedding-8b` (required)

## Library Modules (`lib/`)

Core utility modules for robust, resumable, and observable operations:

- **`logger.py`**: Multi-file logging setup (main, API, errors)
- **`state_manager.py`**: Checkpoint management for resume functionality
- **`rag_client.py`**: HTTP client for SmartKnowledge API with retry logic
- **`ragas_evaluator.py`**: Ragas evaluation wrapper with custom LLM/embeddings
- **`data_transformer.py`**: Transform executor JSONL to Ragas Dataset format
- **`report_generator.py`**: Generate markdown evaluation reports
- **`result_writer.py`**: Incremental CSV/JSONL writing
- **`openrouter_chat.py`**: Custom ChatOpenAI wrapper for OpenRouter compatibility
- **`api_validator.py`**: Pre-flight API connectivity validation
- **`metadata_loader.py`**: Disk-based document metadata loading
- **`progress_tracker.py`**: Real-time progress tracking

## Configuration (`config.yaml`)

### Key Sections

**API Configuration**:
```yaml
api:
  base_url: "https://openrouter.ai/api/v1"
```

**Model Selection**:
```yaml
llm:
  model: "x-ai/grok-code-fast-1"
embeddings:
  model: "qwen/qwen3-embedding-8b"  # Required - do not change
```

**RAG Executor**:
```yaml
rag_executor:
  api_base_url: "http://localhost:3000"  # SmartKnowledge endpoint
  timeout_seconds: 120
  max_retries: 3
```

**Ragas Evaluation**:
```yaml
ragas_evaluation:
  llm:
    model: "x-ai/grok-code-fast-1"  # Can differ from main LLM
  # Note: metrics are specified via --metric argument, not in config
  llm_timeout_seconds: 300
  embeddings_timeout_seconds: 180
```

**Factual Correctness Evaluation**:
```yaml
factual_correctness_evaluation:
  llm:
    model: "x-ai/grok-code-fast-1"
  mode: "F1"              # F1 | precision | recall
  atomicity: "low"        # low (faster) | high (precise)
  coverage: "low"         # low (faster) | high (comprehensive)
  sample_timeout_seconds: 600   # 10 min per sample
  max_sample_attempts: 3
```

## Data Flow

### File Formats and Locations

**Testset Generation**:
- Input: `knowledge-files/*.md` (8 documents, 12MB total)
- Cache: `.cache/metadata/*.json` (139KB)
- Output: `multihop_testset.csv` → `multihop_translated.csv`

**RAG Execution**:
- Input: `multihop_translated.csv`
- Output: `results/rag_execution_YYMMDD_HHMMSS.jsonl` (timestamped JSONL format)
- Checkpoint: `executor_checkpoint_YYMMDD_HHMMSS.json` (timestamped)

**Ragas Evaluation**:
- Input: `results/rag_execution_YYMMDD_HHMMSS.jsonl` (from executor)
- Output:
  - `results/ragas_eval_detailed.jsonl` (per-query scores)
  - `results/ragas_eval_report.md` (summary report)
- Checkpoint: `ragas_checkpoint.json`

**Factual Correctness Evaluation**:
- Input: `results/rag_execution_YYMMDD_HHMMSS.jsonl` (from executor)
- Output:
  - `results/factual_correctness_detailed.jsonl` (per-query scores)
  - `results/factual_correctness_report.md` (summary + timeout analysis)
- Checkpoint: `factual_correctness_checkpoint.json`

### JSONL Schema

**Executor Output** (`rag_execution_YYMMDD_HHMMSS.jsonl`):
```json
{
  "query_id": 0,
  "user_input": "Query text",
  "reference_contexts": "[\"Context 1\", \"Context 2\"]",
  "reference": "Reference answer",
  "actual_contexts": ["Retrieved context 1", "Retrieved context 2"],
  "actual_answer": "Generated answer",
  "tool_calls": [...],
  "api_latency_ms": 1234,
  "status": "SUCCESS|FAILED",
  "error": ""
}
```

**Ragas Detailed Output** (`ragas_eval_detailed.jsonl`):
```json
{
  "user_input": "Query text",
  "reference_contexts": ["Context 1", "Context 2"],
  "reference": "Reference answer",
  "retrieved_contexts": ["Retrieved context 1", "Retrieved context 2"],
  "response": "Generated answer",
  "context_precision": 0.85,
  "context_recall": 0.90,
  ...
}
```

**Factual Correctness Detailed Output** (`factual_correctness_detailed.jsonl`):
```json
{
  "query_id": 0,
  "user_input": "Query text",
  "response": "Generated answer",
  "reference": "Reference answer",
  "factual_correctness": 0.85,
  "response_length": 1234,
  "reference_length": 987,
  "processing_time_ms": 5432,
  "num_attempts": 1,
  "status": "SUCCESS",
  "error": "",
  "evaluation_config": {
    "mode": "F1",
    "atomicity": "low",
    "coverage": "low"
  }
}
```

## Checkpoint/Resume System

All three main scripts support checkpoint/resume:

- **Checkpoint files**: Auto-saved at intervals (configurable)
- **Resume flag**: `--resume` to continue from last checkpoint
- **Automatic cleanup**: Checkpoints cleared on successful completion
- **RAG Executor**: Checkpoint and output files are timestamped together
  - On fresh run: Generates timestamp (e.g., `250122_143022`)
  - On resume: Finds latest `executor_checkpoint_*.json` and reuses its timestamp
  - Output file uses same timestamp: `rag_execution_250122_143022.jsonl`

**When interrupted**:
1. Process stopped mid-execution
2. Checkpoint file contains last processed index
3. Re-run with `--resume` flag
4. Skips already processed items
5. For RAG executor: Resumes to the same timestamped output file

## Logging

Three log files created in `logs/` directory:
- `main_YYYYMMDD_HHMMSS.log`: All operations (DEBUG level)
- `api_YYYYMMDD_HHMMSS.log`: API calls and responses
- `errors_YYYYMMDD_HHMMSS.log`: Errors only

Console output level: INFO (use `--verbose` for DEBUG)

## Environment Variables

Required in `.env` file:
```bash
OPENROUTER_API_KEY=your_api_key_here
```

## Common Issues

### OpenRouter API Compatibility
- OpenRouter does not support the `n` parameter for multiple completions
- Solution: Use `OpenRouterChatOpenAI` wrapper with `LangchainLLMWrapper(bypass_n=True)`
- This tells Ragas to batch prompts instead: `[prompt, prompt, prompt]`

### Large Document Memory Issues
- Problem: Loading 12MB documents causes memory errors
- Solution: Use two-phase workflow (extract metadata first)
- Run `extract_metadata.py` before `generate_multihop_testset.py`

### SmartKnowledge API Not Running
- Error: Health check fails in `run_rag_executor.py`
- Solution: Start SmartKnowledge at `http://localhost:3000`
- Verify with: `curl http://localhost:3000/`

### Failed Queries in Evaluation
- Default: Queries without contexts get zero scores (included in averages)
- Alternative: Use `--skip-failed` to exclude from evaluation
- Note: Affects average scores and sample count

### Missing Required --metric Argument
- Error: `the following arguments are required: --metric`
- Solution: Always specify a single metric using `--metric <metric_name>`
- Example: `uv run python run_ragas_evaluation.py --metric faithfulness`
- Valid metrics: `context_precision`, `context_recall`, `context_entity_recall`, `answer_relevancy`, `faithfulness`, `answer_correctness`, `answer_similarity`, `context_utilization`

### Timeout Errors During Evaluation
- **Symptom**: LLM-as-judge evaluations timing out frequently
- **Root cause**: Large responses/references generate too many claims for verification
- **Solution 1**: Use `evaluate_factual_correctness.py` with low atomicity/coverage
  ```bash
  uv run python evaluate_factual_correctness.py --input results/rag_execution.jsonl
  ```
- **Solution 2**: Use precision-only mode (50% faster)
  ```bash
  uv run python evaluate_factual_correctness.py --input results/rag_execution.jsonl --mode precision
  ```
- **Solution 3**: Increase timeout per sample
  ```bash
  uv run python evaluate_factual_correctness.py --input results/rag_execution.jsonl --timeout 900
  ```
- **Diagnosis**: Check the timeout correlation analysis in the report
  - Identifies length threshold where timeouts become common
  - Suggests max length limits for future evaluations
  - Shows average processing time by sample length
