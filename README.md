# Testset Generation - Quick Start

## Prerequisites

1. Set up environment variables in `.env`:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

2. Install dependencies:
```bash
uv sync
```

## Basic Usage

```bash
uv run python generate_testset_improved.py knowledge-files/ --size 50
```

## Common Examples

### Balanced distribution (default)
```bash
uv run python generate_testset_improved.py knowledge-files/ --size 50
```
Result: 50% single-hop, 25% multi-hop specific, 25% multi-hop abstract

### Easier questions (more single-hop)
```bash
uv run python generate_testset_improved.py knowledge-files/ \
    --size 50 \
    --single-hop 0.7 \
    --multi-hop-specific 0.2 \
    --multi-hop-abstract 0.1
```

### Harder questions (more multi-hop)
```bash
uv run python generate_testset_improved.py knowledge-files/ \
    --size 50 \
    --single-hop 0.3 \
    --multi-hop-specific 0.4 \
    --multi-hop-abstract 0.3
```

### Verbose output
```bash
uv run python generate_testset_improved.py knowledge-files/ --size 50 --verbose
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--size` | 50 | Number of questions to generate |
| `--single-hop` | 0.5 | Ratio of single-hop questions (0-1) |
| `--multi-hop-specific` | 0.25 | Ratio of multi-hop specific (0-1) |
| `--multi-hop-abstract` | 0.25 | Ratio of multi-hop abstract (0-1) |
| `--chunk-min` | 300 | Minimum tokens per chunk |
| `--chunk-max` | 1000 | Maximum tokens per chunk |
| `--model` | openai/gpt-5.1-codex-mini | LLM model via OpenRouter |
| `--embedding-model` | qwen/qwen3-embedding-8b | Embedding model via OpenRouter |
| `--output` | test_questions.jsonl | Output file path |

**Note:** Question ratios must sum to 1.0

## Output

Generates a JSONL file with test questions:
```jsonl
{"query_id": 0, "user_input": "...", "reference": "...", "reference_contexts": [...]}
{"query_id": 1, "user_input": "...", "reference": "...", "reference_contexts": [...]}
```

## Next Steps

1. Validate the output:
```bash
uv run python validate_data.py test_questions.jsonl --skip-response-check
```

2. Execute through your RAG system:
```bash
uv run python execute_rag.py test_questions.jsonl
```

3. Evaluate results:
```bash
uv run python evaluate_factual_correctness_standalone.py \
    --input results/rag_execution_*.jsonl \
    --output-dir results
```
