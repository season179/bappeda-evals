# Usage Guide - RAG Evaluation System

## Quick Start

### 1. Fix the 401 Error First

Before running the generation, you need to resolve the API authentication issue:

```bash
# Step 1: Verify your API key works
python generate_testset.py --validate-api
```

If you see **401 errors**, do the following:

#### Option A: Check OpenRouter Credits
1. Visit https://openrouter.ai/credits
2. Verify you have sufficient credits
3. Add credits if needed

#### Option B: Verify API Key
1. Visit https://openrouter.ai/settings/keys
2. Check if your API key is valid
3. Copy the key exactly (no extra spaces)
4. Update `.env` file:
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
   ```

#### Option C: Try Different Models
Edit `config.yaml` and use more accessible models:

```yaml
llm:
  model: "openai/gpt-4o-mini"  # Instead of x-ai/grok-code-fast-1

embeddings:
  model: "openai/text-embedding-3-small"  # Instead of qwen/qwen3-embedding-8b
```

Then test again:
```bash
python generate_testset.py --validate-api
```

### 2. Run Test Mode

Once validation passes, test with a small dataset:

```bash
python generate_testset.py --test
```

This will:
- Use only 1 document
- Generate 5 samples
- Complete in a few minutes
- Validate the entire pipeline works

### 3. Full Production Run

When test mode succeeds:

```bash
python generate_testset.py
```

This will:
- Process all 8 documents
- Generate 100 samples
- Take approximately 46 minutes (based on previous runs)
- Save checkpoints every 10 samples
- Save progress to `progress_summary.json`
- Save partial results to `testset_partial.csv`

## Monitoring Progress

### Real-Time Progress

While the script is running, open a new terminal and check progress:

```bash
# View progress summary
cat progress_summary.json

# View partial results (updates as samples are generated)
tail -f testset_partial.csv

# View main log
tail -f logs/main_*.log

# View API call log
tail -f logs/api_calls.log

# View errors only
tail -f logs/errors.log
```

### Progress Summary Example

```json
{
  "started_at": "2025-01-20T10:30:00",
  "last_updated": "2025-01-20T10:42:34",
  "status": "running",
  "current_phase": "HeadlinesExtractor",
  "current_document": "LKPJ_2024.md",
  "documents_processed": 6,
  "total_documents": 8,
  "samples_generated": 42,
  "target_samples": 100,
  "elapsed_seconds": 754,
  "estimated_remaining_seconds": 1125,
  "progress_percentage": 42.0
}
```

## Handling Interruptions

### If the Script Crashes or You Stop It

The system automatically saves checkpoints. To resume:

```bash
python generate_testset.py --resume
```

This will:
- Load the last checkpoint
- Show you what was completed
- Continue from where it left off
- Skip already-processed documents

### If You Want to Start Over

```bash
python generate_testset.py --reset
```

This will:
- Clear the checkpoint
- Delete progress file
- Start fresh from document 1

### If You Get Interrupted Mid-Run

Press `Ctrl+C` to stop gracefully. The script will:
- Save partial results to `testset_partial.csv`
- Save checkpoint with current progress
- Show you how many samples were completed
- Tell you to use `--resume` to continue

## Command Reference

### Basic Commands

```bash
# Normal run (with checkpointing)
python generate_testset.py

# Resume from checkpoint
python generate_testset.py --resume

# Start fresh (clear checkpoint)
python generate_testset.py --reset

# Test mode (1 doc, 5 samples)
python generate_testset.py --test

# Validate API only
python generate_testset.py --validate-api
```

### Advanced Commands

```bash
# Enable verbose debug logging
python generate_testset.py --verbose

# Custom configuration file
python generate_testset.py --config my_config.yaml

# Custom output file
python generate_testset.py --output my_results.csv

# Disable checkpointing (not recommended)
python generate_testset.py --no-checkpoint

# Combine flags
python generate_testset.py --test --verbose --reset
```

## Configuration

### Edit `config.yaml` to customize:

```yaml
# Change number of samples
generation:
  test_size: 100  # Change to 50, 200, etc.

# Change query distribution
generation:
  distribution:
    single_hop: 0.40
    multi_hop_abstract: 0.35
    multi_hop_specific: 0.25

# Change checkpoint frequency
checkpoint:
  save_every_n_samples: 10  # Save after every 10 samples
  save_every_document: true  # Save after each document

# Change progress update frequency
progress:
  update_interval: 30  # Update progress file every 30 seconds

# Change logging level
logging:
  console_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file_level: "DEBUG"

# Change error handling
error_handling:
  max_retries: 3
  initial_delay: 5.0
  backoff_factor: 2.0
```

## Output Files

### Generated Files

| File | Description | When Created |
|------|-------------|--------------|
| `testset_final.csv` | Final consolidated results | When generation completes successfully |
| `testset_partial.csv` | Incremental results during run | As samples are generated |
| `checkpoint.json` | Checkpoint state for resume | After each document/10 samples |
| `progress_summary.json` | Real-time progress stats | Updated every 30 seconds during run |
| `logs/main_*.log` | Complete execution log | Each run creates new timestamped log |
| `logs/api_calls.log` | API-specific debug log | Appended across runs |
| `logs/errors.log` | Errors only | Appended across runs |

### Backup Files

The system creates backups automatically:
- `checkpoint.json.backup` - Before clearing checkpoint
- `testset_partial.csv.backup` - Before overwriting results
- `*.corrupted` - If checkpoint file is corrupted

## Troubleshooting

### Problem: 401 "User not found" error

**Diagnosis:**
```bash
python generate_testset.py --validate-api
```

**Solutions:**
1. Check OpenRouter credits: https://openrouter.ai/credits
2. Verify API key: https://openrouter.ai/settings/keys
3. Try different models (see Option C above)

### Problem: Script seems stuck

**Check Progress:**
```bash
# View progress
cat progress_summary.json

# Check if it's actually running
tail -f logs/main_*.log
```

**Note:** HeadlinesExtractor phase can take 1-6 minutes per document. This is normal.

### Problem: Out of memory

**Solution:**
Reduce test size or use fewer documents:

```yaml
# In config.yaml
generation:
  test_size: 50  # Instead of 100
```

Or use test mode:
```bash
python generate_testset.py --test
```

### Problem: Want to see more detail

**Enable verbose logging:**
```bash
python generate_testset.py --verbose
```

This shows DEBUG-level logs in console.

### Problem: Lost partial results

**Check these files:**
- `testset_partial.csv` - Incremental results
- `testset_partial.csv.backup` - Backup from previous run
- `logs/main_*.log` - Full execution history

## Best Practices

### 1. Always Validate First
```bash
python generate_testset.py --validate-api
```

### 2. Test Before Full Run
```bash
python generate_testset.py --test
```

### 3. Monitor Long Runs
Open a second terminal to watch progress:
```bash
watch -n 5 cat progress_summary.json
```

### 4. Keep Logs
Don't delete the `logs/` directory - it's invaluable for debugging.

### 5. Backup Results
After successful completion:
```bash
cp testset_final.csv testset_final_$(date +%Y%m%d).csv
```

## Example Workflow

### Scenario 1: First Time Run

```bash
# 1. Validate API
python generate_testset.py --validate-api

# 2. Test with small dataset
python generate_testset.py --test

# 3. If test succeeds, run full generation
python generate_testset.py

# 4. Monitor in another terminal
watch -n 5 cat progress_summary.json
```

### Scenario 2: Resume After Interruption

```bash
# 1. Check what was completed
cat progress_summary.json

# 2. Resume from checkpoint
python generate_testset.py --resume

# 3. Monitor progress
tail -f logs/main_*.log
```

### Scenario 3: Retry After 401 Error

```bash
# 1. Fix API credentials in .env

# 2. Validate it works
python generate_testset.py --validate-api

# 3. Resume generation (don't lose progress!)
python generate_testset.py --resume
```

## Getting Help

### Check Logs First
```bash
# View recent errors
tail -100 logs/errors.log

# View full execution
cat logs/main_*.log

# Search for specific error
grep "401" logs/main_*.log
```

### Still Stuck?
1. Check `IMPLEMENTATION_SUMMARY.md` for detailed architecture
2. Review error messages - they include troubleshooting steps
3. Verify all dependencies are installed: `uv sync`

## Success Indicators

You'll know it's working when you see:

✅ API validation passes without errors
✅ Progress summary file updates every 30 seconds
✅ Partial results file grows as samples are generated
✅ Logs show "Document marked as processed"
✅ Checkpoint file updates after each document

## Next Steps

Once you successfully generate a testset:

1. **Review Results:**
   ```bash
   head testset_final.csv
   ```

2. **Backup Results:**
   ```bash
   cp testset_final.csv backups/testset_$(date +%Y%m%d_%H%M%S).csv
   ```

3. **Use for RAG Evaluation:**
   Load the CSV and use with your RAG evaluation metrics

4. **Generate More:**
   Modify `config.yaml` to try different distributions or sample sizes

Good luck with your RAG evaluation! 🚀
