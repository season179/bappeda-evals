#!/usr/bin/env python3
import pandas as pd
import json
from pathlib import Path
import sys

# Find latest rag_execution_*.jsonl file
results_dir = Path('results')
execution_files = sorted(results_dir.glob('rag_execution_*.jsonl'), reverse=True)

if not execution_files:
    print("Error: No rag_execution_*.jsonl files found in results/ directory")
    sys.exit(1)

latest_file = execution_files[0]
print(f"Analyzing file: {latest_file}")
print()

# Read JSONL file
with open(latest_file, 'r', encoding='utf-8') as f:
    results = [json.loads(line) for line in f]

df = pd.DataFrame(results)

print("=" * 80)
print("RAG EXECUTOR RESULTS ANALYSIS")
print("=" * 80)
print(f"\nTotal queries processed: {len(df)}")
print(f"Successful: {(df['status'] == 'SUCCESS').sum()}")
print(f"Failed: {(df['status'] == 'FAILED').sum()}")

print("\n" + "=" * 80)
print("PER-QUERY ANALYSIS")
print("=" * 80)

for idx, row in df.iterrows():
    # actual_contexts is already a list (not JSON string)
    actual_contexts = row['actual_contexts']
    print(f"\nQuery {row['query_id']}:")
    print(f"  Status: {row['status']}")
    print(f"  Retrieved contexts: {len(actual_contexts)}")
    print(f"  Answer length: {len(row['actual_answer'])} chars")
    print(f"  API latency: {row['api_latency_ms']}ms")
    if row['error']:
        print(f"  Error: {row['error']}")

    # Show first 100 chars of answer
    answer_preview = row['actual_answer'][:100] + "..." if len(row['actual_answer']) > 100 else row['actual_answer']
    print(f"  Answer preview: {answer_preview}")
