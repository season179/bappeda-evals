#!/usr/bin/env python3
import pandas as pd
import json

df = pd.read_csv('results/eval_results.csv')

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
    actual_contexts = json.loads(row['actual_contexts'])
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
