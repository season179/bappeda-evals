#!/usr/bin/env python3
"""
Verify query_id mapping between tuples.md and testset.jsonl
"""

import json


def verify_mapping():
    """Verify that query_id mapping is correct"""

    print("="*80)
    print("QUERY ID MAPPING VERIFICATION")
    print("="*80)
    print()

    # Read testset
    test_cases = []
    with open("domain-approach/testset.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line))

    # Read tuples
    tuples = []
    with open("domain-approach/tuples.md", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and line.startswith("Query"):
                tuples.append(line.strip())

    print(f"📊 Total test cases: {len(test_cases)}")
    print(f"📊 Total tuples: {len(tuples)}")
    print()

    # Check mapping
    all_correct = True
    for tc in test_cases:
        query_id = tc["query_id"]
        if query_id > len(tuples):
            print(f"❌ Query ID {query_id} out of range!")
            all_correct = False

    if all_correct:
        print("✅ All query IDs are within valid range!")
        print()

    # Show sample mappings
    print("📝 Sample Query ID Mappings:")
    print()

    samples = [1, 25, 50, 75, 100, 115]
    for query_id in samples:
        if query_id <= len(test_cases) and query_id <= len(tuples):
            tc = test_cases[query_id - 1]
            tuple_def = tuples[query_id - 1]

            print(f"Query {query_id}:")
            print(f"  Tuple: {tuple_def}")
            print(f"  Question: {tc['user_input'][:70]}...")
            print()

    print("="*80)
    print("✅ VERIFICATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    verify_mapping()
