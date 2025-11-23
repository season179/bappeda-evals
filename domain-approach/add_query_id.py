#!/usr/bin/env python3
"""
Add query_id to testset.jsonl
"""

import json
from pathlib import Path


def add_query_id_to_testset(input_file: str = "domain-approach/testset.jsonl",
                             output_file: str = "domain-approach/testset.jsonl"):
    """Add query_id to each test case in testset.jsonl"""

    test_cases = []

    # Read all test cases
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            if line.strip():
                test_case = json.loads(line)
                # Add query_id
                test_case_with_id = {
                    "query_id": line_num,
                    "user_input": test_case["user_input"],
                    "reference": test_case["reference"]
                }
                test_cases.append(test_case_with_id)

    # Write back with query_id
    with open(output_file, 'w', encoding='utf-8') as f:
        for test_case in test_cases:
            f.write(json.dumps(test_case, ensure_ascii=False) + '\n')

    print(f"✅ Added query_id to {len(test_cases)} test cases in {output_file}")


def add_query_id_to_tuples(input_file: str = "domain-approach/tuples.md",
                           output_file: str = "domain-approach/tuples.md"):
    """Add query_id to each tuple in tuples.md"""

    lines = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            if line.strip() and line_num <= 115:
                # Add query_id at the beginning
                new_line = f"Query {line_num}: {line.strip()}\n"
                lines.append(new_line)
            else:
                lines.append(line)

    # Write back with query_id
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"✅ Added query_id to tuples in {output_file}")


def main():
    print("Adding query_id to testset and tuples...\n")

    # Add query_id to testset.jsonl
    add_query_id_to_testset()

    # Add query_id to tuples.md
    add_query_id_to_tuples()

    print("\n✅ All files updated successfully!")


if __name__ == "__main__":
    main()
