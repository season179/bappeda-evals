#!/usr/bin/env python3
"""
Verify the generated testset.jsonl file for completeness and quality.
"""

import json
from pathlib import Path
from collections import defaultdict


def verify_testset(testset_file: str = "domain-approach/testset.jsonl"):
    """Verify the testset for completeness and quality."""

    print("="*70)
    print("TESTSET VERIFICATION REPORT")
    print("="*70)
    print()

    # Check if file exists
    if not Path(testset_file).exists():
        print(f"❌ ERROR: File {testset_file} not found!")
        return False

    test_cases = []
    user_inputs = []
    duplicate_questions = defaultdict(list)
    issues = []

    # Read all test cases
    with open(testset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            if line.strip():
                try:
                    test_case = json.loads(line)
                    test_cases.append(test_case)

                    # Check for required fields
                    if "user_input" not in test_case:
                        issues.append(f"Line {line_num}: Missing 'user_input' field")
                    if "reference" not in test_case:
                        issues.append(f"Line {line_num}: Missing 'reference' field")

                    # Track user inputs for duplicate detection
                    user_input = test_case.get("user_input", "")
                    if user_input:
                        user_inputs.append(user_input)
                        duplicate_questions[user_input].append(line_num)

                    # Check for empty fields
                    if not test_case.get("user_input", "").strip():
                        issues.append(f"Line {line_num}: Empty 'user_input' field")
                    if not test_case.get("reference", "").strip():
                        issues.append(f"Line {line_num}: Empty 'reference' field")

                except json.JSONDecodeError as e:
                    issues.append(f"Line {line_num}: Invalid JSON - {str(e)}")

    # Report total count
    print(f"📊 Total Test Cases: {len(test_cases)}")
    print(f"   Expected: 115")
    if len(test_cases) == 115:
        print("   ✅ Count is correct!")
    else:
        print(f"   ⚠️  Count mismatch! Missing {115 - len(test_cases)} test cases")
    print()

    # Check for duplicates
    duplicates = {q: lines for q, lines in duplicate_questions.items() if len(lines) > 1}
    if duplicates:
        print(f"⚠️  Found {len(duplicates)} duplicate questions:")
        for question, lines in list(duplicates.items())[:5]:  # Show first 5
            print(f"   - '{question[:60]}...' (lines: {lines})")
        if len(duplicates) > 5:
            print(f"   ... and {len(duplicates) - 5} more duplicates")
        print()
    else:
        print("✅ No duplicate questions found!")
        print()

    # Report issues
    if issues:
        print(f"⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more issues")
        print()
    else:
        print("✅ No structural issues found!")
        print()

    # Sample verification - show a few examples
    print("📝 Sample Test Cases:")
    print()
    for i in [0, 24, 49, 74, 96, 114]:  # Sample from each batch
        if i < len(test_cases):
            tc = test_cases[i]
            print(f"   Test Case {i+1}:")
            print(f"   Q: {tc.get('user_input', '')[:70]}...")
            print(f"   A: {tc.get('reference', '')[:70]}...")
            print()

    # Statistics
    print("📈 Statistics:")
    avg_question_len = sum(len(tc.get('user_input', '')) for tc in test_cases) / len(test_cases) if test_cases else 0
    avg_reference_len = sum(len(tc.get('reference', '')) for tc in test_cases) / len(test_cases) if test_cases else 0

    print(f"   Average question length: {avg_question_len:.0f} characters")
    print(f"   Average reference length: {avg_reference_len:.0f} characters")
    print()

    # Final verdict
    print("="*70)
    if len(test_cases) == 115 and not duplicates and not issues:
        print("✅ VERIFICATION PASSED - All 115 test cases are valid!")
    else:
        print("⚠️  VERIFICATION COMPLETED WITH WARNINGS - Please review the issues above")
    print("="*70)

    return True


def main():
    verify_testset()


if __name__ == "__main__":
    main()
