#!/usr/bin/env python3
"""
Script to calculate factual correctness statistics from evaluation results.

Reads factual_correctness_detailed.jsonl and calculates:
- Total sum of factual_correctness scores
- Count of records
- Average score
- Final Score (sum / max_possible_score)
- Min and max scores
"""

import json
from pathlib import Path


def calculate_factual_correctness_stats(jsonl_path: str = "factual_correctness_detailed.jsonl"):
    """
    Calculate statistics from factual correctness evaluation results.

    Args:
        jsonl_path: Path to the JSONL file (relative to script directory)

    Returns:
        Dictionary containing calculated statistics
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    file_path = script_dir / jsonl_path

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    scores = []
    error_count = 0

    # Read and parse JSONL file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if 'factual_correctness' in data:
                    score = data['factual_correctness']
                    if isinstance(score, (int, float)):
                        scores.append(float(score))
                    else:
                        print(f"Warning: Line {line_num} has non-numeric factual_correctness: {score}")
                        error_count += 1
                else:
                    print(f"Warning: Line {line_num} missing 'factual_correctness' field")
                    error_count += 1
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}")
                error_count += 1

    # Calculate statistics
    if not scores:
        raise ValueError("No valid factual_correctness scores found in the file")

    total_sum = sum(scores)
    count = len(scores)
    average = total_sum / count
    max_possible_score = 1.0 * count  # Each record can have max score of 1.0
    final_score = total_sum / max_possible_score
    min_score = min(scores)
    max_score = max(scores)

    return {
        'total_sum': total_sum,
        'count': count,
        'average': average,
        'final_score': final_score,
        'min_score': min_score,
        'max_score': max_score,
        'error_count': error_count
    }


def main():
    """Main function to run the calculation and display results."""
    try:
        stats = calculate_factual_correctness_stats()

        print("=" * 60)
        print("FACTUAL CORRECTNESS STATISTICS")
        print("=" * 60)
        print(f"Total Sum:           {stats['total_sum']:.4f}")
        print(f"Count:               {stats['count']}")
        print(f"Average:             {stats['average']:.4f}")
        print(f"Final Score:         {stats['final_score']:.4f} ({stats['final_score']*100:.2f}%)")
        print(f"Min Score:           {stats['min_score']:.4f}")
        print(f"Max Score:           {stats['max_score']:.4f}")
        print("=" * 60)

        if stats['error_count'] > 0:
            print(f"\nWarning: {stats['error_count']} errors encountered while parsing")

        print(f"\nNote: Final Score = Total Sum / Max Possible Score")
        print(f"      where Max Possible Score = 1.0 × Count = {stats['count']}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
