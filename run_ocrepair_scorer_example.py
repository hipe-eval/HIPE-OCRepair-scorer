"""Example script showing how to use the HIPE-OCRepair scorer programmatically.

This script demonstrates both entry points:
1. Single file pair (reference + hypothesis)
2. Folder pair (reference directory + hypothesis directory)

Run from the repository root:
    python run_ocrepair_scorer_example.py
"""

import json
from pathlib import Path

from hipe_ocrepair_scorer.cli import (
    load_jsonl,
    load_schema,
    merge_reference_hypothesis,
    find_reference_files,
    find_hypothesis_file,
)
from hipe_ocrepair_scorer.ocrepair_eval import Evaluation


SAMPLE_DIR = Path("hipe_ocrepair_scorer/data/sample")
REF_DIR = SAMPLE_DIR / "reference"
HYP_DIR = SAMPLE_DIR / "hypothesis" / "no_edits_baseline"

# Pick the first reference file for the single-file example
REF_FILE = sorted(REF_DIR.glob("*.jsonl"))[0]
HYP_FILE = sorted(HYP_DIR.glob("*.jsonl"))[0]


def example_file_pair():
    """Evaluate a single reference/hypothesis file pair."""
    print("=" * 60)
    print("Example 1: Single file pair")
    print("=" * 60)
    print(f"  Reference:  {REF_FILE.name}")
    print(f"  Hypothesis: {HYP_FILE.name}")
    print()

    ref_records = load_jsonl(REF_FILE)
    hyp_records = load_jsonl(HYP_FILE)
    merged = merge_reference_hypothesis(ref_records, hyp_records)

    print(f"  Merged {len(merged)} documents")

    evaluation = Evaluation(merged)
    results = evaluation.score_over_datasets(normalize=True)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print()


def example_folder_pair():
    """Evaluate all matching files across reference and hypothesis directories."""
    print("=" * 60)
    print("Example 2: Folder pair")
    print("=" * 60)
    print(f"  Reference dir:  {REF_DIR}")
    print(f"  Hypothesis dir: {HYP_DIR}")
    print()

    ref_files = find_reference_files(REF_DIR)
    all_merged = []

    for ref_path in ref_files:
        hyp_path = find_hypothesis_file(HYP_DIR, ref_path)
        if hyp_path is None:
            print(f"  [SKIP] No hypothesis file for {ref_path.name}")
            continue

        print(f"  {ref_path.name} <-> {hyp_path.name}")
        ref_records = load_jsonl(ref_path)
        hyp_records = load_jsonl(hyp_path)
        merged = merge_reference_hypothesis(ref_records, hyp_records)
        all_merged.extend(merged)

    print(f"\n  Total: {len(all_merged)} documents")

    evaluation = Evaluation(all_merged)
    results = evaluation.score_over_datasets(normalize=True)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print()


if __name__ == "__main__":
    example_file_pair()
    example_folder_pair()
