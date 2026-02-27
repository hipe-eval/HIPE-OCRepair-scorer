"""CLI entry point for the HIPE-OCRepair scorer.

Two modes:
  1. Single file pair:
     hipe-ocrepair-scorer --reference ref.jsonl --hypothesis hyp.jsonl

  2. Folder pair:
     hipe-ocrepair-scorer --reference-dir data/reference/ --hypothesis-dir data/hypothesis/
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonschema import validate, ValidationError

from hipe_ocrepair_scorer.ocrepair_eval import Evaluation


# ---------------------------------------------------------------------------
# Schema loading and validation
# ---------------------------------------------------------------------------

_BUILTIN_SCHEMA_PATH = Path(__file__).parent.parent / "data" / "schema" / "hipe-ocrepair.schema.json"


def load_schema(schema_path: Path) -> dict:
    """Load a JSON Schema file."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_jsonl_record(record: dict, schema: dict, filepath: str, line_num: int) -> List[str]:
    """Validate a single JSONL record against the schema.

    Returns a list of error messages (empty if valid).
    """
    errors = []
    try:
        validate(instance=record, schema=schema)
    except ValidationError as e:
        errors.append(f"[SCHEMA] {filepath} (line {line_num}): {e.message}")
    return errors


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_jsonl(filepath: Path, schema: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts.

    If a schema is provided, each record is validated against it.
    Raises SystemExit on JSON parse errors or schema violations.
    """
    records = []
    errors = []

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"[JSON] {filepath} (line {i}): {e}")
                continue

            if schema is not None:
                errors.extend(validate_jsonl_record(record, schema, str(filepath), i))

            records.append(record)

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        print(f"[FAIL] {filepath}: {len(errors)} error(s).", file=sys.stderr)
        sys.exit(1)

    return records


# ---------------------------------------------------------------------------
# Merging reference and hypothesis
# ---------------------------------------------------------------------------

def merge_reference_hypothesis(
    ref_records: List[Dict[str, Any]],
    hyp_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge reference and hypothesis records by document_id.

    Takes ground_truth and ocr_hypothesis from the reference,
    and ocr_postcorrection_output from the hypothesis.

    Validates that:
    - All reference document_ids are present in the hypothesis.
    - The postcorrection output is not the placeholder "None".
    """
    hyp_by_id = {}
    for rec in hyp_records:
        doc_id = rec["document_metadata"]["document_id"]
        if doc_id in hyp_by_id:
            print(f"[ERROR] Duplicate document_id in hypothesis: {doc_id}", file=sys.stderr)
            sys.exit(1)
        hyp_by_id[doc_id] = rec

    merged = []
    missing = []
    placeholder = []

    for ref in ref_records:
        doc_id = ref["document_metadata"]["document_id"]
        if doc_id not in hyp_by_id:
            missing.append(doc_id)
            continue

        hyp = hyp_by_id[doc_id]
        output_text = hyp["ocr_postcorrection_output"]["transcription_unit"]
        if output_text == "None":
            placeholder.append(doc_id)
            continue

        merged.append({
            "document_metadata": ref["document_metadata"],
            "ground_truth": ref["ground_truth"],
            "ocr_hypothesis": ref["ocr_hypothesis"],
            "ocr_postcorrection_output": hyp["ocr_postcorrection_output"],
        })

    if missing:
        print(
            f"[ERROR] {len(missing)} reference document(s) not found in hypothesis: "
            f"{missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    if placeholder:
        print(
            f"[WARNING] {len(placeholder)} document(s) skipped — postcorrection "
            f"output is placeholder 'None': {placeholder}",
            file=sys.stderr,
        )

    if not merged:
        print("[ERROR] No valid documents to evaluate after merging.", file=sys.stderr)
        sys.exit(1)

    return merged


# ---------------------------------------------------------------------------
# Folder-mode helpers
# ---------------------------------------------------------------------------

def find_reference_files(ref_dir: Path) -> List[Path]:
    """Find all JSONL files in the reference directory."""
    files = sorted(ref_dir.glob("*.jsonl"))
    if not files:
        print(f"[ERROR] No .jsonl files found in {ref_dir}", file=sys.stderr)
        sys.exit(1)
    return files


def find_hypothesis_file(hyp_dir: Path, ref_path: Path) -> Optional[Path]:
    """Find the hypothesis file matching a reference file.

    Looks for any JSONL file in hyp_dir whose name contains the reference
    file stem. Falls back to substring matching.
    """
    # Pattern: teamname_<refname>_runX.jsonl
    candidates = list(hyp_dir.glob(f"*_{ref_path.stem}_*.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(
            f"[WARNING] Multiple hypothesis files match {ref_path.name}: "
            f"{[c.name for c in candidates]}. Using first.",
            file=sys.stderr,
        )
        return candidates[0]

    # Broader fallback: any file containing the ref stem
    candidates = [f for f in hyp_dir.glob("*.jsonl") if ref_path.stem in f.stem]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(
            f"[WARNING] Multiple hypothesis files match {ref_path.name}: "
            f"{[c.name for c in candidates]}. Using first.",
            file=sys.stderr,
        )
        return candidates[0]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _round_scores(results: dict, decimals: int = 4) -> dict:
    """Round all score tuples in the results dict."""
    for section in ("fold_scores", "averaged_scores"):
        if section == "fold_scores":
            for fold in results[section]:
                for metric in results[section][fold]:
                    vals = results[section][fold][metric]
                    results[section][fold][metric] = tuple(round(v, decimals) for v in vals)
        else:
            for metric in results[section]:
                vals = results[section][metric]
                results[section][metric] = tuple(round(v, decimals) for v in vals)
    return results


def run_evaluation(merged_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run the evaluation on merged records and return the score dict."""
    evaluation = Evaluation(merged_records)
    results = evaluation.score_over_datasets(normalize=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        prog="hipe-ocrepair-scorer",
        description="HIPE-OCRepair scorer: evaluate OCR post-correction outputs.",
    )

    # Mode 1: single file pair
    parser.add_argument(
        "--reference",
        help="Path to reference JSONL file.",
    )
    parser.add_argument(
        "--hypothesis",
        help="Path to hypothesis JSONL file.",
    )

    # Mode 2: folder pair
    parser.add_argument(
        "--reference-dir",
        help="Path to directory containing reference JSONL files.",
    )
    parser.add_argument(
        "--hypothesis-dir",
        help="Path to directory containing hypothesis JSONL files.",
    )

    # Optional
    parser.add_argument(
        "--schema",
        default=None,
        help="Path to JSON schema file for validation. "
             "Uses built-in schema if not specified.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        default=False,
        help="In folder mode, also compute aggregate scores across all files.",
    )

    args = parser.parse_args()

    # Determine mode
    file_mode = args.reference is not None or args.hypothesis is not None
    dir_mode = args.reference_dir is not None or args.hypothesis_dir is not None

    if file_mode and dir_mode:
        parser.error("Cannot mix --reference/--hypothesis with --reference-dir/--hypothesis-dir.")
    if not file_mode and not dir_mode:
        parser.error("Provide either --reference/--hypothesis or --reference-dir/--hypothesis-dir.")

    # Load schema
    schema_path = Path(args.schema) if args.schema else _BUILTIN_SCHEMA_PATH
    if schema_path.exists():
        schema = load_schema(schema_path)
    else:
        print(f"[WARNING] Schema file not found at {schema_path}, skipping validation.", file=sys.stderr)
        schema = None

    if file_mode:
        if not args.reference or not args.hypothesis:
            parser.error("Both --reference and --hypothesis are required in file mode.")

        ref_path = Path(args.reference)
        hyp_path = Path(args.hypothesis)
        if not ref_path.is_file():
            print(f"[ERROR] Reference file not found: {ref_path}", file=sys.stderr)
            sys.exit(1)
        if not hyp_path.is_file():
            print(f"[ERROR] Hypothesis file not found: {hyp_path}", file=sys.stderr)
            sys.exit(1)

        ref_records = load_jsonl(ref_path, schema)
        hyp_records = load_jsonl(hyp_path, schema)
        merged = merge_reference_hypothesis(ref_records, hyp_records)
        results = run_evaluation(merged)
        results = _round_scores(results)
        print(json.dumps(results, indent=2, ensure_ascii=False))

    else:  # dir_mode
        if not args.reference_dir or not args.hypothesis_dir:
            parser.error("Both --reference-dir and --hypothesis-dir are required in folder mode.")

        ref_dir = Path(args.reference_dir)
        hyp_dir = Path(args.hypothesis_dir)
        if not ref_dir.is_dir():
            print(f"[ERROR] Reference directory not found: {ref_dir}", file=sys.stderr)
            sys.exit(1)
        if not hyp_dir.is_dir():
            print(f"[ERROR] Hypothesis directory not found: {hyp_dir}", file=sys.stderr)
            sys.exit(1)

        ref_files = find_reference_files(ref_dir)
        all_results = {}
        all_merged = []

        for ref_path in ref_files:
            hyp_path = find_hypothesis_file(hyp_dir, ref_path)
            if hyp_path is None:
                print(
                    f"[WARNING] No hypothesis file found for {ref_path.name}, skipping.",
                    file=sys.stderr,
                )
                continue

            print(f"Evaluating: {ref_path.name} <-> {hyp_path.name}", file=sys.stderr)
            ref_records = load_jsonl(ref_path, schema)
            hyp_records = load_jsonl(hyp_path, schema)
            merged = merge_reference_hypothesis(ref_records, hyp_records)

            results = run_evaluation(merged)
            results = _round_scores(results)
            all_results[ref_path.stem] = results

            if args.aggregate:
                all_merged.extend(merged)

        if not all_results:
            print("[ERROR] No valid documents to evaluate across all files.", file=sys.stderr)
            sys.exit(1)

        output = {"per_file": all_results}

        if args.aggregate and all_merged:
            aggregate_results = run_evaluation(all_merged)
            aggregate_results = _round_scores(aggregate_results)
            output["aggregate"] = aggregate_results

        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()