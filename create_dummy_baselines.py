"""Create baseline hypothesis files using dummy correction strategies.

Usage:
    python create_dummy_baselines.py <input_hypothesis.jsonl> <output_dir>

Creates one hypothesis file per strategy in the output directory.
"""

import sys
import json
from pathlib import Path


class DummyCorrectReproduce:
    """Identity baseline: copies the OCR hypothesis unchanged."""

    def __init__(self):
        self.name = "same"

    @staticmethod
    def correct_text(sentence, lang):
        return sentence


class DummyCorrectRandom:
    """Random baseline: shuffles words in the OCR hypothesis."""

    def __init__(self):
        from random import seed, shuffle
        self.shuffle = shuffle
        self.name = "random"
        seed(42)

    def correct_text(self, sentence, lang):
        words = sentence.split()
        self.shuffle(words)
        return " ".join(words)


STRATEGIES = [DummyCorrectReproduce, DummyCorrectRandom]


def create_baseline(input_path, output_path, strategy):
    """Apply a correction strategy to each record and write the result."""
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            lang = record["document_metadata"].get("language", "unknown")
            ocr_text = record["ocr_hypothesis"]["transcription_unit"]
            record["ocr_postcorrection_output"]["transcription_unit"] = \
                strategy.correct_text(ocr_text, lang)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_hypothesis.jsonl> <output_dir>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    for cls in STRATEGIES:
        strategy = cls()
        output_path = output_dir / f"{strategy.name}_{input_path.stem}_run1.jsonl"
        count = create_baseline(input_path, output_path, strategy)
        print(f"[{strategy.name}] Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()