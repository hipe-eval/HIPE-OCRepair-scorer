# HIPE-OCRepair-scorer

The HIPE-OCRepair-scorer is a **Python module for evaluating OCR post-correction**.

It is developed and used in the context of the **HIPE-OCRepair-2026** ICDAR
Competition on OCR post-correction for historical documents, which is part of the
broader [HIPEval](https://github.com/hipe-eval) (Historical Information Processing Evaluation) initiative, a series of
shared tasks on historical document processing.

#### Related repositories and websites:
- [HIPE-OCRepair](https://hipe-eval.github.io/HIPE-OCRepair-2026/): Website of the
  competition hosted at [ICDAR-2026](https://icdar2026.org/)
- [HIPE-OCRepair-2026-data](https://github.com/hipe-eval/HIPE-OCRepair-2026-data):
  public data releases (training, validation and test sets) for the HIPE-OCRepair-2026 shared task.
- [HIPE-OCRepair-2026-eval](https://github.com/hipe-eval/HIPE-OCRepair-2026-eval):
  for the Hugging Face leaderboard

#### Release history
- 2x Feb 2026: [v0.9](https://github.com/hipe-eval/HIPE-OCRepair-scorer/releases/tag/v0.9), initial release of the OCR post-correction scorer

[Main functionalities](#main-functionalities) | [Input format, scorer entry points, and naming conventions](#input-format-scorer-entry-points-and-naming-conventions) | [Installation and usage](#installation-and-usage) | [About](#about)

## Main functionalities 📊

The scorer evaluates OCR post-correction outputs against ground-truth
transcriptions. It computes match error rates at character and word level
(cMER/wMER) as well as preference metrics that compare the post-correction
output to the raw OCR hypothesis.

### Metrics

All metrics are based on **Match Error Rate (MER)**, computed as:

$$\text{MER} = \frac{S + D + I}{H + S + D + I}$$

where H = hits, S = substitutions, D = deletions, I = insertions. Unlike
standard CER/WER, MER is capped in [0, 1] because insertions are included in the
denominator. This reduces sensitivity to extreme hallucinations while remaining
easy to interpret. MER is equivalent to the **normalized CER** in the sense of
the OCR-D evaluation spec (see
https://ocr-d.de/en/spec/ocrd_eval.html#character-error-rate-cer).

**Primary metrics**

- **cMER (character-level MER, micro-averaged)**: corpus-level character match
  error rate, the main evaluation metric. Micro-averaged so longer documents
  contribute more than shorter ones.
- **Preference score (macro average)**: a simple sign-based metric computed per
  input document and then averaged unweighted across documents. For each item *i*:
  $s_i = \text{sign}(\text{cMER}_{\text{in},i} - \text{cMER}_{\text{out},i})$,
  yielding 1 (improved), 0 (tied), or -1 (worse). This captures how consistently
  a system improves over the input, while cMER captures the magnitude of improvement.

**Additional metrics**

- **wMER (word-level MER)**: reported for completeness, but cMER is preferred in
  historical OCR due to spelling variation and transcription conventions.
- **Confidence intervals**: computed for all measures to ensure statistical
  robustness.

### Normalization and stratification

Before scoring, text is normalized as follows:
- Case-folded to lowercase
- Unicode letters and digits are kept (including accented characters such as é, ç, ü)
- All other characters (punctuation, symbols) are replaced with space
- Whitespace is collapsed

This means evaluation is **case-insensitive** and **punctuation-insensitive**, but
**sensitive to accented characters** (é ≠ e).

Results can be stratified by dataset or any user-defined mapping.

## Input format, scorer entry points, and naming conventions

The scorer accepts two entry points (the same example structure is used in both):

1. **A pair of JSONL files**: one for reference, one for hypothesis.
2. **A pair of folders**: containing reference and hypothesis JSONL files respectively.

Each JSONL record should contain a dictionary with these fields:

- `document_metadata`: `{ "document_id": "...", "primary_dataset_name": "..." }`
- `ground_truth`: `{ "transcription_unit": "..." }`
- `ocr_hypothesis`: `{ "transcription_unit": "..." }`
- `ocr_postcorrection_output`: `{ "transcription_unit": "..." }`

All JSON documents conform to the HIPE-OCRepair JSON Schema (add link later).

Sample data for quick inspection is available under `data/sample`.

#### Reference JSONL files

Reference files follow the HIPE-OCRepair canonical naming convention:

```
<file_basename>_<version>_<dataset>_<primary_version>_<split>_<language>.jsonl
```

Where:

- `file_basename`: always `hipe-ocrepair-bench`
- `version`: benchmark version
- `dataset`: dataset name (e.g., `icdar2017`)
- `primary_version`: primary dataset version
- `split`: data split (e.g., `train`, `test`)
- `language`: dataset language (e.g., `en`, `fr`)

#### Submission (hypothesis) JSONL files

Submission files to be evaluated are named as:

```
teamname_<inputfile>_runX.jsonl
```

## Installation and usage 🔧

The scorer requires **Python 3.12** and can be installed as a pip package or used as an editable dependency:

```bash
pip install hipe-ocrepair-scorer
```

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Python usage

´´´python
# overwrite this with any prediction file
import json
from hipe_ocrepair_scorer import Evaluation, align_records

# Load your JSONL files
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

REF = load_jsonl("reference.jsonl")
PRED = load_jsonl("hypothesis.jsonl")

# 1. Align the files by document_id
merged_data = align_records(REF, PRED)

# 2. Run the evaluation
evaluator = Evaluation(merged_data)
results = evaluator.score_over_datasets(normalize=True)

# 3. Print results (e.g., Micro-averaged Character MER)
score, lo, hi = results["averaged_scores"]["cmer_micro"]
print(f"Character MER: {score:.4f} (95% CI: {lo:.4f} - {hi:.4f})")
```

### CLI usage

After installation, the `hipe-ocrepair-scorer` command is available.

#### Evaluate a single file pair

```bash
hipe-ocrepair-scorer \
  --reference hipe_ocrepair_scorer/data/sample/reference/hipe-ocrepair-bench_v0.9_icdar2017_v1.2_train_fr.sample.jsonl \
  --hypothesis hipe_ocrepair_scorer/data/sample/hypothesis/no_edits_baseline/no_edits_hipe-ocrepair-bench_v0.9_icdar2017_v1.2_train_fr.sample_run1.jsonl
```

#### Evaluate all files in a folder pair

```bash
hipe-ocrepair-scorer \
  --reference-dir hipe_ocrepair_scorer/data/sample/reference/ \
  --hypothesis-dir hipe_ocrepair_scorer/data/sample/hypothesis/no_edits_baseline/
```

In folder mode, the scorer matches each reference file to its corresponding
hypothesis file by filename. Hypothesis files are expected to contain the reference
filename stem (see naming conventions above).

#### Output format

Results are printed to stdout as JSON.

**File mode** returns scores for the single file pair:

```json
{
  "averaged_scores": {
    "metric_name": [score, lower_ci, upper_ci],
    ...
  },
  "fold_scores": {
    "dataset_name": {
      "metric_name": [score, lower_ci, upper_ci],
      ...
    }
  }
}
```

**Folder mode** returns per-file results for each reference/hypothesis pair:

```json
{
  "per_file": {
    "reference_filename_1": {
      "averaged_scores": { ... },
      "fold_scores": { ... }
    },
    "reference_filename_2": {
      "averaged_scores": { ... },
      "fold_scores": { ... }
    }
  }
}
```

Each metric is a tuple of `(score, lower_95%_CI, upper_95%_CI)`. Metrics include
`cmer_micro`, `wmer_micro`, `cmer_macro`, `wmer_macro`, `pref_score_cmer_macro`,
`pref_score_wmer_macro`, `pcis_cmer_macro`, and `pcis_wmer_macro`.

## About

### License

See the `LICENSE` file in the repository for details.

### Acknowledgments

The HIPE-2026 organising team expresses its sincere appreciation to the ICDAR 2026 
Conference and Competition Committee for hosting the task. HIPE-eval editions are 
organised within the framework of the [Impresso - Media Monitoring of the Past](https://impresso-project.ch) project, funded by the Swiss National Science Foundation under grant No. CRSII5_213585 and by the Luxembourg National Research Fund under grant No. 17498891.
