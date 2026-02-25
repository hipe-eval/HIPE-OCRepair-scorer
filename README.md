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
transcriptions. It computes character and word error rates (CER/WER) as well as
preference metrics that compare the post-correction output to the raw OCR
hypothesis.

### Metrics

We report MER (match error rate), which we treat as a **normalized CER** in the
sense of the OCR-D evaluation spec (see
https://ocr-d.de/en/spec/ocrd_eval.html#character-error-rate-cer). MER is
micro-averaged across the corpus so longer documents contribute more than shorter
ones, and it is capped in [0, 1], which reduces sensitivity to extreme
hallucinations while remaining easy to interpret.

**Primary metrics**

- **MER (micro-averaged CER)**: corpus-level character error rate, the main
  evaluation metric.
- **Preference score (macro average)**: a simple sign-based metric computed per
  input document and then averaged unweighted across documents. For each item *i*:
  $s_i = \text{sign}(\text{CER}_{\text{in},i} - \text{CER}_{\text{out},i})$,
  yielding 1 (improved), 0 (tied), or -1 (worse). This captures how consistently
  a system improves over the input, while MER captures the magnitude of improvement.

**Additional metrics**

- **WER-based metrics**: reported for completeness, but CER/MER is preferred in
  historical OCR due to spelling variation and transcription conventions.
- **Confidence intervals**: computed for all measures to ensure statistical
  robustness.

### Normalization and stratification

The evaluation can normalize text to lowercase alphanumeric tokens and whitespace
before scoring. It can also stratify results by dataset or any user-defined
mapping.

## Input format, scorer entry points, and naming conventions

The scorer accepts two entry points (the same example structure is used in both):

1. **A pair of JSON documents**: one for reference, one for hypothesis.
2. **Pointers to two folders**:
   - `reference/`: JSONL files with reference data.
   - `hypothesis/`: JSONL files with hypothesis data.

Each JSON or JSONL record should contain a dictionary with these fields:

- `ground_truth`: `{ "transcription_unit": "..." }`
- `ocr_postcorrection_output`: `{ "transcription_unit": "..." }`
- `ocr_hypothesis`: `{ "transcription_unit": "..." }`
- `document_metadata`: `{ "primary_dataset_name": "..." }`

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

#### Python API usage

```python
from hipe_ocrepair_scorer.ocrepair_eval import Evaluation

# TO BE UPDATED
```

#### Output

The `score` and `score_over_datasets` methods return a dict with:

- `fold_scores`: per-stratum metrics as `(mean, low_ci, high_ci)`
- `averaged_scores`: mean across folds with pooled confidence intervals

## About

### License

See the `LICENSE` file in the repository for details.

### Acknowledgments

The HIPE-2026 organising team expresses its sincere appreciation to the ICDAR 2026 
Conference and Competition Committee for hosting the task. HIPE-eval editions are 
organised within the framework of the [Impresso - Media Monitoring of the Past](https://impresso-project.ch) project, funded by the Swiss National Science Foundation under grant No. CRSII5_213585 and by the Luxembourg National Research Fund under grant No. 17498891.
