# HIPE-OCRepair-scorer — Summary

## Repository Overview

This is the **official evaluation scorer** for the HIPE-OCRepair-2026 ICDAR shared task on OCR post-correction for historical documents. Given a system's corrected text alongside the raw OCR input and a ground-truth transcription, it measures how much (and how consistently) the system improves quality.

---

## Unit of Evaluation: the `transcription_unit`

Each JSONL record corresponds to one **document**, identified by `document_id`. The text being compared is the `transcription_unit` string found inside three fields:

| Field                                          | Role                                   |
| ---------------------------------------------- | -------------------------------------- |
| `ground_truth.transcription_unit`              | The reference (correct) transcription  |
| `ocr_hypothesis.transcription_unit`            | The raw OCR output (before correction) |
| `ocr_postcorrection_output.transcription_unit` | The system's post-corrected text       |

The **scope** of a transcription unit varies by dataset (semantic chunk, article, page, or paragraph — see data section below).

Documents are grouped by `primary_dataset_name` (e.g. `"icdar2017"`, `"dta19-l1"`) for aggregation. The `primary_dataset_name` value is the dataset identifier only — **language** and split are separate metadata fields.

---

## Data Release (v0.9)

The v0.9 release contains the following JSONL files, organised by dataset and language. Each filename follows the convention `hipe-ocrepair-bench_<version>_<dataset>_<primary_version>_<split>_<language>.jsonl`.

| Dataset              | Lang       | Splits available                           | Unit scope          | Notes                    |
| -------------------- | ---------- | ------------------------------------------ | ------------------- | ------------------------ |
| `icdar2017`          | en         | train, dev                                 | semantic chunk      |                          |
| `icdar2017`          | fr         | train                                      | semantic chunk      | no dev yet               |
| `dta19-l0`           | de         | train, train-unmatched, dev, dev-unmatched | book page           | noise level 0 (lightest) |
| `dta19-l1`           | de         | train, train-unmatched, dev, dev-unmatched | book page           | noise level 1            |
| `dta19-l2`           | de         | train, train-unmatched, dev, dev-unmatched | book page           | noise level 2 (heaviest) |
| `impresso-nzz`       | de         | train                                      | newspaper page      | no test set              |
| `impresso-snippets`  | de, en, fr | train, dev                                 | newspaper paragraph |                          |
| `overproof-combined` | en         | train, dev                                 | newspaper article   | no test set              |

**Notes on `dta19`:** The guidelines mention two noise levels, but the v0.9 release contains three (`l0`, `l1`, `l2`). Each noise level is a separate fold in the scorer (distinct `primary_dataset_name` values). The `-unmatched` splits contain documents that could not be matched across noise levels and are kept for single-level training.

**Notes on `overproof`:** The sample data used `overproof-ca` as `primary_dataset_name`; the full release uses `overproof-combined` (Chronicling America + Trove merged). Verify the actual `primary_dataset_name` field value in the release files before running cross-file evaluations.

**Note on `impresso-nzz` and `overproof-combined`:** These datasets have **no competition test set**. They are available for training and exploratory development only and do not contribute to the official rankings.

---

## Aggregation Levels

The term "corpus-level" is too vague for this benchmark given its multilingual, multi-dataset structure. There are four well-defined aggregation levels, from most to least specific:

| Level                       | Definition                                                                                   | Scorer support                                                                          |
| --------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **dataset × language**      | All documents of one `primary_dataset_name` from one language file — e.g. `icdar2017` / `fr` | ✓ natural when running scorer on a single JSONL file (`fold_scores`)                    |
| **dataset (all languages)** | All documents of one dataset across languages — e.g. all `icdar2017` regardless of language  | not directly; requires merging files before running scorer                              |
| **language (all datasets)** | All documents of one language across datasets — e.g. all French documents                    | not directly; requires merging files before running scorer                              |
| **overall**                 | All documents across all datasets and languages                                              | not recommended; mixes document types, OCR conditions, and languages in a single number |

The **dataset × language** level is the primary reported unit. Because each release file is language-specific, this is what the scorer naturally produces when run on a single file. The scorer stratifies internally by `primary_dataset_name`, which with a single-language file is equivalent to dataset × language.

Aggregation of scores across multiple dataset × language cells is done by averaging `fold_scores` entries — see [`fold_scores` vs `averaged_scores`](#fold_scores-vs-averaged_scores) below.

### Micro-averaging within a dataset × language cell

The per-document alignment counts (H, S, D, I) for all documents in the cell are **summed first**, then MER is computed once:

$$\text{cMER}_\text{micro} = \frac{\sum S_i + \sum D_i + \sum I_i}{\sum H_i + \sum S_i + \sum D_i + \sum I_i}$$

Effect: **longer documents weigh more**. This is the **primary metric**: it captures the aggregate error rate as if all text in the cell were concatenated. Differences in typical document length between datasets (page vs. chunk) are a known confound when comparing micro-cMER across dataset × language cells.

Bootstrap CIs resample documents (with replacement), sum their counts per resample, and compute MER.

### Macro-averaging within a dataset × language cell

MER is computed **per document first**, then averaged **unweighted** across documents:

$$\text{cMER}_\text{macro} = \frac{1}{N}\sum_{i=1}^N \text{cMER}(d_i)$$

Effect: **each document contributes equally** regardless of length.

The **preference score** (±1/0 per document, then averaged) and **PCIS** are inherently macro — computed per document, then averaged unweighted. These are the metrics to use when asking how _consistently_ a system improves over the raw OCR, as opposed to the _magnitude_ of improvement captured by micro-cMER.

Bootstrap CIs resample the vector of per-document scores and take the mean per resample.

---

## `fold_scores` vs `averaged_scores`

- **`fold_scores`**: micro/macro metrics computed independently per **dataset × language** cell (one entry per `primary_dataset_name` value in the input). When the input is a single language-specific file there is one fold. ⚠️ If a merged multi-language file is fed to the scorer, documents from different languages of the same dataset share one fold entry, losing language separation.
- **`averaged_scores`**: the **unweighted mean of fold-level point estimates**, with CIs computed by averaging the per-fold bootstrap sample arrays element-wise and taking the 2.5/97.5 percentiles. Each dataset × language cell contributes equally regardless of document count or text volume.

  **Important**: for micro metrics (`cmer_micro`, `wmer_micro`), `averaged_scores` is a **macro-of-micros** — it is the mean of per-fold micro-cMER values, _not_ a global micro-average over all documents. Global micro would give more weight to larger folds; this formulation does not. For single-file runs (one fold) the two are identical.

---

## Shared Task Result Compilation Strategy

### Evaluation cells

The atomic unit for reporting is a **dataset × language × split** cell, corresponding to one JSONL file. For the official competition (test split), the cells are:

| Cell | Dataset             | Lang | Noise        | Type                | Status         |
| ---- | ------------------- | ---- | ------------ | ------------------- | -------------- |
| 1    | `icdar2017`         | en   | real         | newspaper chunk     | test available |
| 2    | `icdar2017`         | fr   | real         | newspaper chunk     | test available |
| 3    | `dta19-l0`          | de   | synthetic L0 | book page           | test available |
| 4    | `dta19-l1`          | de   | synthetic L1 | book page           | test available |
| 5    | `dta19-l2`          | de   | synthetic L2 | book page           | test available |
| 6    | `impresso-snippets` | de   | real         | newspaper paragraph | test available |
| 7    | `impresso-snippets` | en   | real         | newspaper paragraph | test available |
| 8    | `impresso-snippets` | fr   | real         | newspaper paragraph | test available |

`overproof-combined` and `impresso-nzz` have no competition test sets and are not evaluation cells. They are available for training and exploratory development only.

### Per-cell reporting (fold_scores)

For each cell, report from `fold_scores`:

| Metric                  | Rationale                                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `cmer_micro`            | Primary: overall character error rate, length-weighted within cell                                                            |
| `cmer_hyp` (raw OCR)    | **Required context**: raw OCR baseline cMER; enables relative improvement computation and cross-cell comparability            |
| `Δcmer_rel`             | Derived: `(cmer_hyp − cmer_sys) / cmer_hyp`; normalises out cross-cell differences in baseline OCR quality and document scope |
| `pref_score_cmer_macro` | Secondary: fraction of documents improved, unaffected by document length                                                      |

Every leaderboard row **must** show `cmer_micro`, `cmer_hyp`, and `Δcmer_rel`. Absolute cMER values are not directly comparable across cells (pages vs. chunks vs. paragraphs have different baseline distributions); `Δcmer_rel` is the comparable quantity across cells.

⚠️ The current scorer output does not surface `cmer_hyp` directly. It must be extracted from the scorer internals or obtained by running the scorer on an identity submission (hypothesis = OCR input).

### Cross-cell aggregation (averaged_scores)

`averaged_scores` gives equal weight to each cell. Three aggregation views are useful:

The scorer's `averaged_scores` is an unweighted mean. The official ranking uses an **explicit weighting scheme** applied externally to per-cell `fold_scores` outputs (see Ranking protocol below). Three diagnostic aggregation views are also useful:

| View                     | Cells included          | How to obtain                                                              | What it captures                                 |
| ------------------------ | ----------------------- | -------------------------------------------------------------------------- | ------------------------------------------------ |
| **Official competition** | cells 1–8               | Weighted mean of per-cell `fold_scores` (see weights table); external step | Single ranking number; design-weight balanced    |
| **By language**          | e.g. cells 2+8 for `fr` | Average relevant `fold_scores` entries externally                          | Language-dimension performance                   |
| **dta19 noise gradient** | cells 3+4+5             | Unweighted average of the three `fold_scores.cmer_micro` values            | Noise-level effect within dta19; diagnostic only |

⚠️ **Do not merge files before running the scorer.** Feeding a merged multi-language or multi-dataset file collapses documents from different languages of the same dataset into one fold entry, destroying dataset × language separation. All cross-cell aggregates must be computed externally from per-file scorer outputs.

### Design weights for the overall score

The overall ranking score is a **weighted mean** over official test cells (cells 1–8). These are **design weights** — they ensure equal influence per conceptual dataset unit, not per document or per byte:

| Cell | Test set                 | Weight |
| ---- | ------------------------ | ------ |
| 1    | `icdar2017` / en         | 1      |
| 2    | `icdar2017` / fr         | 1      |
| 3    | `dta19-l0` / de          | 1/3    |
| 4    | `dta19-l1` / de          | 1/3    |
| 5    | `dta19-l2` / de          | 1/3    |
| 6    | `impresso-snippets` / de | 1      |
| 7    | `impresso-snippets` / en | 1      |
| 8    | `impresso-snippets` / fr | 1      |

The three DTA cells together count as one unit, giving `icdar2017`, `dta19` (as a whole), and each `impresso-snippets` language equal influence. The weighted mean formula is:

$$\text{score}_\text{overall} = \frac{s_1 + s_2 + \tfrac{1}{3}(s_3 + s_4 + s_5) + s_6 + s_7 + s_8}{6}$$

where $s_i$ is the per-cell metric value (`cmer_micro` or `pref_score_cmer_macro`). The denominator is 6 because the sum of weights is $1+1+\tfrac{1}{3}+\tfrac{1}{3}+\tfrac{1}{3}+1+1+1 = 6$. This formula applies identically to the primary and secondary rankings.

The scorer's `averaged_scores` output does **not** apply these weights. Official ranking values must be computed externally using the formula above.

### Ranking protocol

**Primary ranking**: weighted mean of per-cell `cmer_micro` across cells 1–8, **lower is better**, using the design weights defined above. This is the official leaderboard criterion.

**Secondary ranking**: weighted mean of per-cell `pref_score_cmer_macro` across cells 1–8, **higher is better**, using the same weights. Captures whether a system consistently helps rather than occasionally hurts, unaffected by document length.

**Per-cell reporting**: always include `cmer_micro`, `cmer_hyp` (raw OCR baseline), and `Δcmer_rel = (cmer_hyp − cmer_sys) / cmer_hyp` for each cell. Absolute cMER is not directly comparable across cells (pages vs. chunks vs. paragraphs differ in baseline error rate); `Δcmer_rel` provides the cross-cell-comparable view.

**Per-run**: each team submits up to 3 runs per cell. Report all runs; rank by best run.

**System comparison for the final paper**: per-system bootstrap CIs test whether a single system's score is stable, but do not test system differences. For pairwise significance claims in the competition paper, use **paired bootstrap resampling** — resample the same set of documents for both systems on each replicate, then compute the per-replicate difference.

### Communication guidelines

- Always state the aggregation level explicitly: "micro-cMER on `icdar2017`/fr" rather than "corpus cMER".
- When reporting `averaged_scores`, note it is a macro-of-micros (not a global micro) and list which cells are included.
- When comparing across dataset families (e.g. `icdar2017` vs. `impresso-snippets`), note that absolute micro-cMER values are not directly comparable due to different unit scopes and baseline OCR quality.
- For dta19, always specify the noise level or explicitly state "averaged across noise levels".

---

## All Metrics

| Metric                  | Level     | Aggregation                                     |
| ----------------------- | --------- | ----------------------------------------------- |
| `cmer_micro`            | character | micro (pool counts, then MER) — **primary**     |
| `wmer_micro`            | word      | micro                                           |
| `cmer_macro`            | character | macro (mean of per-doc MER)                     |
| `wmer_macro`            | word      | macro                                           |
| `pref_score_cmer_macro` | character | macro (mean of ±1/0 per doc)                    |
| `pref_score_wmer_macro` | word      | macro                                           |
| `pcis_cmer_macro`       | character | macro (mean of normalised relative improvement) |
| `pcis_wmer_macro`       | word      | macro                                           |

All metrics include 95% confidence intervals computed with 10 000 bootstrap resamples at the document level.

---

## MER Formula

All metrics are based on **Match Error Rate (MER)**:

$$\text{MER} = \frac{S + D + I}{H + S + D + I}$$

where H = hits, S = substitutions, D = deletions, I = insertions. Unlike standard CER/WER, MER is capped in [0, 1] because insertions appear in the denominator, reducing sensitivity to hallucinations.

---

## Text Normalisation

Before scoring, text is normalised as follows:

- Case-folded to lowercase
- Unicode-normalised so canonically equivalent forms compare the same (for example, decomposed umlaut sequences are folded to their composed equivalents)
- Explicit ligature / historical-character mappings applied where implemented by the scorer (e.g. `ß → ss`, `œ → oe`, `æ → ae`)
- Unicode letters and digits kept (including accented characters such as é, ç, ü)
- Underscores replaced with space (underscore is `\w` but is explicitly remapped in a separate step)
- All other non-`\w` characters (punctuation, symbols) replaced with space
- Whitespace collapsed

Evaluation is therefore **case-insensitive** and **punctuation-insensitive**. It is also insensitive to the scorer's documented canonical-equivalence and ligature/historical-character folds (for example `ß/ss`, `œ/oe`, `æ/ae`, and decomposed vs. composed umlaut forms), while remaining **sensitive to ordinary accented-vs.-unaccented letter differences** where no explicit mapping is applied (e.g. `é ≠ e`). This normalisation applies to all primary metrics. The participation guidelines have been updated to reflect this as the confirmed policy (corrigendum 20.03.2026).

> **⚠️ Release blocker — layout normalisation:** The guidelines specify a prior layout-normalisation step (soft-hyphen removal, line-break conversion), referenced as `normalise_layout()` in `hipe_ocrepair_scorer/utils/normalisation.py`. This function **does not yet exist** in the v0.9 scorer. Without it, line-break tokens and soft hyphens are treated as regular characters rather than being collapsed. This affects all official competition test cells that carry line-break encoding: `dta19` (all noise levels) and `impresso-snippets`. The `icdar2017` dataset has no line-break encoding and is unaffected. (`overproof-combined` and `impresso-nzz` are also affected but have no competition test sets.) **Layout normalisation must be implemented and validated before the test-phase scorer release.**

---

## Open Issues and Pre-evaluation Checklist

The following must be resolved before the official test-phase evaluation is run.

### Scorer / documentation mismatches

| Issue                                                                                         | Location                                                | Impact                                                       |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| Public task description uses "CER/WER" but scorer computes MER                                | Task website                                            | Misleads participants about the metric formula               |
| Section 5.4 of the participation guidelines states "case-sensitive and punctuation-sensitive" | Participation guidelines v1                             | Contradicts scorer implementation and corrigendum 20.03.2026 |
| `normalise_layout()` is referenced but does not exist                                         | `hipe_ocrepair_scorer/utils/normalisation.py` (missing) | Line-break and soft-hyphen handling absent for 4 datasets    |

### Required scorer changes

1. **Implement `normalise_layout()`** — soft-hyphen removal, line-break-to-space conversion; validate against `dta19` and `impresso-snippets` (the competition cells affected).
2. **Surface `cmer_hyp` in scorer output** — the raw OCR baseline cMER should be a first-class output field, not reconstructed externally.
3. **Implement the design-weighted mean** — the scorer's `averaged_scores` is unweighted; add an explicit weighted aggregation step or provide tooling so organisers can apply the 1/3 DTA weights externally.

### Required documentation updates

1. Align all public-facing text with the confirmed evaluation protocol: MER (not CER/WER), case-insensitive, punctuation-insensitive.
2. Update participation guidelines Section 5.4 to match the corrigendum 20.03.2026 and scorer behaviour.
3. Publish the confirmed weighting scheme (weight 1 per non-DTA test cell, weight 1/3 per DTA test cell) with the explicit formula so participants can reproduce the official ranking.
