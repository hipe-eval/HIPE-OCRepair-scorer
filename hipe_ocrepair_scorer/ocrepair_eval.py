from collections import defaultdict
import numpy as np
from jiwer import process_characters, process_words
import re
import sys

from copy import deepcopy
from typing import Any, Callable, Dict, List, Sequence, Tuple


def norm(string: str) -> str:
    """Normalize a string for evaluation.

    Normalization policy:
    - Case-fold to lowercase.
    - Keep Unicode letters and digits (including accented characters).
    - Replace all other characters (punctuation, symbols) with space.
    - Collapse whitespace.
    - explicate some ligatures.

    This means evaluation is case-insensitive and punctuation-insensitive,
    but sensitive to accented characters (é ≠ e).
    """
    # lowercase
    string = string.lower()

    # ligature and historical character replacements
    string = string.replace("ß", "ss")
    string = string.replace("ꝛ", "r")
    string = string.replace("œ", "oe")
    string = string.replace("æ", "ae")
    string = string.replace("aͤ", "ä")
    string = string.replace("oͤ", "ö")
    string = string.replace("uͤ", "ü")
    
    
    # other normalizations
    string = string.replace("¬\n", "")
    string = re.sub(r"[^\w]", " ", string, flags=re.UNICODE)
    string = re.sub(r"_", " ", string)
    string = re.sub(r"\s+", " ", string)
    string = string.strip()
    return string


def align_records(ref_records: List[Dict], hyp_records: List[Dict]) -> List[Dict]:
    """Aligns reference and hypothesis records by document_id for Evaluation.

    Missing or placeholder ('None') hypothesis records are kept with an empty
    postcorrection output, which results in a maximum error score for that document.
    Warnings are printed to stderr for both cases.

    Records with structural problems (missing keys) are reported with the
    affected document_id and treated as empty output.
    """
    # Build hypothesis lookup, skipping malformed records.
    hyp_by_id = {}
    malformed_hyp = []
    for i, rec in enumerate(hyp_records):
        try:
            doc_id = rec["document_metadata"]["document_id"]
        except (KeyError, TypeError):
            malformed_hyp.append(
                f"  hypothesis record {i}: missing 'document_metadata.document_id'"
            )
            continue
        hyp_by_id[doc_id] = rec

    if malformed_hyp:
        print(
            f"[WARNING] {len(malformed_hyp)} hypothesis record(s) skipped "
            f"(cannot extract document_id):\n" + "\n".join(malformed_hyp),
            file=sys.stderr,
        )

    merged = []
    missing = []
    placeholder = []
    bad_structure = []

    for ref in ref_records:
        doc_id = ref["document_metadata"]["document_id"]

        if doc_id not in hyp_by_id:
            missing.append(doc_id)
            output = {"transcription_unit": ""}
        else:
            hyp = hyp_by_id[doc_id]
            # Guard against missing 'ocr_postcorrection_output' or 'transcription_unit'.
            postcorr = hyp.get("ocr_postcorrection_output")
            if postcorr is None or not isinstance(postcorr, dict):
                bad_structure.append(
                    f"  {doc_id}: missing key 'ocr_postcorrection_output'"
                )
                output = {"transcription_unit": ""}
            elif "transcription_unit" not in postcorr:
                bad_structure.append(
                    f"  {doc_id}: missing key "
                    f"'ocr_postcorrection_output.transcription_unit'"
                )
                output = {"transcription_unit": ""}
            elif postcorr["transcription_unit"] == "None":
                placeholder.append(doc_id)
                output = {"transcription_unit": ""}
            else:
                output = postcorr

        merged.append(
            {
                "document_metadata": ref["document_metadata"],
                "ground_truth": ref["ground_truth"],
                "ocr_hypothesis": ref["ocr_hypothesis"],
                "ocr_postcorrection_output": output,
            }
        )

    if missing:
        print(
            f"[WARNING] {len(missing)} reference document(s) not found in hypothesis, "
            f"scored with empty output: {missing}",
            file=sys.stderr,
        )

    if bad_structure:
        print(
            f"[WARNING] {len(bad_structure)} hypothesis document(s) have malformed "
            f"structure, scored with empty output:\n" + "\n".join(bad_structure),
            file=sys.stderr,
        )

    if placeholder:
        print(
            f"[WARNING] {len(placeholder)} document(s) have placeholder 'None' as "
            f"postcorrection output, scored with empty output: {placeholder}",
            file=sys.stderr,
        )

    return merged


def mer_from_counts(hits, substitutions, deletions, insertions):
    """Compute Match Error Rate (MER) from alignment counts.

    MER = (subs + dels + ins) / (hits + subs + dels + ins)

    Unlike CER/WER, MER is capped at [0, 1] because insertions are included
    in the denominator. This reduces sensitivity to hallucinations.
    Used for both character-level (cmer) and word-level (wmer) metrics.
    """
    total = hits + substitutions + deletions + insertions
    if total == 0:
        return 0.0
    return (substitutions + deletions + insertions) / total


def bootstrap_micro(
    dataset: Sequence[Sequence[float]],
    subset_aggr_fct: Callable[[Sequence[float]], float] = lambda x: sum(x[1:]) / sum(x),
) -> Tuple[float, float, float, List[float]]:
    """Compute a micro-averaged score with bootstrap confidence intervals.

    Args:
        dataset: Rows of per-example counts (hits, subs, dels, ins) or similar.
        subset_aggr_fct: Aggregation function applied to summed counts.

    Returns:
        (score, low_ci, high_ci, bootstrap_samples)
    """
    dataset = np.array(dataset)
    avgs = []
    for _ in range(10000):
        idx = np.random.randint(0, len(dataset), len(dataset))
        sample = dataset[idx]
        sample = sample.sum(axis=0)
        score = subset_aggr_fct(sample)
        avgs.append(score)
    lo = np.percentile(avgs, 2.5)
    hi = np.percentile(avgs, 97.5)
    dataset = dataset.sum(axis=0)
    mainscore = subset_aggr_fct(dataset)
    return mainscore, lo, hi, avgs


def bootstrap_simple(
    dataset: Sequence[float],
) -> Tuple[float, float, float, List[float]]:
    """Compute a macro-averaged score with bootstrap confidence intervals.

    Args:
        dataset: Per-example scalar scores.

    Returns:
        (score, low_ci, high_ci, bootstrap_samples)
    """
    dataset = np.array(dataset)
    avgs = []
    for _ in range(10000):
        idx = np.random.randint(0, len(dataset), len(dataset))
        sample = dataset[idx]
        score = np.mean(sample)
        avgs.append(score)
    lo = np.percentile(avgs, 2.5)
    hi = np.percentile(avgs, 97.5)
    mainscore = np.mean(dataset)
    return mainscore, lo, hi, avgs


class Evaluation:
    """Evaluate OCR post-correction outputs with MER and preference metrics.

    This class computes document-level and dataset-level metrics comparing
    post-correction output to ground truth, optionally against the raw OCR
    hypothesis. It supports normalisation, stratification into folds, and
    bootstrapped confidence intervals for aggregate scores.

    Metrics use Match Error Rate (MER) at both character level (cmer) and
    word level (wmer). MER = (S+D+I) / (H+S+D+I), capped at [0, 1].

    Expected example structure:
    - ground_truth: {transcription_unit: str}
    - ocr_postcorrection_output: {transcription_unit: str}
    - ocr_hypothesis: {transcription_unit: str}
    - document_metadata: {primary_dataset_name: str}

    Primary outputs:
    - fold_scores: per-stratum metrics with (mean, low_ci, high_ci)
    - averaged_scores: mean across folds with pooled CI
    """

    def __init__(
        self,
        list_of_example_dicts: List[Dict[str, Any]],
    ) -> None:
        """..."""
        if not list_of_example_dicts:
            raise ValueError("No documents to evaluate.")
        np.random.seed(42)
        self.data = list_of_example_dicts
        self._init_example_level_measures()
        self._init_dataset_level_measures()
        self.data_stratified = {}
        self.target_unit_key = "transcription_unit"

    def _init_example_level_measures(self) -> None:
        """Register functions that compute per-example metrics."""

        def cmer_sys(example):
            output = process_characters(
                example["ground_truth"][self.target_unit_key],
                example["ocr_postcorrection_output"][self.target_unit_key],
            )
            return mer_from_counts(
                output.hits, output.substitutions, output.deletions, output.insertions
            )

        def cmer_hyp(example):
            output = process_characters(
                example["ground_truth"][self.target_unit_key],
                example["ocr_hypothesis"][self.target_unit_key],
            )
            return mer_from_counts(
                output.hits, output.substitutions, output.deletions, output.insertions
            )

        def wmer_sys(example):
            output = process_words(
                example["ground_truth"][self.target_unit_key],
                example["ocr_postcorrection_output"][self.target_unit_key],
            )
            return mer_from_counts(
                output.hits, output.substitutions, output.deletions, output.insertions
            )

        def wmer_hyp(example):
            output = process_words(
                example["ground_truth"][self.target_unit_key],
                example["ocr_hypothesis"][self.target_unit_key],
            )
            return mer_from_counts(
                output.hits, output.substitutions, output.deletions, output.insertions
            )

        def cmer_stats(ex):
            output = process_characters(
                ex["ground_truth"][self.target_unit_key],
                ex["ocr_postcorrection_output"][self.target_unit_key],
            )
            return (
                output.hits,
                output.substitutions,
                output.deletions,
                output.insertions,
            )

        def wmer_stats(ex):
            output = process_words(
                ex["ground_truth"][self.target_unit_key],
                ex["ocr_postcorrection_output"][self.target_unit_key],
            )
            return (
                output.hits,
                output.substitutions,
                output.deletions,
                output.insertions,
            )

        def pref_score(ex, fct1, fct2):
            s1 = fct1(ex)
            s2 = fct2(ex)
            if s1 < s2:
                return 1
            if s1 > s2:
                return -1
            return 0

        def pcis(ex, fct1, fct2):
            s1 = 1 - fct1(ex)
            s2 = 1 - fct2(ex)
            if s2 == 0:
                return min(1, max(-1, s1))
            normalized_difference = (s1 - s2) / s2
            return normalized_difference

        pref_score_cmer = lambda example: pref_score(example, cmer_sys, cmer_hyp)
        pref_score_wmer = lambda example: pref_score(example, wmer_sys, wmer_hyp)

        pcis_cmer = lambda example: pcis(example, cmer_sys, cmer_hyp)
        pcis_wmer = lambda example: pcis(example, wmer_sys, wmer_hyp)

        self.example_level_measures = {
            "cmer": cmer_sys,
            "wmer": wmer_sys,
            "cmer_stats": cmer_stats,
            "wmer_stats": wmer_stats,
            "pref_score_cmer": pref_score_cmer,
            "pref_score_wmer": pref_score_wmer,
            "pcis_cmer": pcis_cmer,
            "pcis_wmer": pcis_wmer,
        }

    def _init_dataset_level_measures(self) -> None:
        """Register functions that compute dataset-level metrics."""

        def get_cmer_stats_data(ls):
            stats = np.array(
                [self.example_level_measures["cmer_stats"](ex) for ex in ls]
            )
            return stats

        def get_wmer_stats_data(ls):
            stats = np.array(
                [self.example_level_measures["wmer_stats"](ex) for ex in ls]
            )
            return stats

        cmer_micro = lambda ls: bootstrap_micro(get_cmer_stats_data(ls))
        wmer_micro = lambda ls: bootstrap_micro(get_wmer_stats_data(ls))

        cmer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["cmer"](ex) for ex in ls]
        )
        wmer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["wmer"](ex) for ex in ls]
        )

        pref_score_cmer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["pref_score_cmer"](ex) for ex in ls]
        )
        pref_score_wmer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["pref_score_wmer"](ex) for ex in ls]
        )

        pcis_cmer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["pcis_cmer"](ex) for ex in ls]
        )
        pcis_wmer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["pcis_wmer"](ex) for ex in ls]
        )

        self.dataset_level_measures = {
            "cmer_micro": cmer_micro,
            "wmer_micro": wmer_micro,
            "cmer_macro": cmer_macro,
            "wmer_macro": wmer_macro,
            "pref_score_cmer_macro": pref_score_cmer_macro,
            "pref_score_wmer_macro": pref_score_wmer_macro,
            "pcis_cmer_macro": pcis_cmer_macro,
            "pcis_wmer_macro": pcis_wmer_macro,
        }

    def _normalize(self) -> None:
        """Normalize strings for evaluation."""

        new_target_unit_key = self.target_unit_key + "_normalized"

        for example in self.data:
            gt = example["ground_truth"][self.target_unit_key]
            ocr = example["ocr_postcorrection_output"][self.target_unit_key]
            hyp = example["ocr_hypothesis"][self.target_unit_key]
            example["ground_truth"][new_target_unit_key] = norm(gt)
            example["ocr_postcorrection_output"][new_target_unit_key] = norm(ocr)
            example["ocr_hypothesis"][new_target_unit_key] = norm(hyp)

        self.target_unit_key = new_target_unit_key

    def _stratify(
        self,
        mapping: Callable[[Dict[str, Any]], str],
    ) -> None:
        """Group examples into folds using the provided mapping function."""
        self.data_stratified = defaultdict(list)
        set_id = []
        for dic in self.data:
            set_id.append(mapping(dic))
        for i, example in enumerate(self.data):
            self.data_stratified[set_id[i]].append(example)

    def score(
        self,
    ) -> Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]]:
        """Compute fold and averaged scores for all dataset-level metrics."""
        fold_scores = {}
        for fold in self.data_stratified:
            fold_scores[fold] = {}
            data = self.data_stratified[fold]
            for scorename, fct in self.dataset_level_measures.items():
                score = fct(data)
                fold_scores[fold][scorename] = score
        output = {"averaged_scores": {}, "fold_scores": fold_scores}
        for scorename in self.dataset_level_measures:
            scores = [fold_scores[fold][scorename][0] for fold in fold_scores]
            all_sampled_scores = [
                fold_scores[fold][scorename][3] for fold in fold_scores
            ]
            all_sampled_avgs = np.array(all_sampled_scores).mean(axis=0)
            lo = np.percentile(all_sampled_avgs, 2.5)
            hi = np.percentile(all_sampled_avgs, 97.5)
            avg = (sum(scores) / len(fold_scores), lo, hi)
            output["averaged_scores"][scorename] = avg
        for fold in output["fold_scores"]:
            for metric_name in output["fold_scores"][fold]:
                nums = output["fold_scores"][fold][metric_name]
                output["fold_scores"][fold][metric_name] = (
                    float(nums[0]),
                    float(nums[1]),
                    float(nums[2]),
                )
        for metric_name in output["averaged_scores"]:
            nums = output["averaged_scores"][metric_name]
            output["averaged_scores"][metric_name] = (
                float(nums[0]),
                float(nums[1]),
                float(nums[2]),
            )
        return output

    def score_over_datasets(
        self,
        normalize: bool = True,
    ) -> Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]]:
        """Normalize, stratify by dataset name, and score.

        Args:
            normalize: Whether to normalize transcription text before scoring.
        """
        if normalize:
            self._normalize()
        self._stratify(
            lambda example: example["document_metadata"]["primary_dataset_name"]
        )
        output = self.score()
        return output

    @staticmethod
    def scores2latex(
        scoredict: Dict[str, Any],
        systemname: str,
    ) -> str:
        """Render a LaTeX table from a score dictionary."""
        averaged_scores = scoredict.pop("averaged_scores")
        scoredict["fold_scores"]["AVG"] = averaged_scores
        metric_names = list(scoredict["fold_scores"]["AVG"].keys())
        tablelines = []
        for i, mn in enumerate(metric_names):
            metric_names_keep = [mn]
            result = deepcopy(scoredict)
            metric_names = list(result["fold_scores"]["AVG"].keys())
            folds = sorted(list(result["fold_scores"].keys()), reverse=True)
            for fold in folds:
                for metric_name in metric_names:
                    if metric_name not in metric_names_keep:
                        result["fold_scores"][fold].pop(metric_name)
            header = []
            for fold in folds:
                header.append(
                    "\\multicolumn{{{}}}{{{}}}{{{}}}".format(
                        len(result["fold_scores"][fold]), "c", fold
                    )
                )
            header = " & ".join(header)
            content = []
            for fold in folds:
                for metric_name in result["fold_scores"][fold]:
                    score, lo, hi = result["fold_scores"][fold][metric_name]
                    content.append(
                        "$_{{\\text{{{:.2f}}}}}\\text{{{:.2f}}}_{{\\text{{{:.2f}}}}}$"
                        .format(lo, score, hi)
                    )
            content = " & ".join(content)
            if i == 0:
                tablelines.append(header + "\\\\")
                tablelines.append(
                    " & ".join([systemname, mn.replace("_", "-")])
                    + " & "
                    + content
                    + "\\\\"
                )
            else:
                tablelines.append(
                    " & ".join(["", mn.replace("_", "-")]) + " & " + content + "\\\\"
                )
        return "\n".join(tablelines)
