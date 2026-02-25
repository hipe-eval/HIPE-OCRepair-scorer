from collections import defaultdict
import numpy as np
from jiwer import wer, cer, process_characters, process_words
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Sequence, Tuple


def bootstrap_micro(
    dataset: Sequence[Sequence[float]],
    subset_aggr_fct: Callable[[Sequence[float]], float] = lambda x: sum(x[1:]) / sum(x[:-1]),
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


class Evaluation():
    """Evaluate OCR post-correction outputs with CER/WER and preference metrics.

    This class computes document-level (JSON) and dataset-level metrics comparing
    post-correction output to ground truth, optionally against the raw OCR
    hypothesis. It supports normalisation, stratification into folds, and
    bootstrapped confidence intervals for aggregate scores.

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
        """Initialize the evaluator with example dictionaries.

        Each example is expected to include:
        - ground_truth: {transcription_unit: str}
        - ocr_postcorrection_output: {transcription_unit: str}
        - ocr_hypothesis: {transcription_unit: str}
        - document_metadata: {primary_dataset_name: str}
        """
        self.data = list_of_example_dicts
        self._init_example_level_measures()
        self._init_dataset_level_measures()
        self.data_stratified = {}
        self.target_unit_key = "transcription_unit"

    def _init_example_level_measures(self) -> None:
        """Register functions that compute per-example metrics."""
        cer_sys = lambda example: cer(example["ground_truth"][self.target_unit_key],
                                      example["ocr_postcorrection_output"][
                                          self.target_unit_key])
        wer_sys = lambda example: wer(example["ground_truth"][self.target_unit_key],
                                      example["ocr_postcorrection_output"][
                                          self.target_unit_key])

        cer_hyp = lambda example: cer(example["ground_truth"][self.target_unit_key],
                                      example["ocr_hypothesis"][self.target_unit_key])
        wer_hyp = lambda example: wer(example["ground_truth"][self.target_unit_key],
                                      example["ocr_hypothesis"][self.target_unit_key])

        def cer_stats(ex):
            output = process_characters(ex["ground_truth"][self.target_unit_key],
                                        ex["ocr_postcorrection_output"][
                                            self.target_unit_key])
            stats = output.hits, output.substitutions, output.deletions, output.insertions
            assert output.cer == sum(stats[1:]) / sum(stats[:-1])
            return stats

        def wer_stats(ex):
            output = process_words(ex["ground_truth"][self.target_unit_key],
                                   ex["ocr_postcorrection_output"][
                                       self.target_unit_key])
            stats = output.hits, output.substitutions, output.deletions, output.insertions
            assert output.wer == sum(stats[1:]) / sum(stats[:-1])
            return stats

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

        pref_score_cer = lambda example: pref_score(example, cer_sys, cer_hyp)
        pref_score_wer = lambda example: pref_score(example, wer_sys, wer_hyp)

        pcis_cer = lambda example: pcis(example, cer_sys, cer_hyp)
        pcis_wer = lambda example: pcis(example, wer_sys, wer_hyp)

        self.example_level_measures = {"cer": cer_sys,
                                       "wer": wer_sys,
                                       "cer_stats": cer_stats,
                                       "wer_stats": wer_stats,
                                       "pref_score_cer": pref_score_cer,
                                       "pref_score_wer": pref_score_wer,
                                       "pcis_cer": pcis_cer,
                                       "pcis_wer": pcis_wer}

    def _init_dataset_level_measures(self) -> None:
        """Register functions that compute dataset-level metrics."""

        def get_cer_stats_data(ls):
            stats = np.array(
                [self.example_level_measures["cer_stats"](ex) for ex in ls])
            return stats

        def get_wer_stats_data(ls):
            stats = np.array(
                [self.example_level_measures["wer_stats"](ex) for ex in ls])
            return stats

        cer_micro = lambda ls: bootstrap_micro(get_cer_stats_data(ls))
        wer_micro = lambda ls: bootstrap_micro(get_wer_stats_data(ls))

        cer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["cer"](ex) for ex in ls])
        wer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["wer"](ex) for ex in ls])

        pref_score_cer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["pref_score_cer"](ex) for ex in ls])
        pref_score_wer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["pref_score_wer"](ex) for ex in ls])

        pcis_cer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["pcis_cer"](ex) for ex in ls])
        pcis_wer_macro = lambda ls: bootstrap_simple(
            [self.example_level_measures["pcis_wer"](ex) for ex in ls])

        self.dataset_level_measures = {"cer_micro": cer_micro,
                                       "wer_micro": wer_micro,
                                       "cer_macro": cer_macro,
                                       "wer_macro": wer_macro,
                                       "pref_score_cer_macro": pref_score_cer_macro,
                                       "pref_score_wer_macro": pref_score_wer_macro,
                                       "pcis_cer_macro": pcis_cer_macro,
                                       "pcis_wer_macro": pcis_wer_macro,
                                       }

    def _normalize(self) -> None:
        """Normalize strings to lowercase alphanumerics and whitespace."""

        def norm(string):
            string = re.sub(r"[^a-z0-9]", " ", string)
            string = re.sub(r"\s+", " ", string)
            string = string.lower()
            return string

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
            all_sampled_scores = [fold_scores[fold][scorename][3] for fold in
                                  fold_scores]
            all_sampled_avgs = np.array(all_sampled_scores).mean(axis=0)
            lo = np.percentile(all_sampled_avgs, 2.5)
            hi = np.percentile(all_sampled_avgs, 97.5)
            avg = (sum(scores) / len(fold_scores), lo, hi)
            output["averaged_scores"][scorename] = avg
        for fold in output["fold_scores"]:
            for metric_name in output["fold_scores"][fold]:
                nums = output["fold_scores"][fold][metric_name]
                output["fold_scores"][fold][metric_name] = (float(round(nums[0], 2)),
                                                            float(round(nums[1], 2)),
                                                            float(round(nums[2], 2)))
        for metric_name in output["averaged_scores"]:
            nums = output["averaged_scores"][metric_name]
            output["averaged_scores"][metric_name] = (float(round(nums[0], 2)),
                                                      float(round(nums[1], 2)),
                                                      float(round(nums[2], 2)))
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
            lambda example: example["document_metadata"]["primary_dataset_name"])
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
                header.append("\\multicolumn{{{}}}{{{}}}{{{}}}".format(
                    len(result["fold_scores"][fold]), "c", fold))
            header = " & ".join(header)
            content = []
            for fold in folds:
                for metric_name in result["fold_scores"][fold]:
                    score, lo, hi = result["fold_scores"][fold][metric_name]
                    content.append(
                        "$_{{\\text{{{:.2f}}}}}\\text{{{:.2f}}}_{{\\text{{{:.2f}}}}}$".format(
                            lo, score, hi))
            content = " & ".join(content)
            if i == 0:
                tablelines.append(header + "\\\\")
                tablelines.append(" & ".join(
                    [systemname, mn.replace("_", "-")]) + " & " + content + "\\\\")
            else:
                tablelines.append(
                    " & ".join(["", mn.replace("_", "-")]) + " & " + content + "\\\\")
        return "\n".join(tablelines)
