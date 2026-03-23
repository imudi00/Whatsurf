#!/usr/bin/env python3
# labeling/experiments/model_eval/stance_omission_eval.py
"""
stance_score / omission_risk 성능 평가

stance_score  : -1.0(비판적) ~ +1.0(우호적) — 연속값
omission_risk : low/mid/high (또는 -1/0/1) — 범주형

실행:
    # stance 평가
    python labeling/experiments/model_eval/stance_omission_eval.py \
        --ground_truth labeling/experiments/data/ground_truth.jsonl \
        --predictions  labeling/output \
        --feature stance

    # omission 평가
    python labeling/experiments/model_eval/stance_omission_eval.py \
        --ground_truth labeling/experiments/data/ground_truth.jsonl \
        --predictions  labeling/output \
        --feature omission
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_AI   = _HERE.parents[3]
if str(_AI) not in sys.path:
    sys.path.insert(0, str(_AI))

from labeling.experiments.utils.metrics import (
    pearson_r, spearman_rho, mae,
    accuracy, f1_macro, f1_per_class, cohen_kappa, confusion_matrix_str,
)
from labeling.experiments.utils.report import ExperimentResult, save_result, load_results, print_compare
from labeling.experiments.utils.data_loader import load_ground_truth, load_labeled_dir, extract_pairs
from labeling.experiments.model_eval.eval_runner import BaseEvaluator, top_errors


# ──────────────────────────────────────────────
# stance_score 평가
# ──────────────────────────────────────────────

class StanceEvaluator(BaseEvaluator):
    FEATURE  = "stance_score"
    PRED_KEY = "stance_score"
    GT_KEY   = "stance_score"

    def compute_metrics(self, y_true, y_pred) -> dict:
        y_t = [float(v) for v in y_true]
        y_p = [float(v) for v in y_pred]

        # 방향 분류 (비판/중립/우호)
        def to_tone(v: float) -> str:
            if v > 0.15:  return "우호적"
            if v < -0.15: return "비판적"
            return "중립"

        dir_true = [to_tone(v) for v in y_t]
        dir_pred = [to_tone(v) for v in y_p]
        labels   = ["비판적", "중립", "우호적"]

        metrics = {
            "pearson":        pearson_r(y_t, y_p),
            "spearman":       spearman_rho(y_t, y_p),
            "mae":            mae(y_t, y_p),
            "tone_accuracy":  accuracy(dir_true, dir_pred),
            "tone_f1_macro":  f1_macro(dir_true, dir_pred, labels),
            "tone_kappa":     cohen_kappa(dir_true, dir_pred),
            "per_class":      f1_per_class(dir_true, dir_pred, labels),
            "confusion_matrix": confusion_matrix_str(dir_true, dir_pred, labels),
            "n": len(y_t),
        }
        return metrics


# ──────────────────────────────────────────────
# omission_risk 평가
# ──────────────────────────────────────────────

_OMISSION_NORMALIZE = {
    "low":  -1, "mid":  0, "medium": 0, "high":  1,
    -1:     -1,  0:     0,  1:       1,
}

class OmissionEvaluator(BaseEvaluator):
    FEATURE  = "omission_risk"
    PRED_KEY = "omission_risk"
    GT_KEY   = "omission_risk"

    def compute_metrics(self, y_true, y_pred) -> dict:
        # str → int 정규화 (-1/0/1)
        y_t = [_OMISSION_NORMALIZE.get(v, v) for v in y_true]
        y_p = [_OMISSION_NORMALIZE.get(v, v) for v in y_pred]
        labels = [-1, 0, 1]

        return {
            "accuracy":         accuracy(y_t, y_p),
            "f1_macro":         f1_macro(y_t, y_p, labels),
            "cohen_kappa":      cohen_kappa(y_t, y_p),
            "per_class":        f1_per_class(y_t, y_p, labels),
            "confusion_matrix": confusion_matrix_str(y_t, y_p, labels),
            "n": len(y_t),
        }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="stance / omission 평가")
    parser.add_argument("--ground_truth",  required=True)
    parser.add_argument("--predictions",   required=True)
    parser.add_argument("--feature",       choices=["stance", "omission", "both"], default="both")
    parser.add_argument("--run_name",      default="default")
    parser.add_argument("--out_dir",       default="labeling/experiments/results")
    parser.add_argument("--compare_dir",   default=None)
    args = parser.parse_args()

    if args.feature in ("stance", "both"):
        ev = StanceEvaluator(out_dir=args.out_dir)
        r  = ev.evaluate(args.ground_truth, args.predictions, args.run_name)
        if args.compare_dir:
            prev = load_results(args.compare_dir, feature="stance_score")
            print_compare(prev + [r], sort_by="pearson")

    if args.feature in ("omission", "both"):
        ev = OmissionEvaluator(out_dir=args.out_dir)
        r  = ev.evaluate(args.ground_truth, args.predictions, args.run_name)
        if args.compare_dir:
            prev = load_results(args.compare_dir, feature="omission_risk")
            print_compare(prev + [r], sort_by="f1_macro")


if __name__ == "__main__":
    _cli()
