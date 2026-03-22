#!/usr/bin/env python3
# labeling/experiments/model_eval/frame_logic_eval.py
"""
frame / logic 라벨링 성능 평가

실행:
    python labeling/experiments/model_eval/frame_logic_eval.py \
        --ground_truth labeling/experiments/data/ground_truth.jsonl \
        --predictions  labeling/output

비교 실험 (여러 모델 결과 비교):
    python labeling/experiments/model_eval/frame_logic_eval.py \
        --ground_truth labeling/experiments/data/ground_truth.jsonl \
        --predictions  labeling/output \
        --compare_dir  labeling/experiments/results \
        --run_name     llama3.3_70b
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
    accuracy, f1_macro, f1_per_class, cohen_kappa, confusion_matrix_str,
)
from labeling.experiments.utils.report import (
    ExperimentResult, save_result, load_results, compare_results, print_compare,
)
from labeling.experiments.utils.data_loader import load_ground_truth, load_labeled_dir, extract_pairs
from labeling.experiments.model_eval.eval_runner import BaseEvaluator


# ──────────────────────────────────────────────
# 레이블 정의
# ──────────────────────────────────────────────

FRAME_LABELS = [
    "사건 원인 집중", "갈등/대립 강조", "개인 사례 중심",
    "경제적 영향 강조", "윤리/도덕 판단", "안전/안보 위협", "권리/인권 강조",
]
FRAME_INT_LABELS = list(range(1, 8))  # 1~7

LOGIC_LABELS = [
    "정책적 비난", "전문가 견해", "피해자 서사",
    "파급효과", "해결책 제시", "사실/정보 전달",
]
LOGIC_INT_LABELS = list(range(1, 7))  # 1~6


# ──────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────

class FrameEvaluator(BaseEvaluator):
    FEATURE  = "frame"
    PRED_KEY = "frame"     # predictions에서 읽을 키
    GT_KEY   = "frame"     # ground_truth labels에서 읽을 키

    def compute_metrics(self, y_true, y_pred) -> dict:
        # int 또는 str 모두 허용
        labels = FRAME_INT_LABELS if isinstance(y_true[0], int) else FRAME_LABELS
        return {
            "accuracy":         accuracy(y_true, y_pred),
            "f1_macro":         f1_macro(y_true, y_pred, labels),
            "cohen_kappa":      cohen_kappa(y_true, y_pred),
            "per_class":        f1_per_class(y_true, y_pred, labels),
            "confusion_matrix": confusion_matrix_str(y_true, y_pred, labels),
            "n":                len(y_true),
        }


class LogicEvaluator(BaseEvaluator):
    FEATURE  = "logic"
    PRED_KEY = "logic"
    GT_KEY   = "logic"

    def compute_metrics(self, y_true, y_pred) -> dict:
        labels = LOGIC_INT_LABELS if isinstance(y_true[0], int) else LOGIC_LABELS
        return {
            "accuracy":         accuracy(y_true, y_pred),
            "f1_macro":         f1_macro(y_true, y_pred, labels),
            "cohen_kappa":      cohen_kappa(y_true, y_pred),
            "per_class":        f1_per_class(y_true, y_pred, labels),
            "confusion_matrix": confusion_matrix_str(y_true, y_pred, labels),
            "n":                len(y_true),
        }


# ──────────────────────────────────────────────
# 모델 비교 실험
# ──────────────────────────────────────────────

def compare_models(
    ground_truth_path: str,
    prediction_dirs: dict[str, str],  # {run_name: predictions_dir}
    out_dir: str,
):
    """
    여러 모델(또는 프롬프트) 결과를 한 번에 비교.

    Args:
        prediction_dirs: {"llama3_70b": "output/llama3_70b", "qwen3": "output/qwen3", ...}
    """
    print("\n" + "=" * 70)
    print("  frame / logic 모델 비교 실험")
    print("=" * 70)

    frame_results = []
    logic_results = []

    for run_name, pred_dir in prediction_dirs.items():
        frame_eval = FrameEvaluator(out_dir=out_dir)
        logic_eval = LogicEvaluator(out_dir=out_dir)

        r_frame = frame_eval.evaluate(ground_truth_path, pred_dir, run_name=run_name)
        r_logic = logic_eval.evaluate(ground_truth_path, pred_dir, run_name=run_name)

        frame_results.append(r_frame)
        logic_results.append(r_logic)

    print("\n[frame 비교]")
    print_compare(frame_results, sort_by="f1_macro")

    print("\n[logic 비교]")
    print_compare(logic_results, sort_by="f1_macro")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="frame/logic 평가")
    parser.add_argument("--ground_truth",  required=True)
    parser.add_argument("--predictions",   required=True)
    parser.add_argument("--run_name",      default="default")
    parser.add_argument("--out_dir",       default="labeling/experiments/results")
    parser.add_argument("--compare_dir",   default=None,
                        help="이전 실험 결과 디렉토리 (비교용)")
    args = parser.parse_args()

    # frame 평가
    frame_eval = FrameEvaluator(out_dir=args.out_dir)
    r_frame = frame_eval.evaluate(args.ground_truth, args.predictions, args.run_name)

    # logic 평가
    logic_eval = LogicEvaluator(out_dir=args.out_dir)
    r_logic = logic_eval.evaluate(args.ground_truth, args.predictions, args.run_name)

    # 이전 결과와 비교
    if args.compare_dir:
        print("\n[이전 결과와 비교 — frame]")
        prev_frame = load_results(args.compare_dir, feature="frame")
        print_compare(prev_frame + [r_frame], sort_by="f1_macro")

        print("\n[이전 결과와 비교 — logic]")
        prev_logic = load_results(args.compare_dir, feature="logic")
        print_compare(prev_logic + [r_logic], sort_by="f1_macro")


if __name__ == "__main__":
    _cli()
