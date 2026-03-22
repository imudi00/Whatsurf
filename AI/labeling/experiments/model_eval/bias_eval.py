#!/usr/bin/env python3
# labeling/experiments/model_eval/bias_eval.py
"""
bias_x / bias_y 성능 평가

bias_x : -1.0(진보) ~ 0.0(중립) ~ +1.0(보수)
bias_y : -1.0(감성) ~ 0.0(균형) ~ +1.0(사실)

메트릭:
  - 연속값: Pearson, Spearman, MAE
  - 사분면 정확도: (bias_x 부호 × bias_y 부호) → 4가지 중 맞춘 비율
  - 방향 정확도: 진보/중립/보수 3분류 정확도

실행:
    python labeling/experiments/model_eval/bias_eval.py \
        --ground_truth labeling/experiments/data/ground_truth.jsonl \
        --predictions  labeling/output
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
    pearson_r, spearman_rho, mae, accuracy, f1_macro, f1_per_class,
)
from labeling.experiments.utils.report import ExperimentResult, save_result, load_results, print_compare
from labeling.experiments.utils.data_loader import load_ground_truth, load_labeled_dir, extract_pairs
from labeling.experiments.model_eval.eval_runner import BaseEvaluator, top_errors


def _to_direction(x: float, threshold: float = 0.2) -> str:
    """연속값 → 방향 레이블"""
    if x > threshold:  return "right"
    if x < -threshold: return "left"
    return "center"


def _to_quadrant(x: float, y: float, thr: float = 0.1) -> str:
    """(bias_x, bias_y) → 사분면 문자열"""
    xi = "+" if x > thr else ("-" if x < -thr else "0")
    yi = "+" if y > thr else ("-" if y < -thr else "0")
    return f"x{xi}y{yi}"


class BiasEvaluator(BaseEvaluator):
    """bias_x + bias_y 동시 평가"""
    FEATURE = "bias"

    def compute_metrics(self, y_true, y_pred) -> dict:
        # bias_x / bias_y 별도 처리 — y_true/y_pred 는 (x, y) 튜플 리스트
        raise NotImplementedError("use evaluate_bias() directly")

    def evaluate_bias(
        self,
        ground_truth_path: str,
        predictions_dir:   str,
        run_name:          str = "default",
    ) -> ExperimentResult:
        gt_records   = load_ground_truth(ground_truth_path)
        pred_records = load_labeled_dir(predictions_dir)

        # bias_x
        tx, px, ids_x = extract_pairs(gt_records, pred_records, "bias_x")
        # bias_y
        ty, py, ids_y = extract_pairs(gt_records, pred_records, "bias_y")

        if not tx or not ty:
            print("  [경고] bias_x 또는 bias_y 레이블 없음")
            return ExperimentResult(name=run_name, feature="bias",
                                    config={}, metrics={"error": "no_samples"}, n_samples=0)

        tx = [float(v) for v in tx]
        px = [float(v) for v in px]
        ty = [float(v) for v in ty]
        py = [float(v) for v in py]

        print(f"\n[bias] 평가 — bias_x: {len(tx)}개  bias_y: {len(ty)}개")

        # 연속값 메트릭
        mx = {"pearson": pearson_r(tx, px), "spearman": spearman_rho(tx, px), "mae": mae(tx, px)}
        my = {"pearson": pearson_r(ty, py), "spearman": spearman_rho(ty, py), "mae": mae(ty, py)}

        print(f"\n  [bias_x] pearson={mx['pearson']:.4f}  spearman={mx['spearman']:.4f}  mae={mx['mae']:.4f}")
        print(f"  [bias_y] pearson={my['pearson']:.4f}  spearman={my['spearman']:.4f}  mae={my['mae']:.4f}")

        # 방향 정확도 (bias_x 기준)
        dir_true = [_to_direction(v) for v in tx]
        dir_pred = [_to_direction(v) for v in px]
        dir_acc  = accuracy(dir_true, dir_pred)
        dir_f1   = f1_macro(dir_true, dir_pred, ["left","center","right"])
        dir_pc   = f1_per_class(dir_true, dir_pred, ["left","center","right"])

        print(f"\n  [방향 정확도 — bias_x] acc={dir_acc:.4f}  f1_macro={dir_f1:.4f}")
        for lbl, m in dir_pc.items():
            print(f"    {lbl:<8} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")

        # 사분면 정확도
        quad_true = [_to_quadrant(x, y) for x, y in zip(tx, ty)]
        quad_pred = [_to_quadrant(x, y) for x, y in zip(px, py)]
        quad_acc  = accuracy(quad_true, quad_pred)
        print(f"\n  [사분면 정확도] acc={quad_acc:.4f}")

        # 오류 분석
        texts_by_id = {str(r.get("id","")): r.get("text","") for r in gt_records}
        texts_list = [texts_by_id.get(i,"") for i in ids_x]
        errors_x = top_errors(ids_x, tx, px, texts_list, n=3)
        if errors_x:
            print("\n  [bias_x 오차 큰 샘플]")
            for e in errors_x:
                print(f"    id={e['id']}  true={e['true']:.2f}  pred={e['pred']:.2f}  err={e['error']:.2f}")

        metrics = {
            "bias_x_pearson":   mx["pearson"],
            "bias_x_spearman":  mx["spearman"],
            "bias_x_mae":       mx["mae"],
            "bias_y_pearson":   my["pearson"],
            "bias_y_spearman":  my["spearman"],
            "bias_y_mae":       my["mae"],
            "dir_accuracy":     dir_acc,
            "dir_f1_macro":     dir_f1,
            "quadrant_accuracy":quad_acc,
        }

        result = ExperimentResult(
            name=run_name, feature="bias", config={},
            metrics=metrics, n_samples=min(len(tx), len(ty)),
        )
        save_result(result, self.out_dir)
        return result


def _cli():
    parser = argparse.ArgumentParser(description="bias_x/y 평가")
    parser.add_argument("--ground_truth",  required=True)
    parser.add_argument("--predictions",   required=True)
    parser.add_argument("--run_name",      default="default")
    parser.add_argument("--out_dir",       default="labeling/experiments/results")
    args = parser.parse_args()

    ev = BiasEvaluator(out_dir=args.out_dir)
    ev.evaluate_bias(args.ground_truth, args.predictions, args.run_name)


if __name__ == "__main__":
    _cli()
