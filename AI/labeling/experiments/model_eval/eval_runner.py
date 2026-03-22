#!/usr/bin/env python3
# labeling/experiments/model_eval/eval_runner.py
"""
모델 기반 피처 공통 평가 러너

모든 feature별 evaluator가 상속받는 베이스 클래스.
단독으로도 실행 가능:

    python labeling/experiments/model_eval/eval_runner.py \
        --ground_truth labeling/experiments/data/ground_truth.jsonl \
        --predictions  labeling/output \
        --feature      frame
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# ── sys.path 설정 ─────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_AI   = _HERE.parents[3]
if str(_AI) not in sys.path:
    sys.path.insert(0, str(_AI))

from labeling.experiments.utils.metrics import (
    accuracy, f1_macro, f1_per_class, cohen_kappa, confusion_matrix_str,
    pearson_r, spearman_rho, mae,
    precision_at_k, recall_at_k, f1_at_k,
)
from labeling.experiments.utils.report import ExperimentResult, save_result, print_compare
from labeling.experiments.utils.data_loader import load_ground_truth, load_labeled_dir, extract_pairs


# ──────────────────────────────────────────────
# 기본 평가 함수 모음
# ──────────────────────────────────────────────

def eval_categorical(
    y_true: list[Any],
    y_pred: list[Any],
    labels: list[Any] | None = None,
) -> dict:
    """범주형 피처 평가 (frame, logic, omission_risk 등)"""
    metrics: dict = {}
    metrics["accuracy"]    = accuracy(y_true, y_pred)
    metrics["f1_macro"]    = f1_macro(y_true, y_pred, labels)
    metrics["cohen_kappa"] = cohen_kappa(y_true, y_pred)
    metrics["n"]           = len(y_true)
    metrics["per_class"]   = f1_per_class(y_true, y_pred, labels)
    metrics["confusion_matrix"] = confusion_matrix_str(y_true, y_pred, labels)
    return metrics


def eval_continuous(
    y_true: list[float],
    y_pred: list[float],
) -> dict:
    """연속값 피처 평가 (stance_score, bias_x, bias_y, body_depth)"""
    return {
        "pearson":  pearson_r(y_true, y_pred),
        "spearman": spearman_rho(y_true, y_pred),
        "mae":      mae(y_true, y_pred),
        "n":        len(y_true),
    }


def eval_list_feature(
    y_true_sets: list[set],
    y_pred_lists: list[list],
    k: int | None = None,
) -> dict:
    """리스트형 피처 평가 (loaded_words, cmt_words)"""
    return {
        "precision": precision_at_k(y_true_sets, y_pred_lists, k),
        "recall":    recall_at_k(y_true_sets, y_pred_lists, k),
        "f1":        f1_at_k(y_true_sets, y_pred_lists, k),
        "n":         len(y_true_sets),
    }


# ──────────────────────────────────────────────
# 오류 분석 헬퍼
# ──────────────────────────────────────────────

def top_errors(
    ids: list[str],
    y_true: list[Any],
    y_pred: list[Any],
    texts: list[str] | None = None,
    n: int = 10,
) -> list[dict]:
    """
    예측이 틀린 샘플 상위 n개 반환.
    (연속값은 오차 큰 순, 범주형은 틀린 것 모두 반환 후 n개 자름)
    """
    errors = []
    for i, (tid, t, p) in enumerate(zip(ids, y_true, y_pred)):
        if isinstance(t, float) and isinstance(p, float):
            err = abs(t - p)
            errors.append({"id": tid, "true": t, "pred": p, "error": round(err, 4),
                           "text_snippet": (texts[i][:200] if texts else "")})
        elif t != p:
            errors.append({"id": tid, "true": t, "pred": p,
                           "text_snippet": (texts[i][:200] if texts else "")})

    # 연속값: 오차 큰 순 정렬
    if errors and isinstance(errors[0].get("error"), float):
        errors.sort(key=lambda x: x["error"], reverse=True)

    return errors[:n]


# ──────────────────────────────────────────────
# 베이스 Evaluator
# ──────────────────────────────────────────────

class BaseEvaluator(ABC):
    """피처별 Evaluator 베이스 클래스"""

    FEATURE: str = ""           # 오버라이드 필수
    PRED_KEY: str = ""          # predictions dict에서 읽을 키 (None이면 FEATURE 사용)
    GT_KEY:   str = ""          # ground_truth labels dict에서 읽을 키

    def __init__(self, out_dir: str = "labeling/experiments/results"):
        self.out_dir = out_dir

    @abstractmethod
    def compute_metrics(self, y_true: list, y_pred: list) -> dict:
        """메트릭 계산 — 서브클래스에서 구현"""
        ...

    def evaluate(
        self,
        ground_truth_path: str,
        predictions_dir:   str,
        run_name:          str = "default",
    ) -> ExperimentResult:
        """
        ground_truth JSONL + predictions 디렉토리로 평가 실행.
        Returns: ExperimentResult
        """
        gt_records  = load_ground_truth(ground_truth_path)
        pred_records = load_labeled_dir(predictions_dir)

        pred_key = self.PRED_KEY or self.FEATURE
        gt_key   = self.GT_KEY   or self.FEATURE

        y_true, y_pred, ids = extract_pairs(gt_records, pred_records, gt_key, pred_key)

        if not y_true:
            print(f"  [경고] {self.FEATURE}: 매칭되는 샘플 없음 (ground_truth에 '{gt_key}' 레이블 있는지 확인)")
            return ExperimentResult(
                name=run_name, feature=self.FEATURE, config={},
                metrics={"error": "no_samples"}, n_samples=0,
            )

        print(f"\n[{self.FEATURE}] 평가 시작 — {len(y_true)}개 샘플")
        metrics = self.compute_metrics(y_true, y_pred)

        # per_class / confusion_matrix는 별도 출력
        per_class = metrics.pop("per_class", None)
        conf_mat  = metrics.pop("confusion_matrix", None)

        result = ExperimentResult(
            name=run_name,
            feature=self.FEATURE,
            config={"ground_truth": ground_truth_path, "predictions": predictions_dir},
            metrics=metrics,
            n_samples=len(y_true),
            predictions=[{"id": i, "true": t, "pred": p}
                         for i, t, p in zip(ids, y_true, y_pred)],
        )

        # 콘솔 출력
        print(result.summary())
        if per_class:
            print("\n  [클래스별 F1]")
            for lbl, m in per_class.items():
                print(f"    {str(lbl):<20} P={m['precision']:.3f}  R={m['recall']:.3f}  "
                      f"F1={m['f1']:.3f}  (n={m['support']})")
        if conf_mat:
            print("\n  [혼동 행렬]")
            for line in conf_mat.split("\n"):
                print("    " + line)

        # 오류 분석
        texts_by_id = {str(r.get("id","")): r.get("text","") for r in gt_records}
        texts_list = [texts_by_id.get(i,"") for i in ids]
        errors = top_errors(ids, y_true, y_pred, texts_list, n=5)
        if errors:
            print("\n  [주요 오류 샘플 (상위 5개)]")
            for e in errors:
                print(f"    id={e['id']}  true={e['true']}  pred={e['pred']}")
                if e.get("text_snippet"):
                    print(f"    {e['text_snippet'][:120]}...")

        save_result(result, self.out_dir)
        return result


# ──────────────────────────────────────────────
# CLI (단순 평가 래퍼)
# ──────────────────────────────────────────────

_FEATURE_TYPES = {
    "frame":        "categorical",
    "logic":        "categorical",
    "omission_risk":"categorical",
    "stance_label": "categorical",
    "body_depth":   "continuous",
    "stance_score": "continuous",
    "bias_x":       "continuous",
    "bias_y":       "continuous",
    "loaded_words": "list",
    "art_words":    "list",
}

_OMISSION_MAP  = {"low": -1, "mid": 0, "high": 1}


def _quick_eval(ground_truth_path: str, predictions_dir: str, feature: str, run_name: str, out_dir: str):
    """feature에 맞는 메트릭 자동 선택으로 빠른 평가"""
    gt_records   = load_ground_truth(ground_truth_path)
    pred_records = load_labeled_dir(predictions_dir)

    # frame 레이블 매핑
    _FRAME_MAP = {
        "사건 원인 집중": 1, "갈등/대립 강조": 2, "개인 사례 중심": 3,
        "경제적 영향 강조": 4, "윤리/도덕 판단": 5, "안전/안보 위협": 6, "권리/인권 강조": 7,
    }
    _LOGIC_MAP = {
        "정책적 비난": 1, "전문가 견해": 2, "피해자 서사": 3,
        "파급효과": 4, "해결책 제시": 5, "사실/정보 전달": 6,
    }

    y_true, y_pred, ids = extract_pairs(gt_records, pred_records, feature)

    if not y_true:
        print(f"  [경고] '{feature}' 레이블이 없거나 매칭 불가.")
        return

    feat_type = _FEATURE_TYPES.get(feature, "categorical")
    print(f"\n[{feature}] {feat_type} 평가 — {len(y_true)}개 샘플")

    if feat_type == "continuous":
        metrics = eval_continuous([float(t) for t in y_true], [float(p) for p in y_pred])
    elif feat_type == "list":
        y_true_sets  = [set(t) if isinstance(t, list) else set() for t in y_true]
        y_pred_lists = [p if isinstance(p, list) else [] for p in y_pred]
        metrics = eval_list_feature(y_true_sets, y_pred_lists)
    else:
        metrics = eval_categorical(y_true, y_pred)
        per_class = metrics.pop("per_class", {})
        conf_mat  = metrics.pop("confusion_matrix", "")
        for lbl, m in per_class.items():
            print(f"  {str(lbl):<25} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        print(conf_mat)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<15} {v:.4f}")
        else:
            print(f"  {k:<15} {v}")

    result = ExperimentResult(
        name=run_name, feature=feature, config={},
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        n_samples=len(y_true),
    )
    save_result(result, out_dir)


def _cli():
    parser = argparse.ArgumentParser(description="모델 피처 빠른 평가")
    parser.add_argument("--ground_truth",  required=True)
    parser.add_argument("--predictions",   required=True)
    parser.add_argument("--feature",       required=True, choices=list(_FEATURE_TYPES.keys()))
    parser.add_argument("--run_name",      default="eval")
    parser.add_argument("--out_dir",       default="labeling/experiments/results")
    args = parser.parse_args()
    _quick_eval(args.ground_truth, args.predictions, args.feature,
                args.run_name, args.out_dir)


if __name__ == "__main__":
    _cli()
