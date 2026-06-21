# labeling/experiments/utils/metrics.py
"""
피처 평가용 메트릭 모음

연속값 (body_depth, stance_score, bias_x/y):
    pearson_r, spearman_rho, mae, rmse

범주형 (frame, logic, omission_risk):
    accuracy, f1_macro, f1_per_class, cohen_kappa, confusion_matrix_str

리스트형 (loaded_words, cmt_words):
    precision_at_k, recall_at_k, f1_at_k, jaccard
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


# ──────────────────────────────────────────────
# 연속값 메트릭
# ──────────────────────────────────────────────

def pearson_r(y_true: list[float], y_pred: list[float]) -> float:
    """피어슨 상관계수 (-1 ~ 1)"""
    n = len(y_true)
    if n < 2:
        return float("nan")
    mx = sum(y_true) / n
    my = sum(y_pred) / n
    num = sum((a - mx) * (b - my) for a, b in zip(y_true, y_pred))
    dx  = math.sqrt(sum((a - mx) ** 2 for a in y_true))
    dy  = math.sqrt(sum((b - my) ** 2 for b in y_pred))
    denom = dx * dy
    return round(num / denom, 4) if denom > 0 else float("nan")


def _rank(arr: list[float]) -> list[float]:
    """오름차순 순위 (동점은 평균 순위)"""
    sorted_vals = sorted(enumerate(arr), key=lambda x: x[1])
    ranks = [0.0] * len(arr)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) - 1 and sorted_vals[j + 1][1] == sorted_vals[j][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(y_true: list[float], y_pred: list[float]) -> float:
    """스피어만 순위 상관계수 (-1 ~ 1)"""
    if len(y_true) < 2:
        return float("nan")
    return pearson_r(_rank(y_true), _rank(y_pred))


def mae(y_true: list[float], y_pred: list[float]) -> float:
    """평균 절대 오차"""
    n = len(y_true)
    if n == 0:
        return float("nan")
    return round(sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n, 4)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    """루트 평균 제곱 오차"""
    n = len(y_true)
    if n == 0:
        return float("nan")
    return round(math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n), 4)


# ──────────────────────────────────────────────
# 범주형 메트릭
# ──────────────────────────────────────────────

def accuracy(y_true: list[Any], y_pred: list[Any]) -> float:
    """정확도"""
    if not y_true:
        return float("nan")
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    return round(correct / len(y_true), 4)


def f1_per_class(
    y_true: list[Any],
    y_pred: list[Any],
    labels: list[Any] | None = None,
) -> dict[Any, dict]:
    """클래스별 precision / recall / F1"""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    result = {}
    for lbl in labels:
        tp = sum(t == lbl and p == lbl for t, p in zip(y_true, y_pred))
        fp = sum(t != lbl and p == lbl for t, p in zip(y_true, y_pred))
        fn = sum(t == lbl and p != lbl for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[lbl] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   sum(t == lbl for t in y_true),
        }
    return result


def f1_macro(
    y_true: list[Any],
    y_pred: list[Any],
    labels: list[Any] | None = None,
) -> float:
    """macro-average F1"""
    per = f1_per_class(y_true, y_pred, labels)
    scores = [v["f1"] for v in per.values()]
    return round(sum(scores) / len(scores), 4) if scores else float("nan")


def cohen_kappa(y_true: list[Any], y_pred: list[Any]) -> float:
    """Cohen's Kappa (범주형 일치도)"""
    n = len(y_true)
    if n == 0:
        return float("nan")
    labels = list(set(y_true) | set(y_pred))
    # 관측 일치율
    po = sum(a == b for a, b in zip(y_true, y_pred)) / n
    # 기대 일치율
    tc = Counter(y_true)
    pc = Counter(y_pred)
    pe = sum((tc[l] / n) * (pc[l] / n) for l in labels)
    return round((po - pe) / (1 - pe), 4) if (1 - pe) > 0 else float("nan")


def confusion_matrix_str(
    y_true: list[Any],
    y_pred: list[Any],
    labels: list[Any] | None = None,
) -> str:
    """텍스트 형태 혼동 행렬"""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    lbl_idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        if t in lbl_idx and p in lbl_idx:
            matrix[lbl_idx[t]][lbl_idx[p]] += 1

    col_w = max(max(len(str(l)) for l in labels), 5) + 2
    header = " " * col_w + "".join(str(l).rjust(col_w) for l in labels) + "  (pred→)"
    lines = [header]
    for i, lbl in enumerate(labels):
        row = str(lbl).rjust(col_w) + "".join(str(matrix[i][j]).rjust(col_w) for j in range(n))
        lines.append(row)
    lines.append("(true↓)")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 리스트형 메트릭 (loaded_words 등)
# ──────────────────────────────────────────────

def precision_at_k(
    y_true_sets: list[set],
    y_pred_lists: list[list],
    k: int | None = None,
) -> float:
    """Precision@K: 예측 상위 k개 중 정답 비율 (평균)"""
    scores = []
    for true_set, pred_list in zip(y_true_sets, y_pred_lists):
        pred_k = set(pred_list[:k]) if k else set(pred_list)
        if not pred_k:
            scores.append(0.0)
        else:
            scores.append(len(true_set & pred_k) / len(pred_k))
    return round(sum(scores) / len(scores), 4) if scores else float("nan")


def recall_at_k(
    y_true_sets: list[set],
    y_pred_lists: list[list],
    k: int | None = None,
) -> float:
    """Recall@K: 정답 중 예측 상위 k개에 포함된 비율 (평균)"""
    scores = []
    for true_set, pred_list in zip(y_true_sets, y_pred_lists):
        pred_k = set(pred_list[:k]) if k else set(pred_list)
        if not true_set:
            scores.append(1.0)  # 정답이 없으면 완벽
        else:
            scores.append(len(true_set & pred_k) / len(true_set))
    return round(sum(scores) / len(scores), 4) if scores else float("nan")


def f1_at_k(
    y_true_sets: list[set],
    y_pred_lists: list[list],
    k: int | None = None,
) -> float:
    p = precision_at_k(y_true_sets, y_pred_lists, k)
    r = recall_at_k(y_true_sets, y_pred_lists, k)
    if math.isnan(p) or math.isnan(r) or (p + r) == 0:
        return float("nan")
    return round(2 * p * r / (p + r), 4)


def jaccard(
    y_true_sets: list[set],
    y_pred_sets: list[set],
) -> float:
    """Jaccard 유사도 평균"""
    scores = []
    for t, p in zip(y_true_sets, y_pred_sets):
        union = t | p
        scores.append(len(t & p) / len(union) if union else 1.0)
    return round(sum(scores) / len(scores), 4) if scores else float("nan")


# ──────────────────────────────────────────────
# 요약 출력 헬퍼
# ──────────────────────────────────────────────

def score_distribution(scores: list[float]) -> dict:
    """점수 분포 통계 (min, max, mean, std, 구간 비율)"""
    if not scores:
        return {}
    n = len(scores)
    mean = sum(scores) / n
    std  = math.sqrt(sum((x - mean) ** 2 for x in scores) / n)
    return {
        "n":    n,
        "min":  round(min(scores), 4),
        "max":  round(max(scores), 4),
        "mean": round(mean, 4),
        "std":  round(std,  4),
        "low_pct":  round(sum(s < 0.35 for s in scores) / n * 100, 1),
        "mid_pct":  round(sum(0.35 <= s < 0.65 for s in scores) / n * 100, 1),
        "high_pct": round(sum(s >= 0.65 for s in scores) / n * 100, 1),
    }
