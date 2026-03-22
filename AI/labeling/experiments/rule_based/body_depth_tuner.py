#!/usr/bin/env python3
# labeling/experiments/rule_based/body_depth_tuner.py
"""
body_depth 파라미터 튜닝 실험

모드 1 — 정답 없이 분포 분석 (정답 불필요):
    python labeling/experiments/rule_based/body_depth_tuner.py \
        --labeled_dir labeling/output \
        --mode distribution

모드 2 — Grid Search (정답 필요):
    python labeling/experiments/rule_based/body_depth_tuner.py \
        --ground_truth labeling/experiments/data/ground_truth.jsonl \
        --mode grid \
        --metric pearson

모드 3 — 단일 config 적용 후 결과 저장:
    python labeling/experiments/rule_based/body_depth_tuner.py \
        --ground_truth labeling/experiments/data/ground_truth.jsonl \
        --mode single --config_name diversity_heavy

튜닝 가능한 파라미터:
    WEIGHTS: length / diversity / quotes / numerics / structure (합=1.0)
    LENGTH_SOFT_MAX   : 본문 길이 포화점 (char)
    DIVERSITY_TTR_REF : 어절 TTR 기준값
    QUOTE_RATIO_REF   : 인용 비율 기준값
    NUMERIC_DENSITY_REF: 수치 패턴 밀도 기준값
    STRUCTURE_PARA_MAX : 문단 수 포화점
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

# ── sys.path 설정 ─────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_AI   = _HERE.parents[3]   # AI/
if str(_AI) not in sys.path:
    sys.path.insert(0, str(_AI))

from labeling.experiments.utils.metrics import (
    pearson_r, spearman_rho, mae, score_distribution,
)
from labeling.experiments.utils.report import ExperimentResult, save_result, compare_results
from labeling.experiments.utils.data_loader import load_ground_truth, load_labeled_dir


# ──────────────────────────────────────────────
# Config 데이터클래스
# ──────────────────────────────────────────────

@dataclass
class BodyDepthConfig:
    name: str
    # 가중치 (합이 1.0이어야 함)
    w_length:    float = 0.20
    w_diversity: float = 0.25
    w_quotes:    float = 0.20
    w_numerics:  float = 0.20
    w_structure: float = 0.15
    # 정규화 상수
    length_soft_max:      int   = 2000
    diversity_ttr_ref:    float = 0.55
    quote_ratio_ref:      float = 0.12
    numeric_density_ref:  float = 0.06
    structure_para_max:   int   = 8

    def as_param_dict(self) -> dict:
        d = asdict(self)
        d.pop("name")
        return d

    def weight_sum(self) -> float:
        return self.w_length + self.w_diversity + self.w_quotes + self.w_numerics + self.w_structure


# ──────────────────────────────────────────────
# 사전 정의 Config 목록
# ──────────────────────────────────────────────

PRESET_CONFIGS: list[BodyDepthConfig] = [
    # 0. 현재 기본값 (baseline)
    BodyDepthConfig("baseline"),

    # 1. 다양성·인용 중심
    BodyDepthConfig("diversity_quote_heavy",
                    w_length=0.10, w_diversity=0.35, w_quotes=0.30,
                    w_numerics=0.15, w_structure=0.10),

    # 2. 길이·수치 중심 (보도자료형 기사에 유리)
    BodyDepthConfig("length_numeric_heavy",
                    w_length=0.35, w_diversity=0.15, w_quotes=0.15,
                    w_numerics=0.30, w_structure=0.05),

    # 3. 구조 중심 (기획 기사 감지)
    BodyDepthConfig("structure_heavy",
                    w_length=0.15, w_diversity=0.20, w_quotes=0.20,
                    w_numerics=0.15, w_structure=0.30),

    # 4. 균등 가중치
    BodyDepthConfig("uniform",
                    w_length=0.20, w_diversity=0.20, w_quotes=0.20,
                    w_numerics=0.20, w_structure=0.20),

    # 5. 길이 포화점 낮춤 (짧은 기사 차별화)
    BodyDepthConfig("short_article_focus",
                    length_soft_max=1000, structure_para_max=5),

    # 6. 길이 포화점 높임 (긴 기사 차별화)
    BodyDepthConfig("long_article_focus",
                    length_soft_max=4000, structure_para_max=15),

    # 7. 인용 기준 낮춤 (인용 많은 기사에 후한 점수)
    BodyDepthConfig("quote_sensitive",
                    quote_ratio_ref=0.06, w_quotes=0.30,
                    w_length=0.15, w_diversity=0.20, w_numerics=0.20, w_structure=0.15),

    # 8. 수치 기준 낮춤 (수치 많은 기사에 후한 점수)
    BodyDepthConfig("numeric_sensitive",
                    numeric_density_ref=0.03, w_numerics=0.30,
                    w_length=0.15, w_diversity=0.20, w_quotes=0.20, w_structure=0.15),
]


# ──────────────────────────────────────────────
# body_depth 계산 (Config 기반)
# ──────────────────────────────────────────────

def compute_with_config(text: str, cfg: BodyDepthConfig) -> float:
    """주어진 Config로 body_depth 계산"""
    import re
    # 정규식 패턴 (local/body_depth.py와 동일)
    _QUOTE_RE = re.compile(
        r'["\u201c\u201d\u300c\u300d\u300e\u300f\u2018\u2019]'
        r'[^"\u201c\u201d\u300c\u300d\u300e\u300f\u2018\u2019]{3,}'
        r'["\u201c\u201d\u300c\u300d\u300e\u300f\u2018\u2019]'
    )
    _NUMERIC_RE = re.compile(
        r'\d[\d,]*(?:\.\d+)?'
        r'(?:\s*(?:%|억|만|천|원|개|명|건|회|배|㎞|km|m|㎡|℃|년|월|일|분|초|위))?'
    )

    if not text or not text.strip():
        return 0.0
    words = text.split()
    n_words = max(len(words), 1)

    # length
    n = len(text)
    l_score = min(math.log(n + 1) / math.log(cfg.length_soft_max + 1), 1.0)

    # diversity (TTR)
    ttr = len(set(words)) / n_words
    d_score = min(ttr / cfg.diversity_ttr_ref, 1.0)

    # quotes
    quoted_chars = sum(len(m.group()) for m in _QUOTE_RE.finditer(text))
    ratio = quoted_chars / max(len(text), 1)
    q_score = min(ratio / cfg.quote_ratio_ref, 1.0)

    # numerics
    n_nums = len(_NUMERIC_RE.findall(text))
    n_score = min((n_nums / n_words) / cfg.numeric_density_ref, 1.0)

    # structure
    lines = [ln for ln in text.split("\n") if ln.strip()]
    s_score = min(len(lines) / cfg.structure_para_max, 1.0)

    score = (
        cfg.w_length    * l_score
        + cfg.w_diversity * d_score
        + cfg.w_quotes    * q_score
        + cfg.w_numerics  * n_score
        + cfg.w_structure * s_score
    )
    return round(score, 4)


# ──────────────────────────────────────────────
# 모드 1: 분포 분석 (정답 불필요)
# ──────────────────────────────────────────────

def run_distribution(texts: list[str], ids: list[str], out_dir: str):
    """
    사전 정의 모든 config로 body_depth 계산 후
    점수 분포 및 순위 변화를 분석.
    정답 없이 '어느 config가 더 고르게 분포되는지' 확인용.
    """
    print("\n" + "=" * 60)
    print("  body_depth 분포 분석 (정답 없음)")
    print("=" * 60)

    all_scores: dict[str, list[float]] = {}
    for cfg in PRESET_CONFIGS:
        scores = [compute_with_config(t, cfg) for t in texts]
        all_scores[cfg.name] = scores
        dist = score_distribution(scores)
        print(f"\n[{cfg.name}]")
        print(f"  mean={dist['mean']:.3f}  std={dist['std']:.3f}  "
              f"low={dist['low_pct']}%  mid={dist['mid_pct']}%  high={dist['high_pct']}%")

    # 순위 상관: 각 config vs baseline 비교
    baseline = all_scores["baseline"]
    print("\n\n[baseline 대비 스피어만 순위 상관]")
    for name, scores in all_scores.items():
        if name == "baseline":
            continue
        rho = spearman_rho(baseline, scores)
        print(f"  {name:<30} ρ = {rho:.4f}")

    # CSV 저장
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "body_depth_distribution.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = "id," + ",".join(cfg.name for cfg in PRESET_CONFIGS)
        f.write(header + "\n")
        for i, aid in enumerate(ids):
            row = aid + "," + ",".join(str(all_scores[cfg.name][i]) for cfg in PRESET_CONFIGS)
            f.write(row + "\n")
    print(f"\n  → CSV 저장: {csv_path}")


# ──────────────────────────────────────────────
# 모드 2: Grid Search (정답 필요)
# ──────────────────────────────────────────────

# 탐색 공간 정의
GRID_SPACE = {
    "w_length":    [0.10, 0.20, 0.30],
    "w_diversity": [0.15, 0.25, 0.35],
    "w_quotes":    [0.15, 0.20, 0.30],
    "w_numerics":  [0.10, 0.20, 0.30],
    # w_structure = 1 - (w_length + w_diversity + w_quotes + w_numerics)
    "length_soft_max":     [1000, 2000, 3000],
    "diversity_ttr_ref":   [0.45, 0.55, 0.65],
    "quote_ratio_ref":     [0.06, 0.12, 0.18],
    "numeric_density_ref": [0.03, 0.06, 0.09],
    "structure_para_max":  [5, 8, 12],
}


def _random_configs(n: int = 100, seed: int = 42) -> list[BodyDepthConfig]:
    """랜덤 서치용 config 생성"""
    import random
    rng = random.Random(seed)
    configs = []
    for i in range(n):
        wl = rng.choice(GRID_SPACE["w_length"])
        wd = rng.choice(GRID_SPACE["w_diversity"])
        wq = rng.choice(GRID_SPACE["w_quotes"])
        wn = rng.choice(GRID_SPACE["w_numerics"])
        ws = max(round(1.0 - wl - wd - wq - wn, 2), 0.05)

        configs.append(BodyDepthConfig(
            name=f"random_{i:03d}",
            w_length=wl, w_diversity=wd, w_quotes=wq, w_numerics=wn, w_structure=ws,
            length_soft_max=rng.choice(GRID_SPACE["length_soft_max"]),
            diversity_ttr_ref=rng.choice(GRID_SPACE["diversity_ttr_ref"]),
            quote_ratio_ref=rng.choice(GRID_SPACE["quote_ratio_ref"]),
            numeric_density_ref=rng.choice(GRID_SPACE["numeric_density_ref"]),
            structure_para_max=rng.choice(GRID_SPACE["structure_para_max"]),
        ))
    return configs


def run_grid(
    texts:        list[str],
    y_true:       list[float],
    ids:          list[str],
    metric_name:  str = "pearson",
    n_random:     int = 100,
    out_dir:      str = "labeling/experiments/results",
):
    """
    랜덤 서치 + 사전 정의 preset으로 최적 Config 탐색.

    Args:
        metric_name: "pearson" | "spearman" | "mae"
    """
    metric_fn: Callable = {
        "pearson":  pearson_r,
        "spearman": spearman_rho,
        "mae":      lambda t, p: -mae(t, p),   # mae는 낮을수록 좋으므로 부호 반전
    }[metric_name]

    all_configs = PRESET_CONFIGS + _random_configs(n_random)
    results = []

    print(f"\n  총 {len(all_configs)}개 config 평가 중 (metric={metric_name}) ...")
    for cfg in all_configs:
        y_pred = [compute_with_config(t, cfg) for t in texts]
        score  = metric_fn(y_true, y_pred)

        results.append(ExperimentResult(
            name=cfg.name,
            feature="body_depth",
            config=cfg.as_param_dict(),
            metrics={
                metric_name: score,
                "pearson":   pearson_r(y_true, y_pred),
                "spearman":  spearman_rho(y_true, y_pred),
                "mae":       mae(y_true, y_pred),
            },
            n_samples=len(texts),
            predictions=[{"id": i, "pred": p, "true": t}
                         for i, p, t in zip(ids, y_pred, y_true)],
        ))

    results.sort(key=lambda r: r.metrics.get(metric_name, float("-inf")), reverse=True)

    print("\n" + "=" * 70)
    print(f"  Top 5 결과 (기준: {metric_name})")
    print("=" * 70)
    for r in results[:5]:
        print(f"  {r.summary()}")
        print(f"    config: {r.config}")

    print(compare_results(results[:10], sort_by=metric_name))

    # 전체 저장
    os.makedirs(out_dir, exist_ok=True)
    for r in results[:5]:
        save_result(r, out_dir)

    return results[0]


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="body_depth 파라미터 튜닝")
    parser.add_argument("--labeled_dir",   default=None, help="*_labeled.json 디렉토리 (mode=distribution)")
    parser.add_argument("--ground_truth",  default=None, help="ground_truth.jsonl 경로 (mode=grid)")
    parser.add_argument("--mode",          choices=["distribution", "grid", "preset"], default="distribution")
    parser.add_argument("--metric",        default="pearson", choices=["pearson","spearman","mae"])
    parser.add_argument("--n_random",      type=int, default=100, help="랜덤 서치 횟수")
    parser.add_argument("--out_dir",       default="labeling/experiments/results")
    parser.add_argument("--config_name",   default="baseline", help="mode=preset 시 사용할 config 이름")
    args = parser.parse_args()

    if args.mode == "distribution":
        src = args.labeled_dir or args.ground_truth
        if not src:
            print("[오류] --labeled_dir 또는 --ground_truth 필요")
            return
        if src.endswith(".jsonl"):
            records = load_ground_truth(src)
            texts = [r.get("text","") for r in records]
            ids   = [str(r.get("id","")) for r in records]
        else:
            records = load_labeled_dir(src)
            texts = [f"{r.get('title','')}\n\n{r.get('body','')}" for r in records]
            ids   = [str(r.get("article_id","")) for r in records]
        run_distribution(texts, ids, args.out_dir)

    elif args.mode == "grid":
        if not args.ground_truth:
            print("[오류] --ground_truth 필요")
            return
        records = load_ground_truth(args.ground_truth)
        texts  = [r.get("text","") for r in records]
        y_true = [r["labels"].get("body_depth") for r in records]
        ids    = [str(r.get("id","")) for r in records]

        # body_depth 정답 없는 레코드 제거
        valid = [(t, y, i) for t, y, i in zip(texts, y_true, ids) if y is not None]
        if not valid:
            print("[오류] ground_truth에 body_depth 레이블이 없습니다.")
            return
        texts, y_true, ids = zip(*valid)
        print(f"  body_depth 정답 있는 샘플: {len(texts)}개")
        run_grid(list(texts), list(y_true), list(ids),
                 metric_name=args.metric, n_random=args.n_random, out_dir=args.out_dir)

    elif args.mode == "preset":
        cfg = next((c for c in PRESET_CONFIGS if c.name == args.config_name), None)
        if not cfg:
            names = [c.name for c in PRESET_CONFIGS]
            print(f"[오류] config 이름 불명: {args.config_name}\n  가능한 이름: {names}")
            return
        print(f"\n[preset] {cfg.name}")
        for k, v in cfg.as_param_dict().items():
            print(f"  {k} = {v}")


if __name__ == "__main__":
    _cli()
