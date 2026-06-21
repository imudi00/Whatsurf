#!/usr/bin/env python3
# labeling/experiments/rule_based/loaded_words_tuner.py
"""
loaded_words 파라미터 / 사전(dictionary) 튜닝 실험

모드 1 — 분포 분석 (정답 없음):
    python labeling/experiments/rule_based/loaded_words_tuner.py \
        --labeled_dir labeling/output \
        --mode distribution

모드 2 — 사전 평가 (정답 필요):
    python labeling/experiments/rule_based/loaded_words_tuner.py \
        --ground_truth labeling/experiments/data/ground_truth.jsonl \
        --mode eval

모드 3 — 단어 중요도 분석 (어떤 단어가 가장 자주 / 유용하게 감지되는지):
    python labeling/experiments/rule_based/loaded_words_tuner.py \
        --labeled_dir labeling/output \
        --mode word_freq

튜닝 가능한 파라미터:
    max_words       : 최대 추출 단어 수 (기본 10)
    min_guaranteed  : 최소 보장 단어 수 (기본 2)
    tier_mode       : "tier1_only" | "tier1_2" | "all" (어느 tier까지 사용할지)
    custom_words    : 추가 편향어 목록
    exclude_words   : 제거할 단어 목록 (노이즈 단어)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── sys.path 설정 ─────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_AI   = _HERE.parents[3]
if str(_AI) not in sys.path:
    sys.path.insert(0, str(_AI))

from labeling.features.loaded_words import (
    BIAS_WORDS, OPINION_WORDS,
    detect_loaded_words, detect_opinion_words, detect_informal_patterns,
)
from labeling.experiments.utils.metrics import (
    precision_at_k, recall_at_k, f1_at_k, jaccard,
    accuracy, score_distribution,
)
from labeling.experiments.utils.report import ExperimentResult, save_result, compare_results
from labeling.experiments.utils.data_loader import load_ground_truth, load_labeled_dir


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

@dataclass
class LoadedWordsConfig:
    name:           str
    max_words:      int   = 10
    min_guaranteed: int   = 2
    # "tier1_only"=BIAS_WORDS만, "tier1_2"=BIAS+OPINION, "all"=전체(tier3 포함)
    tier_mode:      str   = "all"
    # 추가 편향어 (BIAS_WORDS에 임시 추가)
    custom_words:   list[str] = field(default_factory=list)
    # 제거할 단어 (노이즈 단어 후보)
    exclude_words:  list[str] = field(default_factory=list)

    def effective_bias_words(self) -> list[str]:
        return [w for w in (BIAS_WORDS + self.custom_words) if w not in self.exclude_words]

    def effective_opinion_words(self) -> list[str]:
        return [w for w in OPINION_WORDS if w not in self.exclude_words]


PRESET_CONFIGS: list[LoadedWordsConfig] = [
    # 0. 기본값
    LoadedWordsConfig("baseline"),

    # 1. tier-1만 사용 (정치/이념 편향어만, 정밀도 높이기)
    LoadedWordsConfig("tier1_only", tier_mode="tier1_only"),

    # 2. tier-1+2 사용 (강한 의견 포함)
    LoadedWordsConfig("tier1_2", tier_mode="tier1_2"),

    # 3. 최대 단어 수 늘리기
    LoadedWordsConfig("max20", max_words=20),

    # 4. 최대 단어 수 줄이기 (핵심만)
    LoadedWordsConfig("max5", max_words=5),

    # 5. 최소 보장 해제 (진짜 없으면 빈 리스트)
    LoadedWordsConfig("no_guarantee", min_guaranteed=0),

    # 6. 노이즈 가능 단어 제거 후보 (흔히 오탐되는 단어들)
    LoadedWordsConfig("denoise_common",
                      exclude_words=["완전", "진심", "당연히", "솔직히", "절대", "반드시"]),

    # 7. 정치 강화 (추가 편향어 예시)
    LoadedWordsConfig("politics_enhanced",
                      custom_words=["내로남불", "이중잣대", "내편감싸기", "정치쇼", "꼼수"]),
]


# ──────────────────────────────────────────────
# Config 기반 loaded_words 추출
# ──────────────────────────────────────────────

def extract_with_config(text: str, cfg: LoadedWordsConfig) -> dict:
    """Config 설정으로 loaded_words 추출"""
    bias_words    = cfg.effective_bias_words()
    opinion_words = cfg.effective_opinion_words()

    # Tier-1
    tier1 = [w for w in bias_words if w in text]

    result: list[str] = list(dict.fromkeys(tier1))

    if cfg.tier_mode in ("tier1_2", "all"):
        # Tier-2
        tier2 = [w for w in opinion_words if w in text and w not in result]
        tier2 += [w for w in detect_informal_patterns(text) if w not in result and w not in tier2]
        result = list(dict.fromkeys(result + tier2))

    if cfg.tier_mode == "all":
        # Tier-3 fallback (최소 보장)
        if len(result) < cfg.min_guaranteed:
            import re
            tokens = re.split(r'[\s,!?.\'"·…\-~ㆍ]+', text)
            stopwords = {"이","가","은","는","을","를","에","의","와","과","도","만"}
            freq: dict[str, int] = {}
            for t in tokens:
                t = t.strip()
                if len(t) >= 2 and t not in stopwords and not t.isdigit():
                    freq[t] = freq.get(t, 0) + 1
            extra = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])
                     if w not in result]
            result += extra[: cfg.min_guaranteed - len(result)]

    total_words = max(len(text.split()), 1)
    density = sum(text.count(w) for w in tier1) / total_words

    return {
        "loaded_words":        result[:cfg.max_words],
        "loaded_word_density": round(density, 4),
        "is_biased":           len(tier1) > 0,
        "tier1_count":         len(tier1),
    }


# ──────────────────────────────────────────────
# 모드 1: 분포 분석
# ──────────────────────────────────────────────

def run_distribution(texts: list[str], ids: list[str], out_dir: str):
    print("\n" + "=" * 60)
    print("  loaded_words 분포 분석")
    print("=" * 60)

    for cfg in PRESET_CONFIGS:
        results = [extract_with_config(t, cfg) for t in texts]
        biased_n   = sum(r["is_biased"] for r in results)
        avg_words  = sum(len(r["loaded_words"]) for r in results) / max(len(results), 1)
        tier1_avg  = sum(r["tier1_count"] for r in results) / max(len(results), 1)
        densities  = [r["loaded_word_density"] for r in results]
        dist       = score_distribution(densities)

        print(f"\n[{cfg.name}]")
        print(f"  편향 기사 비율: {biased_n}/{len(results)} ({biased_n/max(len(results),1)*100:.1f}%)")
        print(f"  평균 단어 수  : {avg_words:.2f}개  (tier-1 평균: {tier1_avg:.2f}개)")
        print(f"  density 분포  : mean={dist['mean']:.4f}  std={dist['std']:.4f}")


# ──────────────────────────────────────────────
# 모드 2: 정답 기반 평가
# ──────────────────────────────────────────────

def run_eval(
    texts:    list[str],
    y_true_sets: list[set],
    ids:      list[str],
    out_dir:  str,
):
    """
    ground_truth의 loaded_words와 예측 결과 비교.
    정답이 없는 기사는 제외.
    """
    print(f"\n  loaded_words 평가 샘플: {len(texts)}개")

    results: list[ExperimentResult] = []

    for cfg in PRESET_CONFIGS:
        predictions = [extract_with_config(t, cfg) for t in texts]
        pred_lists  = [p["loaded_words"] for p in predictions]

        p_k = precision_at_k(y_true_sets, pred_lists)
        r_k = recall_at_k(y_true_sets, pred_lists)
        f1  = f1_at_k(y_true_sets, pred_lists)
        jac = jaccard(y_true_sets, [set(p) for p in pred_lists])

        # 편향 여부 분류 (is_biased)
        true_biased = [len(s) > 0 for s in y_true_sets]
        pred_biased = [p["is_biased"] for p in predictions]
        acc_biased  = accuracy(true_biased, pred_biased)

        r = ExperimentResult(
            name=cfg.name,
            feature="loaded_words",
            config={"max_words": cfg.max_words, "tier_mode": cfg.tier_mode,
                    "min_guaranteed": cfg.min_guaranteed,
                    "custom_n": len(cfg.custom_words), "exclude_n": len(cfg.exclude_words)},
            metrics={"precision": p_k, "recall": r_k, "f1": f1,
                     "jaccard": jac, "is_biased_acc": acc_biased},
            n_samples=len(texts),
        )
        results.append(r)
        save_result(r, out_dir)

    print(compare_results(results, sort_by="f1"))
    return results


# ──────────────────────────────────────────────
# 모드 3: 단어 빈도/유용성 분석
# ──────────────────────────────────────────────

def run_word_freq(texts: list[str], out_dir: str):
    """
    전체 텍스트에서 BIAS_WORDS / OPINION_WORDS 출현 빈도 분석.
    잘 감지되지 않는 단어(노이즈 후보)와 자주 감지되는 단어를 식별.
    """
    bias_freq:    Counter = Counter()
    opinion_freq: Counter = Counter()

    for text in texts:
        for w in detect_loaded_words(text):
            bias_freq[w] += 1
        for w in detect_opinion_words(text):
            opinion_freq[w] += 1

    n = max(len(texts), 1)

    print("\n" + "=" * 60)
    print("  BIAS_WORDS 출현 빈도 (상위 20)")
    print("=" * 60)
    for w, cnt in bias_freq.most_common(20):
        print(f"  {w:<20} {cnt:>5}건  ({cnt/n*100:.1f}%)")

    print("\n  OPINION_WORDS 출현 빈도 (상위 20)")
    print("=" * 60)
    for w, cnt in opinion_freq.most_common(20):
        print(f"  {w:<20} {cnt:>5}건  ({cnt/n*100:.1f}%)")

    # 한번도 안 감지된 단어 (사전에는 있지만 데이터에 없음)
    unseen_bias    = [w for w in BIAS_WORDS    if w not in bias_freq]
    unseen_opinion = [w for w in OPINION_WORDS if w not in opinion_freq]
    print(f"\n  감지 0건 편향어({len(unseen_bias)}개): {unseen_bias[:15]}...")
    print(f"  감지 0건 의견어({len(unseen_opinion)}개): {unseen_opinion[:15]}...")

    # 저장
    os.makedirs(out_dir, exist_ok=True)
    freq_data = {
        "bias_freq":    dict(bias_freq.most_common()),
        "opinion_freq": dict(opinion_freq.most_common()),
        "unseen_bias":  unseen_bias,
        "unseen_opinion": unseen_opinion,
        "n_articles": n,
    }
    out_path = os.path.join(out_dir, "loaded_words_freq.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(freq_data, f, ensure_ascii=False, indent=2)
    print(f"\n  → 빈도 데이터 저장: {out_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="loaded_words 튜닝 실험")
    parser.add_argument("--labeled_dir",  default=None)
    parser.add_argument("--ground_truth", default=None)
    parser.add_argument("--mode", choices=["distribution", "eval", "word_freq"], default="distribution")
    parser.add_argument("--out_dir", default="labeling/experiments/results")
    args = parser.parse_args()

    # 데이터 로드
    if args.mode in ("distribution", "word_freq"):
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

        if args.mode == "distribution":
            run_distribution(texts, ids, args.out_dir)
        else:
            run_word_freq(texts, args.out_dir)

    elif args.mode == "eval":
        if not args.ground_truth:
            print("[오류] --ground_truth 필요")
            return
        records = load_ground_truth(args.ground_truth)
        valid = [(r["text"], set(r["labels"].get("loaded_words") or []), str(r["id"]))
                 for r in records if r.get("labels", {}).get("loaded_words") is not None]
        if not valid:
            print("[오류] ground_truth에 loaded_words 레이블이 없습니다.")
            return
        texts, y_true_sets, ids = zip(*valid)
        run_eval(list(texts), list(y_true_sets), list(ids), args.out_dir)


if __name__ == "__main__":
    _cli()
