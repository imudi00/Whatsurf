# labeling/experiments/utils/data_loader.py
"""
실험용 데이터 로드 / 정답 템플릿 생성 유틸

Ground Truth JSONL 형식:
    {"id": "123", "text": "기사 제목\\n\\n본문...",
     "labels": {
         "body_depth":      0.75,       (float 0~1)
         "frame":           "갈등/대립 강조",
         "logic":           "사실/정보 전달",
         "bias_x":          0.3,        (float -1~1)
         "bias_y":          0.1,        (float -1~1)
         "omission_risk":   "low",      (low|mid|high)
         "stance_score":    -0.2,       (float -1~1)
         "loaded_words":    ["극우", "재앙"]
     }}

실행:
    # labeled.json 결과물에서 정답 템플릿 생성 (수동 검토용)
    python -m labeling.experiments.utils.data_loader \
        --labeled_dir labeling/output \
        --out labeling/experiments/data/ground_truth.jsonl \
        --sample 30
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────
# Ground Truth JSONL 로드
# ──────────────────────────────────────────────

def load_ground_truth(path: str) -> list[dict]:
    """
    JSONL 형식 정답 파일 로드.
    각 줄: {"id":..., "text":..., "labels": {...}}
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [경고] 줄 {lineno} 파싱 실패: {e}")
    return records


def split_ground_truth(
    records: list[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    train / test 분할.
    Returns: (train_records, test_records)
    """
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    return shuffled[n_test:], shuffled[:n_test]


# ──────────────────────────────────────────────
# labeled.json 디렉토리 로드
# ──────────────────────────────────────────────

def load_labeled_dir(directory: str) -> list[dict]:
    """
    labeling/output/ 의 *_labeled.json 파일들을 로드.
    run_labeling.py 출력 형식 → 통합 리스트 반환.
    """
    records = []
    for path in sorted(Path(directory).glob("*_labeled.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            records.append(data)
        except Exception as e:
            print(f"  [경고] {path.name} 로드 실패: {e}")
    return records


# ──────────────────────────────────────────────
# 정답 템플릿 생성 (수동 어노테이션 전처리)
# ──────────────────────────────────────────────

def make_annotation_template(
    labeled_records: list[dict],
    out_path: str,
    sample_n: int | None = None,
    seed: int = 42,
    features: list[str] | None = None,
) -> int:
    """
    labeled.json 결과물 → 어노테이션 템플릿 JSONL 생성.

    labels 필드에 모델 예측값을 채워 넣고,
    검토자가 틀린 값을 수정하면 ground_truth로 사용 가능.

    Args:
        labeled_records: load_labeled_dir() 결과
        out_path: 출력 JSONL 경로
        sample_n: 무작위 샘플링 개수 (None=전체)
        features: 포함할 피처 키 목록 (None=전체)

    Returns:
        저장된 레코드 수
    """
    _DEFAULT_FEATURES = [
        "body_depth", "frame", "logic",
        "bias_x", "bias_y", "omission_risk",
        "stance_score", "loaded_words", "art_words",
    ]
    target_features = features or _DEFAULT_FEATURES

    records = list(labeled_records)
    if sample_n and sample_n < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, sample_n)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            aid  = str(rec.get("article_id") or rec.get("id") or "")
            text = f"{rec.get('title','')}\n\n{rec.get('body','') or rec.get('text','')}"

            labels: dict[str, Any] = {}
            for feat in target_features:
                val = rec.get(feat)
                if feat == "loaded_words":
                    val = rec.get("art_words") or rec.get("loaded_words") or []
                if val is not None:
                    labels[feat] = val

            entry = {"id": aid, "text": text.strip(), "labels": labels}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"  → {written}개 템플릿 저장: {out_path}")
    print("  ※ labels 필드를 직접 수정하여 ground truth로 사용하세요.")
    return written


# ──────────────────────────────────────────────
# 피처별 정답/예측 쌍 추출
# ──────────────────────────────────────────────

def extract_pairs(
    ground_truth: list[dict],
    predictions:  list[dict],
    feature: str,
    pred_feature: str | None = None,
) -> tuple[list, list, list[str]]:
    """
    ground_truth와 predictions를 id 기준으로 정렬하여
    (true_values, pred_values, matched_ids) 반환.

    Args:
        ground_truth : load_ground_truth() 결과
        predictions  : load_labeled_dir() 결과
        feature      : ground_truth["labels"][feature]에서 읽을 키
        pred_feature : predictions[pred_feature]에서 읽을 키 (None이면 feature 그대로)

    Returns:
        (y_true, y_pred, ids) — None 값은 제외
    """
    pred_feature = pred_feature or feature
    pred_by_id: dict[str, Any] = {}
    for p in predictions:
        pid = str(p.get("article_id") or p.get("id") or "")
        pred_by_id[pid] = p

    y_true, y_pred, ids = [], [], []
    for gt in ground_truth:
        gid = str(gt.get("id") or "")
        true_val = gt.get("labels", {}).get(feature)
        pred_val = pred_by_id.get(gid, {}).get(pred_feature)

        if true_val is None or pred_val is None:
            continue

        y_true.append(true_val)
        y_pred.append(pred_val)
        ids.append(gid)

    return y_true, y_pred, ids


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="labeled.json 결과 → 어노테이션 템플릿 JSONL 생성"
    )
    parser.add_argument("--labeled_dir", required=True, help="*_labeled.json 디렉토리")
    parser.add_argument("--out", required=True, help="출력 JSONL 경로")
    parser.add_argument("--sample", type=int, default=None, help="샘플링 개수")
    parser.add_argument("--features", nargs="*", default=None, help="포함할 피처 (기본: 전체)")
    args = parser.parse_args()

    records = load_labeled_dir(args.labeled_dir)
    print(f"  로드된 기사: {len(records)}개")
    make_annotation_template(records, args.out, args.sample, features=args.features)


if __name__ == "__main__":
    _cli()
