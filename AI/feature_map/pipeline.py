# pipeline.py
"""
feature_map 통합 파이프라인
감정 / 논조 / 맥락 피처를 순서대로 실행하고 결과를 병합한다.

실행 순서:
    1. [맥락] keyword_extractor  → NER/통계 피처
    2. [논조] preprocessor       → 구조 분해
    3. [맥락] body_depth         → 규칙 기반 깊이 점수
    4. [맥락] omission_risk      → 클러스터 대비 누락 위험도
    5. [논조] frame              → LLM 프레임 분류
    6. [논조] logic              → LLM 논거 유형 분류
    7. [논조] stance             → LLM 논조 점수 (frame/logic 맥락 활용)
"""
import os
import json
from datetime import datetime
from typing import List, Optional

# ── 맥락 피처 ──────────────────────────────────
from context.src.feature_map.keyword_extractor import extract_features
from context.src.feature_map.body_depth import compute_body_depth, describe_body_depth
from context.src.feature_map.omission_risk import compute_omission_risk

# ── 논조 피처 ──────────────────────────────────
from stance.src.feature_map.preprocessor import build_article_struct
from stance.src.feature_map.frame import extract_frame
from stance.src.feature_map.logic import extract_logic
from stance.src.feature_map.stance import extract_stance


# ──────────────────────────────────────────────
# 메인 분석 함수
# ──────────────────────────────────────────────

def analyze_article(
    text: str,
    cluster_articles: Optional[List[str]] = None,
) -> dict:
    """
    단일 기사를 분석해 모든 피처를 반환

    Args:
        text: 기사 전문
        cluster_articles: 동종 클러스터 기사 목록 (omission_risk 계산용)

    Returns:
        dict: 최종 피처 + 디버깅용 설명 필드(_접두어)
    """
    # Step 1: 맥락 — NER/통계 피처 추출
    features = extract_features(text)

    # Step 2: 논조 — 구조 분해
    struct = build_article_struct(text)

    # Step 3: 맥락 — 깊이 점수 (규칙 기반, 빠름)
    depth = compute_body_depth(text, features)

    # Step 4: 맥락 — 누락 위험도
    omission = (
        compute_omission_risk(text, cluster_articles, extract_features)
        if cluster_articles
        else "low"
    )

    # Step 5-7: 논조 — LLM 순차 분석 (앞 결과가 뒤에 맥락으로 전달)
    frame_result  = extract_frame(struct)
    logic_result  = extract_logic(struct)
    stance_result = extract_stance(struct, frame_result["frame"], logic_result["logic"])

    return {
        # ── 최종 피처 ─────────────────────────
        "frame":         frame_result["frame"],
        "logic":         logic_result["logic"],
        "stance_score":  stance_result["stance_score"],
        "body_depth":    depth,
        "body_depth_level": describe_body_depth(depth),
        "omission_risk": omission,
        # ── 디버깅/검증용 ─────────────────────
        "_frame_reason":    frame_result.get("reason"),
        "_logic_reason":    logic_result.get("reason"),
        "_dominant_tone":   stance_result.get("dominant_tone"),
        "_key_evidence":    stance_result.get("key_evidence"),
        "_top_entities":    [e['word'] for e in features["entities"][:5]],
    }


# ──────────────────────────────────────────────
# 배치 분석 함수
# ──────────────────────────────────────────────

def analyze_articles(texts: List[str]) -> List[dict]:
    """
    기사 목록을 일괄 분석
    omission_risk는 목록 내 상호 비교로 계산
    """
    results = []
    for i, text in enumerate(texts):
        cluster = [t for j, t in enumerate(texts) if j != i]
        results.append(analyze_article(text, cluster_articles=cluster))
    return results


# ──────────────────────────────────────────────
# 결과 저장 함수
# ──────────────────────────────────────────────

def save_result(result: dict, save_dir: str = "file/save/") -> str:
    """분석 결과를 타임스탬프 파일명으로 저장"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = os.path.join(save_dir, f"data_{timestamp}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return file_path


# ──────────────────────────────────────────────
# 샘플 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    sample = """"대구·경북 통합해달라"‥그러면 충남·대전은? 외통수 몰린 국민의힘
어제 무제한 토론을 돌연 중단한 국민의힘은 더불어민주당에 대구·경북 행정통합을 요구하고 있는데요.
하지만 민주당은 대구·경북 통합을 위해서는 충남·대전 통합법 처리에 협조하라며 국민의힘을 압박하고 있습니다.
대구·경북 통합을 요구하며 무제한 토론을 끝낸 국민의힘.
민주당을 향해 빨리 법사위를 열어 법안을 처리해 달라고 요구했습니다.
[송언석/국민의힘 원내대표] "오늘이라도 법사위와 원포인트 본회의를 열어서 대구·경북 특별법을 처리할 것을 촉구합니다."
여당 주도로 전남·광주 통합법만 처리되면서 지역 민심이 이탈할 조짐이 일자, 사실상 백기를 든 셈입니다."""

    result = analyze_article(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    saved_path = save_result(result)
    print(f"\n[저장 완료] {saved_path}")
