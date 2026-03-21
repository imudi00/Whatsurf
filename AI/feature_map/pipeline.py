#!/usr/bin/env python3
# run_pipeline.py
"""
키워드 기반 뉴스 분석 파이프라인 진입점

실행 예시:
    python run_pipeline.py --keyword 탄핵
    python run_pipeline.py --keyword 탄핵 --limit 10 --batch_size 3 --out_dir ./results

처리 흐름:
    1. Supabase ai_test 테이블에서 keyword 기준으로 id/title/body/comments 로드
    2. 뉴스 원문 (title + body):
       - 규칙 기반 피처: body_depth, omission_risk, loaded_words(최대 3개), bias_vector
       - LLM 배치 (batch_size개 묶음): frame / logic / stance
       - 저장: {out_dir}/{keyword}_{id}.json   (_emotion_probs 제외)
    3. 댓글 (comments):
       - LLM 배치 (기사당 전체 댓글 1회): emotion_label / emotion_intensity / loaded_words
       - 저장: {out_dir}/{id}_comments.json

API 절약 전략:
    - 뉴스: batch_size개씩 묶어 LLM 1회 호출
    - 댓글: 기사당 전체 댓글 LLM 1회 호출
    - RPM/TPM/RPD 예외처리 및 모델 폴백은 llm_client.py에서 처리
"""
import argparse
import json
import os
import sys
from pathlib import Path

# ── sys.path 설정 ──────────────────────────────────────────
_HERE = Path(__file__).resolve().parent  # AI/feature_map/
_AI   = _HERE.parent                     # AI/

sys.path.insert(0, str(_AI))  # source.*, feature_map.*, llm.*

# ── 모듈 임포트 ───────────────────────────────────────────
from source.config.supabase_client import supabase

from feature_map.context.src.feature_map.keyword_extractor import extract_features
from feature_map.context.src.feature_map.body_depth        import compute_body_depth, describe_body_depth
from feature_map.context.src.feature_map.omission_risk     import compute_omission_risk
from feature_map.emotion.src.feature_map.loaded_words      import detect_loaded_words, loaded_word_density
from feature_map.emotion.src.feature_map.bias_vector       import compute_bias_vector, normalize_bias_vector
from feature_map.stance.src.feature_map.preprocessor       import build_article_struct

from llm.llm_batch import analyze_news_batch, analyze_comments_batch


# ──────────────────────────────────────────────────────────
# Supabase 데이터 로드
# ──────────────────────────────────────────────────────────

def load_by_keyword(keyword: str, limit: int) -> list[dict]:
    """ai_test 테이블에서 keyword 컬럼 일치 행을 페이지네이션으로 전부 로드"""
    PAGE_SIZE = 1000
    rows: list[dict] = []
    offset = 0
    while True:
        fetch = PAGE_SIZE if limit == 0 else min(PAGE_SIZE, limit - len(rows))
        resp = (
            supabase.table("ai_test")
            .select("id, title, body, comments")
            .eq("keyword", keyword)
            .range(offset, offset + fetch - 1)
            .execute()
        )
        page = resp.data or []
        rows.extend(page)
        if len(page) < fetch or (limit > 0 and len(rows) >= limit):
            break
        offset += fetch

    # comments 필드: JSON 문자열이면 파싱, 아니면 그대로
    for row in rows:
        c = row.get("comments") or []
        if isinstance(c, str):
            try:
                c = json.loads(c)
            except Exception:
                c = []
        row["comments"] = c

    return rows


# ──────────────────────────────────────────────────────────
# 규칙 기반 피처 추출 (LLM 미사용)
# ──────────────────────────────────────────────────────────

def extract_rule_features(text: str, cluster_texts: list[str]) -> dict:
    features  = extract_features(text)
    depth     = compute_body_depth(text, features)
    omission  = compute_omission_risk(text, cluster_texts, extract_features) if cluster_texts else "low"
    bias      = normalize_bias_vector(compute_bias_vector(text))
    loaded    = detect_loaded_words(text)[:3]   # 최대 3개

    return {
        "body_depth":           round(depth, 4),
        "body_depth_level":     describe_body_depth(depth),
        "omission_risk":        omission,
        "loaded_words":         loaded,
        "loaded_word_density":  round(loaded_word_density(text), 4),
        "bias_lr_score":        round(bias.lr_score, 4),
        "bias_direction":       bias.rationale,
        "_top_entities":        [e["word"] for e in features["entities"][:5]],
    }


# ──────────────────────────────────────────────────────────
# 뉴스 배치 처리
# ──────────────────────────────────────────────────────────

def process_news_batch(rows: list[dict], keyword: str, out_dir: str):
    """
    rows (batch_size개 단위) 를 받아:
    - 규칙 기반 피처 추출
    - LLM 배치 호출 (1회)
    - 각 기사를 {keyword}_{id}.json 으로 저장
    """
    texts = [f"{r['title']}\n\n{r['body']}" for r in rows]

    # omission_risk: 배치 내 상호 비교
    cluster_map = [
        [texts[j] for j in range(len(texts)) if j != i]
        for i in range(len(texts))
    ]

    print(f"    규칙 기반 피처 추출 중 ({len(rows)}개)...")
    rule_results = [
        extract_rule_features(texts[i], cluster_map[i])
        for i in range(len(texts))
    ]

    print(f"    LLM 배치 분석 중 ({len(rows)}개)...")
    structs = [build_article_struct(t) for t in texts]
    llm_results = analyze_news_batch(structs)

    for i, row in enumerate(rows):
        result = {
            "article_id":    str(row["id"]),
            "keyword":       keyword,
            # 논조 피처 (LLM)
            "frame":         llm_results[i]["frame"],
            "frame_reason":  llm_results[i].get("frame_reason"),
            "logic":         llm_results[i]["logic"],
            "logic_reason":  llm_results[i].get("logic_reason"),
            "stance_score":  llm_results[i]["stance_score"],
            "dominant_tone": llm_results[i].get("dominant_tone"),
            "key_evidence":  llm_results[i].get("key_evidence"),
            # 맥락 + 감정(규칙) 피처 (_emotion_probs 제외)
            **rule_results[i],
        }
        _save(result, os.path.join(out_dir, f"{keyword}_{row['id']}.json"))


# ──────────────────────────────────────────────────────────
# 댓글 배치 처리
# ──────────────────────────────────────────────────────────

def process_comments(row: dict, out_dir: str):
    """
    단일 기사의 전체 댓글을 LLM 1회로 분석 후
    {id}_comments.json 으로 저장
    """
    raw_comments = row.get("comments") or []
    if not raw_comments:
        print(f"    id={row['id']} 댓글 없음, 스킵")
        return

    # 댓글이 dict 리스트인 경우 텍스트 추출 (원본 메타 보존)
    texts, metas = [], []
    for c in raw_comments:
        if isinstance(c, dict):
            text = c.get("content") or c.get("text") or c.get("body") or str(c)
        else:
            text = str(c)
            c = {}
        texts.append(text)
        metas.append(c)

    print(f"    id={row['id']} 댓글 {len(texts)}개 LLM 분석 중...")
    llm_results = analyze_comments_batch(texts)

    output = {
        "article_id": str(row["id"]),
        "comments_emotion": [
            {
                **metas[i],
                "text":              texts[i],
                "emotion_label":     llm_results[i]["emotion_label"],
                "emotion_intensity": llm_results[i]["emotion_intensity"],
                "loaded_words":      llm_results[i]["loaded_words"],
            }
            for i in range(len(texts))
        ],
    }
    _save(output, os.path.join(out_dir, f"{row['id']}_comments.json"))


# ──────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────

def _save(obj: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"    → 저장: {path}")


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────

def run(keyword: str, limit: int, batch_size: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  keyword={keyword!r}  limit={limit}  batch_size={batch_size}")
    print(f"  out_dir={out_dir}")
    print(f"{'='*55}\n")

    # 1. Supabase 로드
    print("[1] Supabase ai_test 로드 중...")
    rows = load_by_keyword(keyword, limit)
    print(f"    → {len(rows)}개 로드 완료\n")
    if not rows:
        print("  데이터 없음. 종료.")
        return

    # 2. 뉴스 원문 분석 (batch_size 단위)
    print("[2] 뉴스 원문 분석...")
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        print(f"  배치 [{start+1}~{start+len(batch)}/{len(rows)}]")
        process_news_batch(batch, keyword, out_dir)
    print()

    # 3. 댓글 감정 분석 (기사별)
    print("[3] 댓글 감정 분석...")
    for row in rows:
        process_comments(row, out_dir)
    print()

    print(f"[완료] 결과 저장 위치: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="키워드 기반 뉴스 분석 파이프라인")
    parser.add_argument("--keyword",    required=True,        help="ai_test.keyword 컬럼 값")
    parser.add_argument("--limit",      type=int, default=10, help="가져올 기사 수 (기본: 10)")
    parser.add_argument("--batch_size", type=int, default=3,  help="뉴스 LLM 배치 크기 (기본: 3)")
    parser.add_argument("--out_dir",    default="./results",  help="결과 저장 폴더 (기본: ./results)")
    args = parser.parse_args()

    run(
        keyword=args.keyword,
        limit=args.limit,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
    )
