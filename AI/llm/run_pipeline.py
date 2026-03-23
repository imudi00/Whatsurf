#!/usr/bin/env python3
# llm/run_pipeline.py
"""
키워드 기반 뉴스 분석 파이프라인

실행:
    python run_pipeline.py --keyword 종소세
    python run_pipeline.py --keyword 탄핵 --limit 10 --batch_size 3 --out_dir ./results

저장 파일:
    {out_dir}/{keyword}_{id}.json      ← 뉴스 피처 (_emotion_probs 제외)
    {out_dir}/{id}_comments.json       ← 댓글별 emotion / loaded_words
"""
import argparse, json, os, sys
from pathlib import Path

# ── sys.path 설정 ──────────────────────────────────────────
_HERE = Path(__file__).resolve().parent   # AI/llm/
_AI   = _HERE.parent                      # AI/

sys.path.insert(0, str(_AI))    # source.*, feature_map.*
sys.path.insert(0, str(_HERE))  # llm_batch, llm_client

# ── 임포트 ────────────────────────────────────────────────
from source.config.supabase_client import supabase

from feature_map.context.src.feature_map.keyword_extractor import extract_features
from feature_map.context.src.feature_map.body_depth        import compute_body_depth, describe_body_depth
from feature_map.context.src.feature_map.omission_risk     import compute_omission_risk
from feature_map.emotion.src.feature_map.loaded_words      import detect_loaded_words, loaded_word_density
from feature_map.emotion.src.feature_map.bias_vector       import compute_bias_vector, normalize_bias_vector
from feature_map.stance.src.feature_map.preprocessor import build_article_struct

from llm_batch import analyze_news_batch, analyze_comments_batch


# ── Supabase 로드 ──────────────────────────────────────────

def load_by_keyword(keyword: str, limit: int) -> list:
    PAGE_SIZE = 1000
    rows: list = []
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

    for row in rows:
        c = row.get("comments") or []
        if isinstance(c, str):
            try: c = json.loads(c)
            except: c = []
        row["comments"] = c
    return rows


# ── 규칙 기반 피처 ─────────────────────────────────────────

def extract_rule_features(text: str, cluster_texts: list) -> dict:
    feat     = extract_features(text)
    depth    = compute_body_depth(text, feat)
    omission = compute_omission_risk(text, cluster_texts, extract_features) if cluster_texts else "low"
    bias     = normalize_bias_vector(compute_bias_vector(text))
    loaded   = detect_loaded_words(text)[:3]
    return {
        "body_depth":          round(depth, 4),
        "body_depth_level":    describe_body_depth(depth),
        "omission_risk":       omission,
        "loaded_words":        loaded,
        "loaded_word_density": round(loaded_word_density(text), 4),
        "bias_lr_score":       round(bias.lr_score, 4),
        "bias_direction":      bias.rationale,
        "_top_entities":       [e["word"] for e in feat["entities"][:5]],
    }


# ── 뉴스 배치 처리 ─────────────────────────────────────────

def process_news_batch(rows: list, keyword: str, out_dir: str):
    texts = [f"{r['title']}\n\n{r['body']}" for r in rows]
    clusters = [[texts[j] for j in range(len(texts)) if j != i] for i in range(len(texts))]

    print(f"    규칙 기반 피처 추출 ({len(rows)}개)...")
    rule = [extract_rule_features(texts[i], clusters[i]) for i in range(len(texts))]

    print(f"    LLM 배치 분석 ({len(rows)}개)...")
    structs = [build_article_struct(t) for t in texts]
    llm = analyze_news_batch(structs)

    for i, row in enumerate(rows):
        result = {
            "article_id":    str(row["id"]),
            "keyword":       keyword,
            "frame":         llm[i]["frame"],
            "frame_reason":  llm[i].get("frame_reason"),
            "logic":         llm[i]["logic"],
            "logic_reason":  llm[i].get("logic_reason"),
            "stance_score":  llm[i]["stance_score"],
            "dominant_tone": llm[i].get("dominant_tone"),
            "key_evidence":  llm[i].get("key_evidence"),
            **rule[i],
        }
        _save(result, os.path.join(out_dir, f"{keyword}_{row['id']}.json"))


# ── 댓글 배치 처리 ─────────────────────────────────────────

def process_comments(row: dict, out_dir: str):
    raw = row.get("comments") or []
    if not raw:
        return
    texts, metas = [], []
    for c in raw:
        if isinstance(c, dict):
            texts.append(c.get("content") or c.get("text") or c.get("body") or str(c))
            metas.append(c)
        else:
            texts.append(str(c)); metas.append({})

    print(f"    댓글 {len(texts)}개 LLM 분석 중 (id={row['id']})...")
    llm = analyze_comments_batch(texts)

    output = {
        "article_id": str(row["id"]),
        "comments_emotion": [
            {**metas[i], "text": texts[i],
             "emotion_label":     llm[i]["emotion_label"],
             "emotion_intensity": llm[i]["emotion_intensity"],
             "loaded_words":      llm[i]["loaded_words"]}
            for i in range(len(texts))
        ],
    }
    _save(output, os.path.join(out_dir, f"{row['id']}_comments.json"))


def _save(obj: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"    → 저장: {path}")


# ── 메인 ──────────────────────────────────────────────────

def run(keyword, limit, batch_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*55}\n  keyword={keyword!r}  limit={limit}  batch_size={batch_size}\n{'='*55}\n")

    print("[1] Supabase ai_test 로드 중...")
    rows = load_by_keyword(keyword, limit)
    print(f"    → {len(rows)}개 로드\n")
    if not rows:
        print("데이터 없음."); return

    print("[2] 뉴스 원문 분석...")
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        print(f"  배치 [{start+1}~{start+len(batch)}/{len(rows)}]")
        process_news_batch(batch, keyword, out_dir)

    print("\n[3] 댓글 감정 분석...")
    for row in rows:
        process_comments(row, out_dir)

    print(f"\n[완료] {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword",    required=True)
    ap.add_argument("--limit",      type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=3)
    ap.add_argument("--out_dir",    default="./results")
    a = ap.parse_args()
    run(a.keyword, a.limit, a.batch_size, a.out_dir)
