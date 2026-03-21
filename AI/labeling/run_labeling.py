#!/usr/bin/env python3
# labeling/run_labeling.py
"""
자동 라벨링 통합 진입점

실행:
    python AI/labeling/run_labeling.py --keyword 종소세
    python AI/labeling/run_labeling.py --keyword 종소세 --features emotion stance loaded_words
    python AI/labeling/run_labeling.py --keyword 종소세 --features frame logic bias omission

피처별 처리 담당:
    emotion, stance, loaded_words  → 로컬 HuggingFace (7~13B)
    frame, logic, bias_x, bias_y  → Groq Llama 3.3 70B
    omission_risk                 → Gemini 2.5 Pro (하루 100건)

저장:
    {out_dir}/{keyword}_{id}_labeled.json   ← 기사별 라벨 결과
    {out_dir}/reports/run_report_{ts}.json  ← 실행 리포트 (시간/품질/API)
    {out_dir}/reports/label_dist_{ts}.json  ← 레이블 분포 요약
"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── sys.path 설정 ──────────────────────────────────────────
_HERE = Path(__file__).resolve().parent   # AI/labeling/
_AI   = _HERE.parent                      # AI/

# Python은 스크립트 실행 시 스크립트 디렉토리를 sys.path[0]에 자동 추가함.
# AI/labeling/이 sys.path에 있으면 `from groq import Groq`가 SDK 대신
# 로컬 labeling/groq/ 폴더를 찾아버리므로 명시적으로 제거.
_here_str = str(_HERE)
while _here_str in sys.path:
    sys.path.remove(_here_str)

sys.path.insert(0, str(_AI))  # source.*, feature_map.*, labeling.*

# ── 임포트 ────────────────────────────────────────────────
from source.config.supabase_client import supabase
from labeling.research_report import ResearchReport

# 기사 피처 (emotion은 댓글 파이프라인에서 처리)
ALL_FEATURES = ["stance", "loaded_words", "frame", "logic", "bias", "omission"]


# ──────────────────────────────────────────────────────────
# Supabase 로드
# ──────────────────────────────────────────────────────────

def load_by_keyword(keyword: str, limit: int) -> list[dict]:
    """limit=0 이면 전체 페이지 가져옴."""
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


# ──────────────────────────────────────────────────────────
# 즉시 저장 헬퍼
# ──────────────────────────────────────────────────────────

def _save_partial(label_map: dict, article_ids: list, out_dir: str, keyword: str):
    """배치 완료 즉시 해당 기사만 저장 (에러 발생 시 완료분 보존)"""
    os.makedirs(out_dir, exist_ok=True)
    for aid in article_ids:
        labels = label_map[aid]
        path = os.path.join(out_dir, f"{keyword}_{aid}_labeled.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────
# Groq 블록 (frame / logic + bias)
# ──────────────────────────────────────────────────────────

def _run_groq_block(
    rows: list[dict],
    ids: list[str],
    texts: list[str],
    structs: list,
    features: list[str],
    batch_size: int,
    report: ResearchReport,
) -> "dict[str, dict]":
    """
    frame/logic 및 bias 라벨링을 수행하고
    aid → {frame, frame_reason, logic, logic_reason, frame_model_used,
            bias_x, bias_y, bias_rationale, bias_model_used} 매핑을 반환.
    """
    t0 = time.time()
    print(f"  [Groq thread] 시작 ({time.strftime('%H:%M:%S')})")

    result_map: dict[str, dict] = {}

    # ── 4. Groq 70B: frame / logic ────────────────────────
    if "frame" in features or "logic" in features:
        print("\n[Groq 70B] frame / logic 라벨링...")
        from labeling.groq.frame_labeler import label_frame_logic_batch

        for start in range(0, len(structs), batch_size):
            batch_structs = structs[start:start+batch_size]
            batch_ids     = ids[start:start+batch_size]
            n = len(batch_structs)

            with report.time_feature("frame_logic")(n):
                try:
                    results = label_frame_logic_batch(batch_structs)
                    report.record_api_call("groq")
                except Exception as e:
                    report.record_retry("groq", str(e))
                    print(f"  [경고] frame/logic 배치 실패 (건너뜀): {e}")
                    results = [{"frame": None, "logic": None, "frame_reason": "api_error", "logic_reason": "api_error"} for _ in batch_structs]

            try:
                from labeling.groq.groq_client import last_call_info as _g_info
                _groq_model = _g_info.get("model", "")
            except Exception:
                _groq_model = ""

            for i, res in enumerate(results):
                aid = batch_ids[i]
                if aid not in result_map:
                    result_map[aid] = {}
                result_map[aid].update({
                    "frame":             res["frame"],
                    "frame_reason":      res.get("frame_reason"),
                    "logic":             res["logic"],
                    "logic_reason":      res.get("logic_reason"),
                    "frame_model_used":  _groq_model,
                })
                report.record_labels("frame", [res["frame"]])
                report.record_labels("logic", [res["logic"]])
                report.record_sample("frame", aid, {
                    "frame": res["frame"],
                    "logic": res["logic"],
                    "frame_reason": res.get("frame_reason"),
                    "logic_reason": res.get("logic_reason"),
                })
                # fallback 감지
                if res["frame"] == "인과" and "인과" not in str(structs[start+i].get("judgment_words","")):
                    report.record_fallback("frame")
                if res["logic"] == "사실/정보 전달":
                    report.record_fallback("logic")

        print(f"  → {len(texts)}개 완료 (frame/logic)")

    # ── 5. Groq 70B: bias_x / bias_y ─────────────────────
    if "bias" in features:
        print("\n[Groq 70B] bias_x / bias_y 라벨링...")
        from labeling.groq.bias_labeler import label_bias_batch

        for start in range(0, len(structs), batch_size):
            batch_structs = structs[start:start+batch_size]
            batch_ids     = ids[start:start+batch_size]
            n = len(batch_structs)

            with report.time_feature("bias")(n):
                try:
                    results = label_bias_batch(batch_structs)
                    report.record_api_call("groq")
                except Exception as e:
                    report.record_retry("groq", str(e))
                    print(f"  [경고] bias 배치 실패 (건너뜀): {e}")
                    results = [{"bias_x": None, "bias_y": None, "bias_reason": "api_error"} for _ in batch_structs]

            try:
                from labeling.groq.groq_client import last_call_info as _g_info
                _groq_model = _g_info.get("model", "")
            except Exception:
                _groq_model = ""

            for i, res in enumerate(results):
                aid = batch_ids[i]
                if aid not in result_map:
                    result_map[aid] = {}
                result_map[aid].update({
                    "bias_x":          res["bias_x"],
                    "bias_y":          res["bias_y"],
                    "bias_rationale":  res.get("bias_reason"),
                    "bias_model_used": _groq_model,
                })
                report.record_labels("bias_x", [res["bias_x"]])
                report.record_labels("bias_y", [res["bias_y"]])
                report.record_sample("bias", aid, {
                    "bias_x":         res["bias_x"],
                    "bias_y":         res["bias_y"],
                    "bias_rationale": res.get("bias_reason"),
                })

        print(f"  → {len(texts)}개 완료 (bias)")

    elapsed = time.time() - t0
    print(f"  [Groq thread] 완료 ({time.strftime('%H:%M:%S')}, {elapsed:.1f}s)")
    return result_map


# ──────────────────────────────────────────────────────────
# Gemini 블록 (omission)
# ──────────────────────────────────────────────────────────

def _run_gemini_block(
    rows: list[dict],
    ids: list[str],
    texts: list[str],
    features: list[str],
    batch_size: int,
    max_gemini: int,
    report: ResearchReport,
    cluster_entity_lists: list[list[str]],
) -> "dict[str, dict]":
    """
    omission_risk 라벨링을 수행하고
    aid → {omission_risk, omission_reason, omission_model_used} 매핑을 반환.
    """
    t0 = time.time()
    print(f"  [Gemini thread] 시작 ({time.strftime('%H:%M:%S')})")

    result_map: dict[str, dict] = {}

    # ── 6. Gemini Pro: omission_risk ──────────────────────
    if "omission" in features:
        print("\n[Gemini Pro] omission_risk 라벨링...")
        from labeling.gemini.omission_labeler import label_omission_batch
        from labeling.gemini.gemini_client import rpd_remaining

        remaining = rpd_remaining()
        # max_gemini 적용 (0=무제한)
        if max_gemini > 0:
            remaining = min(remaining, max_gemini)
        print(f"  RPD 잔여: {rpd_remaining()}건  / 이번 실행 최대: {remaining}건")

        # omission_labeler가 기대하는 형식으로 article dict 구성
        articles = [
            {"id": ids[j], "title": rows[j]["title"],
             "body_snippet": rows[j]["body"][:400]}
            for j in range(len(rows))
        ]

        for start in range(0, len(articles), batch_size):
            batch_articles = articles[start:start+batch_size]
            # 배치 내 클러스터 엔티티를 하나의 flat list로 합산
            seen: set = set()
            batch_cluster: list = []
            for cl in cluster_entity_lists[start:start+batch_size]:
                for e in cl:
                    if e not in seen:
                        batch_cluster.append(e)
                        seen.add(e)
            batch_ids = ids[start:start+batch_size]
            n = len(batch_articles)

            with report.time_feature("omission")(n):
                try:
                    results = label_omission_batch(batch_articles, batch_cluster)
                    report.record_api_call("gemini_pro")
                except Exception as e:
                    report.record_retry("gemini_pro", str(e))
                    print(f"  [경고] omission 배치 실패 (건너뜀): {e}")
                    results = [{"omission_risk": None, "omission_reason": str(e)} for _ in batch_articles]

            try:
                from labeling.gemini.gemini_client import last_call_info as _gem_info
                _gemini_model = _gem_info.get("model", "")
            except Exception:
                _gemini_model = ""

            for i, res in enumerate(results):
                aid = batch_ids[i]
                result_map[aid] = {
                    "omission_risk":       res["omission_risk"],
                    "omission_reason":     res.get("omission_reason"),
                    "omission_model_used": _gemini_model,
                }
                report.record_labels("omission_risk", [res["omission_risk"]])
                report.record_sample("omission", aid, {
                    "omission_risk":   res["omission_risk"],
                    "omission_reason": res.get("omission_reason"),
                })

        print(f"  → {len(texts)}개 완료 (omission)")

    elapsed = time.time() - t0
    print(f"  [Gemini thread] 완료 ({time.strftime('%H:%M:%S')}, {elapsed:.1f}s)")
    return result_map


# ──────────────────────────────────────────────────────────
# 라벨링 실행
# ──────────────────────────────────────────────────────────

def run_labeling(rows: list[dict], features: list[str], batch_size: int,
                 out_dir: str, report: ResearchReport,
                 max_gemini: int = 200,
                 parallel: bool = True):

    texts  = [f"{r['title']}\n\n{r['body']}" for r in rows]
    ids    = [str(r["id"]) for r in rows]
    label_map = {aid: {"article_id": aid, "title": rows[i].get("title", "")}
                 for i, aid in enumerate(ids)}
    keyword = report.keyword

    # 텍스트/댓글 통계 기록
    comment_counts = [len(r.get("comments") or []) for r in rows]
    report.record_texts(texts, comment_counts)

    # ── 1. 로컬: stance_score ─────────────────────────────
    if "stance" in features:
        print("\n[로컬] stance_score 라벨링...")
        from labeling.local.stance_labeler import label_stance_batch
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start+batch_size]
            batch_ids   = ids[start:start+batch_size]
            n = len(batch_texts)

            with report.time_feature("stance")(n):
                results = label_stance_batch(batch_texts)

            for i, res in enumerate(results):
                aid = batch_ids[i]
                label_map[aid].update({
                    "stance_score": res["stance_score"],
                    "stance_label": res["stance_label"],
                })
                report.record_labels("stance_score", [res["stance_score"]])
                report.record_labels("stance_label", [res["stance_label"]])
                report.record_sample("stance", aid, {
                    "stance_score": res["stance_score"],
                    "stance_label": res["stance_label"],
                })
            _save_partial(label_map, batch_ids, out_dir, keyword)

        print(f"  → {len(texts)}개 완료")

    # ── 3. 로컬: loaded_words ─────────────────────────────
    if "loaded_words" in features:
        print("\n[로컬] loaded_words 라벨링...")
        from labeling.local.loaded_words_labeler import label_loaded_words_batch
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start+batch_size]
            batch_ids   = ids[start:start+batch_size]
            n = len(batch_texts)

            with report.time_feature("loaded_words")(n):
                results = label_loaded_words_batch(batch_texts)

            for i, res in enumerate(results):
                aid = batch_ids[i]
                label_map[aid].update({
                    "art_words":           res["loaded_words"],      # ERD: article_features.art_words
                    "loaded_word_density": res["loaded_word_density"],
                    "is_biased":           res["is_biased"],
                })
                report.record_labels("loaded_word_density", [res["loaded_word_density"]])
                report.record_labels("is_biased", [res["is_biased"]])
                report.record_labels("art_words_flat", res["loaded_words"])  # 상위 편향 단어 집계용
                report.record_sample("loaded_words", aid, {
                    "art_words":           res["loaded_words"],
                    "loaded_word_density": res["loaded_word_density"],
                    "is_biased":           res["is_biased"],
                })
            _save_partial(label_map, batch_ids, out_dir, keyword)

        print(f"  → {len(texts)}개 완료")

    # ── 사전 계산: structs (Groq에서 사용) ───────────────
    needs_groq   = ("frame" in features or "logic" in features or "bias" in features)
    needs_gemini = ("omission" in features)

    structs: list = []
    if needs_groq:
        from feature_map.stance.src.feature_map.preprocessor import build_article_struct
        structs = [build_article_struct(t) for t in texts]

    # ── 사전 계산: cluster_entity_lists (Gemini에서 사용) ─
    cluster_entity_lists: list = []
    if needs_gemini:
        from feature_map.context.src.feature_map.keyword_extractor import extract_features
        from collections import Counter
        for i in range(len(texts)):
            counter = Counter()
            cluster = [texts[j] for j in range(len(texts)) if j != i]
            for ct in cluster:
                try:
                    feats = extract_features(ct)
                    for e in feats["entities"]:
                        counter[e["word"]] += 1
                except Exception:
                    continue
            total = max(len(cluster), 1)
            core  = [ent for ent, cnt in counter.items() if cnt/total >= 0.3]
            cluster_entity_lists.append(core)

    # ── API 블록 실행 (parallel or sequential) ────────────
    groq_result:   dict[str, dict] = {}
    gemini_result: dict[str, dict] = {}

    if needs_groq or needs_gemini:
        if parallel and needs_groq and needs_gemini:
            print(f"\n[병렬 실행] Groq + Gemini 동시 시작 ({time.strftime('%H:%M:%S')})")
            t_parallel_start = time.time()

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_groq = executor.submit(
                    _run_groq_block,
                    rows, ids, texts, structs, features, batch_size, report,
                )
                future_gemini = executor.submit(
                    _run_gemini_block,
                    rows, ids, texts, features, batch_size, max_gemini, report,
                    cluster_entity_lists,
                )
                groq_result   = future_groq.result()
                gemini_result = future_gemini.result()

            elapsed_parallel = time.time() - t_parallel_start
            print(f"\n[병렬 실행] 완료 (총 {elapsed_parallel:.1f}s)")

        else:
            # sequential fallback (또는 한쪽만 필요한 경우)
            if needs_groq:
                groq_result = _run_groq_block(
                    rows, ids, texts, structs, features, batch_size, report,
                )
            if needs_gemini:
                gemini_result = _run_gemini_block(
                    rows, ids, texts, features, batch_size, max_gemini, report,
                    cluster_entity_lists,
                )

        # ── 결과 병합 ──────────────────────────────────────
        for aid, data in groq_result.items():
            label_map[aid].update(data)
        for aid, data in gemini_result.items():
            label_map[aid].update(data)

        # ── 키 로테이션 로그 수집 (스레드 완료 후 메인 스레드에서) ──
        if needs_groq:
            try:
                from labeling.groq.groq_client import rotation_log as groq_rotation_log
                for event in groq_rotation_log:
                    report.record_key_rotation(
                        from_key=event.get("from_key", 0),
                        to_key=event.get("to_key", 1),
                        reason=event.get("reason", ""),
                    )
                groq_rotation_log.clear()
            except Exception:
                pass

        if needs_gemini:
            try:
                from labeling.gemini.gemini_client import rotation_log as gemini_rotation_log
                for event in gemini_rotation_log:
                    report.record_key_rotation(
                        from_key=event.get("from_key", 0),
                        to_key=event.get("to_key", 1),
                        reason=event.get("reason", ""),
                    )
                gemini_rotation_log.clear()
            except Exception:
                pass

    # ── 최종 저장 (모든 피처 완료 후 전체 덮어쓰기) ──────────
    print("\n[저장] 라벨 결과 최종 저장...")
    _save_partial(label_map, ids, out_dir, keyword)
    for aid in ids:
        print(f"  → {os.path.join(out_dir, f'{keyword}_{aid}_labeled.json')}")


# ──────────────────────────────────────────────────────────
# 댓글 라벨링 (cmt_emotion + cmt_words) — API 미사용, 로컬/규칙 기반
# ──────────────────────────────────────────────────────────

def _process_one_article_comments(row: dict, out_dir: str, keyword: str,
                                   cmt_batch_size: int,
                                   report: ResearchReport | None,
                                   report_lock: threading.Lock | None) -> str:
    """
    단일 기사의 댓글을 처리하고 저장. run_comments_labeling에서 병렬 호출됨.
    반환: 저장된 파일 경로 (댓글 없으면 "")
    """
    from labeling.local.emotion_labeler      import label_emotion_batch
    from labeling.local.loaded_words_labeler import label_loaded_words_batch

    aid      = str(row["id"])
    raw_cmts = row.get("comments") or []
    if not raw_cmts:
        return ""

    texts, metas = [], []
    for c in raw_cmts:
        if isinstance(c, dict):
            txt  = c.get("content") or c.get("text") or c.get("body") or ""
            meta = {k: v for k, v in c.items() if k not in ("content", "text", "body")}
        else:
            txt, meta = str(c), {}
        if txt.strip():
            texts.append(txt)
            metas.append(meta)

    if not texts:
        return ""

    emotion_results: list = []
    words_results:   list = []
    for start in range(0, len(texts), cmt_batch_size):
        chunk = texts[start:start + cmt_batch_size]
        try:
            emotion_results.extend(label_emotion_batch(chunk))
        except Exception as _e:
            emotion_results.extend([{
                "primary_emotion": "neutral", "raw_label": "neutral",
                "emotion_intensity": 0.0, "emotion_probs": {},
            }] * len(chunk))
        try:
            words_results.extend(label_loaded_words_batch(chunk))
        except Exception as _e:
            words_results.extend([{
                "loaded_words": [], "loaded_word_density": 0.0, "is_biased": False,
            }] * len(chunk))

    labeled_comments = []
    for i, (txt, meta) in enumerate(zip(texts, metas)):
        emo_label = emotion_results[i]["primary_emotion"]
        cmt_words = words_results[i]["loaded_words"]
        labeled_comments.append({
            **meta,
            "cmt_comment": txt,
            "cmt_emotion": {
                "label":     emo_label,
                "intensity": emotion_results[i]["emotion_intensity"],
                "probs":     emotion_results[i]["emotion_probs"],
            },
            "cmt_words": cmt_words,
        })
        if report is not None:
            if report_lock:
                with report_lock:
                    report.record_comment_labels(emo_label, cmt_words)
            else:
                report.record_comment_labels(emo_label, cmt_words)

    out = {"article_id": aid, "comments": labeled_comments}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{keyword}_{aid}_comments.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return path


def run_comments_labeling(rows: list[dict], out_dir: str, keyword: str,
                          cmt_batch_size: int = 32,
                          report: ResearchReport | None = None,
                          parallel_workers: int = 4):
    """
    각 기사의 댓글마다 cmt_emotion / cmt_words 추출 후
    {out_dir}/{keyword}_{id}_comments.json 으로 저장.

    로컬 전용 (API 미사용). parallel_workers 수만큼 기사를 동시에 처리.
    """
    os.makedirs(out_dir, exist_ok=True)
    report_lock = threading.Lock() if parallel_workers > 1 else None
    active_rows = [r for r in rows if (r.get("comments") or [])]

    if not active_rows:
        print("  댓글이 있는 기사 없음, 건너뜀.")
        return

    print(f"  댓글 병렬 처리: {len(active_rows)}개 기사 × {parallel_workers} workers")

    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        futures = {
            pool.submit(
                _process_one_article_comments,
                row, out_dir, keyword, cmt_batch_size, report, report_lock
            ): str(row["id"])
            for row in active_rows
        }
        done, total = 0, len(futures)
        for future in as_completed(futures):
            aid = futures[future]
            done += 1
            try:
                path = future.result()
                if path:
                    print(f"  [{done}/{total}] 저장: {path}")
            except Exception as e:
                print(f"  [경고] article_id={aid} 댓글 처리 실패: {e}")


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="자동 라벨링 파이프라인")
    ap.add_argument("--keyword",    required=True)
    ap.add_argument("--features",   nargs="+", default=["all"],
                    help=f"라벨링 피처. 'all' 또는 {ALL_FEATURES}")
    ap.add_argument("--limit",        type=int, default=0,
                    help="처리할 기사 수 (기본: 0 = 전체)")
    ap.add_argument("--offset",       type=int, default=0,
                    help="앞에서 N개 건너뜀 (대용량 재개용, 기본: 0)")
    ap.add_argument("--batch_size",   type=int, default=20,
                    help="API 1회 호출당 기사 수 (기본: 20 / Groq 권장 10~30)")
    ap.add_argument("--cmt_batch",    type=int, default=32,
                    help="댓글 로컬 처리 배치 크기 (기본: 32)")
    ap.add_argument("--cmt_workers",  type=int, default=4,
                    help="댓글 병렬 처리 worker 수 (기본: 4)")
    ap.add_argument("--max_gemini",   type=int, default=200,
                    help="이번 실행에서 Gemini로 처리할 최대 기사 수 (기본: 200, 0=무제한)")
    ap.add_argument("--skip_comments", action="store_true",
                    help="댓글 라벨링 건너뜀")
    ap.add_argument("--no_parallel",  action="store_false", dest="parallel",
                    help="Groq/Gemini 병렬 실행 비활성화 (순차 실행)")
    ap.set_defaults(parallel=True)
    ap.add_argument("--out_dir",     default="./label_results")
    args = ap.parse_args()

    features = ALL_FEATURES if "all" in args.features else args.features
    invalid  = [f for f in features if f not in ALL_FEATURES]
    if invalid:
        print(f"알 수 없는 피처: {invalid}\n사용 가능: {ALL_FEATURES}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  keyword={args.keyword!r}  features={features}")
    print(f"  limit={args.limit}  offset={args.offset}  batch_size={args.batch_size}")
    print(f"  댓글 배치={args.cmt_batch}  댓글 workers={args.cmt_workers}  댓글 스킵={args.skip_comments}")
    print(f"  max_gemini={args.max_gemini} (0=무제한)")
    print(f"  parallel={args.parallel}")
    print(f"{'='*55}\n")

    # 리포트 초기화
    report = ResearchReport(
        keyword=args.keyword,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    print("[1] Supabase ai_test 로드...")
    rows = load_by_keyword(args.keyword, args.limit)
    print(f"    → 전체 {len(rows)}개")

    # offset 적용 (대용량 재개용)
    if args.offset > 0:
        rows = rows[args.offset:]
        print(f"    → offset {args.offset} 적용 → {len(rows)}개 처리")
    print()

    if not rows:
        print("처리할 데이터 없음."); sys.exit(0)

    print("[2] 기사 라벨링 시작...")
    try:
        run_labeling(rows, features, args.batch_size, args.out_dir, report,
                     max_gemini=args.max_gemini,
                     parallel=args.parallel)
    except Exception as _article_err:
        print(f"\n[경고] 기사 라벨링 중 오류 발생 (댓글 라벨링은 계속 진행):\n  {_article_err}\n")

    if not args.skip_comments:
        print("\n[3] 댓글 라벨링 시작 (cmt_emotion + cmt_words)...")
        run_comments_labeling(rows, args.out_dir, args.keyword, args.cmt_batch,
                              report, parallel_workers=args.cmt_workers)
    else:
        print("\n[3] 댓글 라벨링 건너뜀 (--skip_comments)")

    # 리포트 저장 (5개 파일)
    print("\n[4] 연구 리포트 저장...")
    report_dir = os.path.join(args.out_dir, "reports")
    report_path, dist_path = report.save(args.out_dir)
    print(f"  → 실행 리포트:    {report_path}")
    print(f"  → 분포 요약:      {dist_path}")
    print(f"  → 피처 타이밍:    {report_dir}/feature_timing_*.json")
    print(f"  → 샘플 라벨:      {report_dir}/sample_labels_*.json")
    print(f"  → 댓글 통계:      {report_dir}/comments_stats_*.json")

    # 콘솔 요약 출력
    report.print_summary()

    print(f"[완료] {os.path.abspath(args.out_dir)}")
