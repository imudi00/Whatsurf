#!/usr/bin/env python3
# labeling/run_labeling.py
"""
자동 라벨링 통합 진입점

실행:
    python AI/labeling/run_labeling.py --keyword 종소세
    python AI/labeling/run_labeling.py --keyword 종소세 --features emotion stance loaded_words
    python AI/labeling/run_labeling.py --keyword 종소세 --features frame logic bias omission

    --resume 옵션 추가 (중단된 라벨링 재개용)
    --skip_comments 옵션 추가 (댓글 라벨링 건너뜀)

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
ALL_FEATURES = ["stance", "loaded_words", "body_depth", "frame", "logic", "bias", "omission"]

# 피처별 api_error 판별 필드
_ERROR_FIELDS: dict[str, list[str]] = {
    "frame":    ["frame_reason", "logic_reason"],
    "logic":    ["frame_reason", "logic_reason"],
    "bias":     ["bias_rationale"],
    "omission": ["omission_reason"],
    "stance":   ["stance_score"],
}

def _is_error(data: dict, feature: str) -> bool:
    """해당 피처가 api_error 또는 미처리(null) 상태인지 확인."""
    fields = _ERROR_FIELDS.get(feature, [])
    for field in fields:
        val = data.get(field)
        if val == "api_error" or val is None:
            return True
    # frame/logic/bias/omission 결과값 자체도 체크
    direct = {
        "frame":    "frame",
        "logic":    "logic",
        "bias":     "bias_x",
        "omission": "omission_risk",
        "stance":   "stance_score",
    }
    key = direct.get(feature)
    if key and data.get(key) is None:
        return True
    return False


def find_error_articles(out_dir: str, keyword: str, features: list[str]) -> dict[str, set[str]]:
    """
    out_dir 안의 {keyword}_*_labeled.json 을 스캔하여
    feature별 api_error / null 상태인 article_id set을 반환.

    반환: {"frame": {"123","456",...}, "bias": {...}, ...}
    """
    import glob
    pattern = os.path.join(out_dir, f"{keyword}_*_labeled.json")
    files = glob.glob(pattern)

    errors: dict[str, set[str]] = {f: set() for f in features}
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        aid = str(data.get("article_id", ""))
        if not aid:
            continue
        for feat in features:
            if feat in _ERROR_FIELDS and _is_error(data, feat):
                errors[feat].add(aid)

    return errors


# ──────────────────────────────────────────────────────────
# Supabase 로드
# ──────────────────────────────────────────────────────────

def load_by_keyword(keyword: str, limit: int) -> list[dict]:
    """ai_test 테이블에서 keyword로 로드. limit=0 이면 전체."""
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


def load_by_query_id(query_id: str, limit: int) -> list[dict]:
    """
    Supabase2 articles 테이블에서 query_id로 로드.
    각 기사의 댓글은 comments 테이블(article_id 공유)에서 cmt_content 로 가져옴.
    limit=0 이면 전체.
    """
    from source.config.supabase2_client import supabase2
    from collections import defaultdict

    # ── 기사 로드 ────────────────────────────────────────
    PAGE_SIZE = 1000
    rows: list = []
    offset = 0
    while True:
        fetch = PAGE_SIZE if limit == 0 else min(PAGE_SIZE, limit - len(rows))
        resp = (
            supabase2.table("articles")
            .select("id, query_id, title, body_text")
            .eq("query_id", query_id)
            .range(offset, offset + fetch - 1)
            .execute()
        )
        page = resp.data or []
        rows.extend(page)
        if len(page) < fetch or (limit > 0 and len(rows) >= limit):
            break
        offset += fetch
    if limit > 0:
        rows = rows[:limit]

    # 컬럼 정규화
    for r in rows:
        r["body"] = r.pop("body_text", "") or ""

    # ── 댓글 로드 (comments 테이블, article_id 기준) ──────
    print("  [Supabase2] comments 테이블 로드 중...")
    article_ids = [r["id"] for r in rows]
    comments_by_article: dict = defaultdict(list)

    CMT_BATCH = 100  # in_ 필터 1회 최대
    for i in range(0, len(article_ids), CMT_BATCH):
        batch_ids = article_ids[i:i + CMT_BATCH]
        cmt_resp = (
            supabase2.table("comments")
            .select("id, article_id, cmt_content")
            .in_("article_id", batch_ids)
            .execute()
        )
        for c in (cmt_resp.data or []):
            # content 키로 정규화해 두면 _process_one_article_comments 가 자동 인식
            comments_by_article[c["article_id"]].append({
                "id":      c["id"],          # DB PK → 업로드 시 사용
                "content": c["cmt_content"],
            })

    total_cmts = sum(len(v) for v in comments_by_article.values())
    print(f"  → 댓글 {total_cmts}개 로드 완료")

    for r in rows:
        r["comments"] = comments_by_article.get(r["id"], [])

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
                    print(f"  [경고] frame/logic 배치 실패 → 1개씩 재시도: {e}")
                    results = []
                    for j, single_struct in enumerate(batch_structs):
                        try:
                            res = label_frame_logic_batch([single_struct])
                            results.append(res[0])
                            report.record_api_call("groq")
                        except Exception as e2:
                            print(f"    [경고] 단일 실패 (idx={start+j}): {e2}")
                            results.append({"frame": None, "logic": None,
                                            "frame_reason": "api_error", "logic_reason": "api_error"})

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
                    print(f"  [경고] bias 배치 실패 → 1개씩 재시도: {e}")
                    results = []
                    for j, single_struct in enumerate(batch_structs):
                        try:
                            res = label_bias_batch([single_struct])
                            results.append(res[0])
                            report.record_api_call("groq")
                        except Exception as e2:
                            print(f"    [경고] 단일 실패 (idx={start+j}): {e2}")
                            results.append({"bias_x": None, "bias_y": None, "bias_reason": "api_error"})

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
                    print(f"  [경고] omission 배치 실패 → 1개씩 재시도: {e}")
                    results = []
                    for j, single_article in enumerate(batch_articles):
                        try:
                            res = label_omission_batch([single_article], batch_cluster)
                            results.append(res[0])
                            report.record_api_call("gemini_pro")
                        except Exception as e2:
                            print(f"    [경고] 단일 실패 (idx={start+j}): {e2}")
                            results.append({"omission_risk": None, "omission_reason": "api_error"})

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
                 parallel: bool = True,
                 resume: bool = False):

    texts  = [f"{r['title']}\n\n{r['body']}" for r in rows]
    ids    = [str(r["id"]) for r in rows]
    keyword = report.keyword

    # ── resume: 기존 _labeled.json 로드해서 label_map 초기화 ──
    label_map: dict[str, dict] = {}
    for i, aid in enumerate(ids):
        base = {"article_id": aid, "title": rows[i].get("title", "")}
        if resume:
            path = os.path.join(out_dir, f"{keyword}_{aid}_labeled.json")
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        existing = json.load(f)
                    base = existing  # 기존 피처 전부 유지
                    print(f"  [resume] 기존 로드: {os.path.basename(path)}")
                except Exception as e:
                    print(f"  [resume] 로드 실패 ({aid}): {e}")
        label_map[aid] = base

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

    # ── 2. 로컬: body_depth (수식 기반, 모델 미사용) ────────
    if "body_depth" in features:
        print("\n[로컬] body_depth 계산 (수식 기반)...")
        from labeling.local.body_depth import compute_body_depth, describe_body_depth
        t0 = time.time()
        for i, (aid, text) in enumerate(zip(ids, texts)):
            score = compute_body_depth(text)
            label_map[aid].update({
                "body_depth":       score,
                "body_depth_label": describe_body_depth(score),
            })
            report.record_labels("body_depth", [score])
        elapsed = time.time() - t0
        print(f"  → {len(texts)}개 완료 ({elapsed:.1f}s)")
        _save_partial(label_map, ids, out_dir, keyword)

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
        from labeling.features.preprocessor import build_article_struct
        structs = [build_article_struct(t) for t in texts]

    # ── 사전 계산: cluster_entity_lists (Gemini에서 사용) ─
    # 기사별 NER을 1회씩만 수행(O(n)), 전체에서 ≥30% 출현 엔티티를 공통 클러스터로 사용
    cluster_entity_lists: list = []
    if needs_gemini:
        from labeling.features.keyword_extractor import extract_features
        from collections import Counter

        print("  [Gemini 전처리] 클러스터 엔티티 추출 중 (기사당 1회)...")
        all_entities: list[list[str]] = []
        for idx, t in enumerate(texts):
            try:
                feats = extract_features(t)
                ents  = [e["word"] for e in feats.get("entities", [])]
            except Exception:
                ents = []
            all_entities.append(ents)
            if (idx + 1) % 20 == 0 or (idx + 1) == len(texts):
                print(f"    {idx+1}/{len(texts)} 완료")

        total   = max(len(texts), 1)
        counter = Counter(e for ents in all_entities for e in ents)
        core    = [ent for ent, cnt in counter.items() if cnt / total >= 0.3]
        print(f"  → 핵심 엔티티 {len(core)}개: {core[:10]}{'...' if len(core) > 10 else ''}")
        cluster_entity_lists = [core] * len(texts)

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
                          parallel_workers: int = 4,
                          resume: bool = False):
    """
    각 기사의 댓글마다 cmt_emotion / cmt_words 추출 후
    {out_dir}/{keyword}_{id}_comments.json 으로 저장.

    로컬 전용 (API 미사용). parallel_workers 수만큼 기사를 동시에 처리.
    resume=True 이면 이미 _comments.json 이 존재하는 기사는 건너뜀.
    """
    os.makedirs(out_dir, exist_ok=True)
    report_lock = threading.Lock() if parallel_workers > 1 else None
    active_rows = [r for r in rows if (r.get("comments") or [])]

    if resume:
        before = len(active_rows)
        active_rows = [
            r for r in active_rows
            if not os.path.exists(
                os.path.join(out_dir, f"{keyword}_{r['id']}_comments.json")
            )
        ]
        skipped = before - len(active_rows)
        if skipped:
            print(f"  [resume] 이미 완료된 댓글 {skipped}개 건너뜀")

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

    # ── 데이터 소스 ───────────────────────────────────────
    ap.add_argument(
        "--source", choices=["ai_test", "supabase2"], default="ai_test",
        help="데이터 소스 (기본: ai_test / 신규: supabase2)",
    )
    ap.add_argument(
        "--keyword",
        help="ai_test 소스: keyword 필드 값 (--source ai_test 에서 필수)",
    )
    ap.add_argument(
        "--query_id",
        help="supabase2 소스: query_id 필드 값 (--source supabase2 에서 필수)",
    )

    # ── 피처 / 처리 옵션 ─────────────────────────────────
    ap.add_argument("--features",   nargs="+", default=["all"],
                    help=f"라벨링 피처. 'all' 또는 {ALL_FEATURES}")
    ap.add_argument("--limit",        type=int, default=0,
                    help="처리할 기사 수 (기본: 0 = 전체)")
    ap.add_argument("--offset",       type=int, default=0,
                    help="앞에서 N개 건너뜀 (대용량 재개용, 기본: 0)")
    ap.add_argument("--batch_size",   type=int, default=100,
                    help="API 1회 호출당 기사 수 (기본: 100)")
    ap.add_argument("--cmt_batch",    type=int, default=32,
                    help="댓글 로컬 처리 배치 크기 (기본: 32)")
    ap.add_argument("--cmt_workers",  type=int, default=4,
                    help="댓글 병렬 처리 worker 수 (기본: 4)")
    ap.add_argument("--max_gemini",   type=int, default=0,
                    help="Gemini 최대 처리 건수 (기본: 0=무제한)")
    ap.add_argument("--skip_comments", action="store_true",
                    help="댓글 라벨링 건너뜀")
    ap.add_argument("--resume",        action="store_true",
                    help="기존 _labeled.json/_comments.json 유지 + 지정 피처만 추가")
    ap.add_argument("--retry_errors",  action="store_true",
                    help="api_error / null 인 피처만 골라서 재시도 (resume 자동 활성화)")
    ap.add_argument("--no_parallel",  action="store_false", dest="parallel",
                    help="Groq/Gemini 병렬 실행 비활성화 (순차 실행)")
    ap.set_defaults(parallel=True)
    ap.add_argument("--out_dir",     default="./label_results")
    args = ap.parse_args()

    # ── 소스별 필수 인자 검증 ─────────────────────────────
    if args.source == "ai_test" and not args.keyword:
        ap.error("--source ai_test 일 때 --keyword 가 필요합니다.")
    if args.source == "supabase2" and not args.query_id:
        ap.error("--source supabase2 일 때 --query_id 가 필요합니다.")

    # 리포트용 식별자 (keyword or query_id)
    run_label = args.keyword if args.source == "ai_test" else args.query_id

    features = ALL_FEATURES if "all" in args.features else args.features
    invalid  = [f for f in features if f not in ALL_FEATURES]
    if invalid:
        print(f"알 수 없는 피처: {invalid}\n사용 가능: {ALL_FEATURES}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  source={args.source!r}  label={run_label!r}")
    print(f"  features={features}")
    print(f"  limit={args.limit}  offset={args.offset}  batch_size={args.batch_size}")
    print(f"  댓글 배치={args.cmt_batch}  댓글 workers={args.cmt_workers}  댓글 스킵={args.skip_comments}")
    print(f"  max_gemini={args.max_gemini} (0=무제한)  parallel={args.parallel}")
    print(f"{'='*60}\n")

    # 리포트 초기화
    report = ResearchReport(
        keyword=run_label,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    # ── 데이터 로드 ───────────────────────────────────────
    if args.source == "ai_test":
        print("[1] Supabase ai_test 로드...")
        rows = load_by_keyword(args.keyword, args.limit)
    else:
        print("[1] Supabase2 articles 로드...")
        rows = load_by_query_id(args.query_id, args.limit)
    print(f"    → 전체 {len(rows)}개")

    # offset 적용 (대용량 재개용)
    if args.offset > 0:
        rows = rows[args.offset:]
        print(f"    → offset {args.offset} 적용 → {len(rows)}개 처리")
    print()

    if not rows:
        print("처리할 데이터 없음."); sys.exit(0)

    if args.resume:
        print("  [resume 모드] 기존 결과 유지 + 지정 피처만 추가\n")

    # ── retry_errors: api_error / null 항목만 필터링 ──────────
    if args.retry_errors:
        args.resume = True  # 기존 값 보존 필수
        retryable = [f for f in features if f in _ERROR_FIELDS]
        error_map = find_error_articles(args.out_dir, run_label, retryable)

        # 재시도 대상 ID 합집합
        all_error_ids = set()
        for feat, ids_set in error_map.items():
            all_error_ids |= ids_set

        if not all_error_ids:
            print("  [retry_errors] api_error 항목 없음 — 재시도 불필요.\n")
        else:
            # 피처별 에러 현황 출력
            print("  [retry_errors] api_error 감지:")
            for feat, ids_set in error_map.items():
                if ids_set:
                    print(f"    {feat}: {len(ids_set)}건")

            # rows를 에러 ID만으로 필터링
            rows = [r for r in rows if str(r["id"]) in all_error_ids]
            # features도 실제 에러가 있는 피처만으로 축소
            features = [f for f in features if error_map.get(f)]
            print(f"  → 재시도 대상: {len(rows)}개 기사, 피처: {features}\n")

    print("[2] 기사 라벨링 시작...")
    try:
        run_labeling(rows, features, args.batch_size, args.out_dir, report,
                     max_gemini=args.max_gemini,
                     parallel=args.parallel,
                     resume=args.resume)
    except Exception as _article_err:
        print(f"\n[경고] 기사 라벨링 중 오류 발생 (댓글 라벨링은 계속 진행):\n  {_article_err}\n")

    if not args.skip_comments:
        print("\n[3] 댓글 라벨링 시작 (cmt_emotion + cmt_words)...")
        run_comments_labeling(rows, args.out_dir, run_label, args.cmt_batch,
                              report, parallel_workers=args.cmt_workers,
                              resume=args.resume)
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
