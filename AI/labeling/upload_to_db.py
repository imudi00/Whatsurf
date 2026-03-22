#!/usr/bin/env python3
"""
upload_to_db.py — label_results/ 결과물을 Supabase2 에 업로드(upsert)

실행 예시:
    # 기사 피처만 (article_features 테이블)
    python AI/labeling/upload_to_db.py --mode articles --keyword 방탄광화문

    # 댓글 피처만 (comments 테이블 — cmt_emotion, cmt_words 컬럼 업데이트)
    python AI/labeling/upload_to_db.py --mode comments --keyword 방탄광화문

    # 기사 + 댓글 한번에
    python AI/labeling/upload_to_db.py --mode all --keyword 방탄광화문

    # label_results 전체 업로드
    python AI/labeling/upload_to_db.py --mode all --all

    # 실제 업로드 없이 미리보기
    python AI/labeling/upload_to_db.py --mode articles --keyword 방탄광화문 --dry_run

field 매핑 (labeled.json → article_features):
    article_id                                     → article_id
    v1.0 + frame_model_used + bias_model_used
          + omission_model_used                    → model_version
    frame         (str)  → frame_id  (int 1~7)
    logic         (str)  → logic_id  (int 1~6)
    omission_risk (str)  → omission_risk (int -1/0/1)
    stance_score, bias_x, bias_y, art_words,
    body_depth                                     → 동일명 그대로

field 매핑 (comments.json → comments 테이블):
    id (DB comment PK)   → id          (on_conflict 기준)
    cmt_emotion.label    → cmt_emotion (str)
    cmt_words            → cmt_words   (array)
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── sys.path ───────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent   # AI/labeling/
_AI   = _HERE.parent                      # AI/

_here_str = str(_HERE)
while _here_str in sys.path:
    sys.path.remove(_here_str)
sys.path.insert(0, str(_AI))

from source.config.supabase2_client import supabase2


# ──────────────────────────────────────────────────────────
# 코드 매핑 (이미지 기준 하드코딩)
# ──────────────────────────────────────────────────────────

FRAME_MAP: dict[str, int] = {
    "사건 원인 집중":   1,
    "갈등/대립 강조":   2,
    "개인 사례 중심":   3,
    "경제적 영향 강조": 4,
    "윤리/도덕 판단":   5,
    "안전/안보 위협":   6,
    "권리/인권 강조":   7,
}

LOGIC_MAP: dict[str, int] = {
    "정책적 비난":    1,
    "전문가 견해":    2,
    "피해자 서사":    3,
    "파급효과":       4,
    "해결책 제시":    5,
    "사실/정보 전달": 6,
}

OMISSION_MAP: dict[str, int] = {
    "low":    -1,
    "mid":     0,
    "med":     0,   # omission_labeler 기본값
    "medium":  0,   # LLM이 medium으로 출력하는 경우 대비
    "high":    1,
}


# ──────────────────────────────────────────────────────────
# 변환 헬퍼
# ──────────────────────────────────────────────────────────

def _to_frame_id(val: Optional[str]) -> Optional[int]:
    """frame 문자열 → int (1~7). 부분 일치 포함."""
    if not val:
        return None
    if val in FRAME_MAP:
        return FRAME_MAP[val]
    for k, v in FRAME_MAP.items():
        if k in val or val in k:
            return v
    return None


def _to_logic_id(val: Optional[str]) -> Optional[int]:
    """logic 문자열 → int (1~6). 부분 일치 포함."""
    if not val:
        return None
    if val in LOGIC_MAP:
        return LOGIC_MAP[val]
    for k, v in LOGIC_MAP.items():
        if k in val or val in k:
            return v
    return None


def _to_omission_int(val) -> Optional[int]:
    """high → 1 / mid → 0 / low → -1"""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    return OMISSION_MAP.get(str(val).lower().strip())


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    if not isinstance(v, (int, float)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _build_model_version(d: dict) -> str:
    frame_m    = d.get("frame_model_used", "") or ""
    bias_m     = d.get("bias_model_used",  "") or ""
    omission_m = d.get("omission_model_used", "") or ""
    return f"v1.0+{frame_m}+{bias_m}+{omission_m}"


def _to_int(v) -> Optional[int]:
    """문자열 포함 정수 변환 (실패 시 None)."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def labeled_to_feature_row(d: dict) -> dict:
    """labeled.json dict → article_features 행"""
    return {
        "article_id":    _to_int(d.get("article_id")),   # int 강제 (DB 타입 일치)
        "model_version": _build_model_version(d),
        "frame_id":      _to_frame_id(d.get("frame")),
        "logic_id":      _to_logic_id(d.get("logic")),
        "stance_score":  _to_float(d.get("stance_score")),
        "bias_x":        _to_float(d.get("bias_x")),
        "bias_y":        _to_float(d.get("bias_y")),
        "art_words":     d.get("art_words") or [],
        "body_depth":    _to_float(d.get("body_depth")),
        "omission_risk": _to_omission_int(d.get("omission_risk")),
        "created_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ──────────────────────────────────────────────────────────
# 업로드
# ──────────────────────────────────────────────────────────

def _upsert_batch(rows: list[dict], dry_run: bool) -> int:
    """
    article_id 기준으로 기존 PK(id) 조회 후:
    - 기존 행 → row 에 id 추가 → upsert (PK 매칭으로 UPDATE)
    - 신규 행 → id 없이 upsert (INSERT)
    article_features 에 article_id unique constraint 가 없어도 동작.
    """
    if not rows:
        return 0
    if dry_run:
        for r in rows:
            print(f"  [dry_run] {json.dumps(r, ensure_ascii=False)}")
        return len(rows)

    article_ids = [r["article_id"] for r in rows if r.get("article_id") is not None]

    # 기존 행의 PK(id) 조회
    existing_resp = (
        supabase2.table("article_features")
        .select("id, article_id")
        .in_("article_id", article_ids)
        .execute()
    )
    existing_pk: dict = {row["article_id"]: row["id"] for row in (existing_resp.data or [])}

    # 기존 행에 PK 삽입 → upsert 가 PK 기준으로 UPDATE 처리
    upsert_rows = []
    for r in rows:
        row = dict(r)
        pk = existing_pk.get(row.get("article_id"))
        if pk is not None:
            row["id"] = pk   # PK 명시 → upsert 가 UPDATE 로 처리
        upsert_rows.append(row)

    resp = supabase2.table("article_features").upsert(upsert_rows).execute()
    return len(resp.data or [])


# ──────────────────────────────────────────────────────────
# comments 업로드
# ──────────────────────────────────────────────────────────

def upload_comments(
    label_dir: str,
    keyword: Optional[str],
    batch_size: int,
    dry_run: bool,
):
    """
    *_comments.json 파일에서 댓글별 cmt_emotion / cmt_words 를
    Supabase2 comments 테이블에 upsert (id 기준).

    comments.json 의 각 댓글에 'id' 키가 있어야 DB row 특정 가능.
    Supabase2 source (--source supabase2) 로 라벨링한 경우 자동으로 포함됨.
    """
    label_dir = os.path.abspath(label_dir)
    all_files = sorted(
        f for f in os.listdir(label_dir) if f.endswith("_comments.json")
    )
    target_files = (
        [f for f in all_files if f.startswith(f"{keyword}_")]
        if keyword else all_files
    )

    if not target_files:
        print(f"[경고] 해당하는 comments.json 없음 (keyword={keyword!r})")
        return

    print(f"[댓글 업로드] {len(target_files)}개 파일  dry_run={dry_run}")

    batch: list[dict] = []
    total_ok = total_skip = total_no_id = 0

    for fname in target_files:
        fpath = os.path.join(label_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"  [스킵] 파일 읽기 실패 — {fname}: {e}")
            total_skip += 1
            continue

        for cmt in (d.get("comments") or []):
            cmt_id = cmt.get("id")
            if not cmt_id:
                total_no_id += 1
                continue  # ai_test 소스는 DB id 없으므로 건너뜀

            emotion_info = cmt.get("cmt_emotion") or {}
            probs = emotion_info.get("probs") or {}
            top10 = sorted(
                ((k, v) for k, v in probs.items() if isinstance(v, (int, float))),
                key=lambda x: x[1], reverse=True
            )[:10]
            cmt_emotion = {k: round(v, 4) for k, v in top10}

            row = {
                "id":          cmt_id,
                "cmt_emotion": cmt_emotion,   # probs >= 0.3 감정만 JSON으로
                "cmt_words":   cmt.get("cmt_words") or [],
            }
            batch.append(row)

            if len(batch) >= batch_size:
                if dry_run:
                    for r in batch:
                        print(f"  [dry_run] {json.dumps(r, ensure_ascii=False)}")
                    total_ok += len(batch)
                else:
                    resp = (
                        supabase2.table("comments")
                        .upsert(batch, on_conflict="id")
                        .execute()
                    )
                    total_ok += len(resp.data or [])
                print(f"  → {total_ok}건 업서트 (누계)")
                batch = []

    if batch:
        if dry_run:
            for r in batch:
                print(f"  [dry_run] {json.dumps(r, ensure_ascii=False)}")
            total_ok += len(batch)
        else:
            resp = (
                supabase2.table("comments")
                .upsert(batch, on_conflict="id")
                .execute()
            )
            total_ok += len(resp.data or [])
        print(f"  → {total_ok}건 업서트 (누계)")

    if total_no_id:
        print(f"  [참고] id 없는 댓글 {total_no_id}개 건너뜀 (ai_test 소스는 DB id 없음)")
    print(f"[완료] 댓글 성공 {total_ok}건 / 파일 스킵 {total_skip}건")


# ──────────────────────────────────────────────────────────
# articles 업로드
# ──────────────────────────────────────────────────────────

def run_upload(
    label_dir: str,
    keyword: Optional[str],
    batch_size: int,
    dry_run: bool,
):
    label_dir = os.path.abspath(label_dir)
    if not os.path.isdir(label_dir):
        print(f"[오류] 디렉토리 없음: {label_dir}")
        sys.exit(1)

    # ── 파일 수집 ────────────────────────────────────────
    all_files = sorted(
        f for f in os.listdir(label_dir) if f.endswith("_labeled.json")
    )
    if keyword:
        target_files = [f for f in all_files if f.startswith(f"{keyword}_")]
    else:
        target_files = all_files

    if not target_files:
        print(f"[경고] 해당하는 labeled.json 없음  (keyword={keyword!r}, dir={label_dir})")
        return

    print(f"[업로드] {len(target_files)}개 파일  dry_run={dry_run}")

    # ── 변환 및 업로드 ───────────────────────────────────
    batch: list[dict] = []
    total_ok = total_skip = total_warn = 0

    for fname in target_files:
        fpath = os.path.join(label_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"  [스킵] 파일 읽기 실패 — {fname}: {e}")
            total_skip += 1
            continue

        row = labeled_to_feature_row(d)
        if not row["article_id"]:
            print(f"  [스킵] article_id 없음 — {fname}")
            total_skip += 1
            continue

        # 매핑 실패 경고
        if d.get("frame") and row["frame_id"] is None:
            print(f"  [경고] frame 매핑 실패: {d['frame']!r} — {fname}")
            total_warn += 1
        if d.get("logic") and row["logic_id"] is None:
            print(f"  [경고] logic 매핑 실패: {d['logic']!r} — {fname}")
            total_warn += 1

        batch.append(row)
        if len(batch) >= batch_size:
            ok = _upsert_batch(batch, dry_run)
            total_ok += ok
            print(f"  → {ok}건 업서트 완료 (누계: {total_ok}건)")
            batch = []

    # 나머지 flush
    if batch:
        ok = _upsert_batch(batch, dry_run)
        total_ok += ok
        print(f"  → {ok}건 업서트 완료 (누계: {total_ok}건)")

    print(
        f"\n[완료] 성공 {total_ok}건 / 스킵 {total_skip}건"
        + (f" / 매핑 경고 {total_warn}건" if total_warn else "")
    )


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="labeled/comments.json → Supabase2 업로드"
    )
    ap.add_argument(
        "--mode", choices=["articles", "comments", "all"], default="all",
        help="업로드 대상 (기본: all)\n"
             "  articles → article_features 테이블\n"
             "  comments → comments 테이블 (cmt_emotion, cmt_words)\n"
             "  all      → 둘 다",
    )
    ap.add_argument(
        "--label_dir", default="./label_results",
        help="결과 디렉토리 (기본: ./label_results)",
    )

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--keyword", help="파일명 prefix 매칭 (예: 방탄광화문)")
    grp.add_argument("--all",     action="store_true", help="디렉토리 내 전체 파일")

    ap.add_argument("--batch_size", type=int, default=50,
                    help="1회 upsert 배치 크기 (기본: 50)")
    ap.add_argument("--dry_run", action="store_true",
                    help="실제 업로드 없이 변환 결과만 출력")
    args = ap.parse_args()

    kw = args.keyword if not args.all else None

    if args.mode in ("articles", "all"):
        run_upload(
            label_dir  = args.label_dir,
            keyword    = kw,
            batch_size = args.batch_size,
            dry_run    = args.dry_run,
        )

    if args.mode in ("comments", "all"):
        upload_comments(
            label_dir  = args.label_dir,
            keyword    = kw,
            batch_size = args.batch_size,
            dry_run    = args.dry_run,
        )
