"""
Feature Map Step
───────────────────────────────────────────────────────────────
AI/labeling/run_labeling.py 로 피처 라벨링 후
AI/labeling/upload_to_db.py 로 Supabase2 article_features / comments 에 자동 업로드.

실행 흐름:
    1. run_labeling.py  → label_results/{query_id}_{aid}_labeled.json
                          label_results/{query_id}_{aid}_comments.json
    2. upload_to_db.py  → article_features upsert
                          comments (cmt_emotion, cmt_words) upsert

반환:
    {
        "exit_code":        int,   # 라벨링 exit code
        "success":          bool,
        "features":         list[str],
        "upload_articles":  bool,  # article_features 업로드 성공 여부
        "upload_comments":  bool,  # comments 업로드 성공 여부
    }
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_RUN_LABELING = Path(__file__).resolve().parents[2] / "labeling" / "run_labeling.py"
_UPLOAD_TO_DB = Path(__file__).resolve().parents[2] / "labeling" / "upload_to_db.py"


def run_feature_map_step(
    query_id: int | str,
    features: list[str] | None = None,
    *,
    limit: int = 0,
    resume: bool = True,
    skip_comments: bool = False,
    out_dir: str = "./label_results",
    extra_args: list[str] | None = None,
    upload: bool = True,
    upload_batch_size: int = 50,
) -> dict:
    """
    Args:
        query_id:          queries 테이블 PK
        features:          라벨링할 피처 목록 (None이면 all)
        limit:             최대 기사 수 (0=전체)
        resume:            중단된 라벨링 재개 여부
        skip_comments:     댓글 라벨링 건너뜀 여부
        out_dir:           라벨 JSON 저장 디렉토리
        extra_args:        추가 CLI 인자 (e.g. ["--no_parallel"])
        upload:            라벨링 완료 후 DB 업로드 여부 (기본: True)
        upload_batch_size: upload_to_db.py 배치 크기 (기본: 50)

    Returns: 실행 결과 dict
    """
    # ── 1. 라벨링 ─────────────────────────────────────────
    cmd = [
        sys.executable,
        str(_RUN_LABELING),
        "--source",   "supabase2",
        "--query_id", str(query_id),
        "--out_dir",  out_dir,
    ]

    if features:
        cmd += ["--features"] + features

    if limit > 0:
        cmd += ["--limit", str(limit)]

    if resume:
        cmd.append("--resume")

    if skip_comments:
        cmd.append("--skip_comments")

    if extra_args:
        cmd.extend(extra_args)

    print(f"  [FeatureMap] 실행: {' '.join(cmd)}")
    print(f"  [FeatureMap] 피처: {features or ['all']}")

    label_result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
    )

    labeling_ok = label_result.returncode == 0
    print(f"  [FeatureMap] {'✓ 라벨링 완료' if labeling_ok else f'✕ 라벨링 실패 (exit {label_result.returncode})'}")

    upload_articles_ok = False
    upload_comments_ok = False

    # ── 2. DB 업로드 ──────────────────────────────────────
    # 라벨링이 실패해도 부분 완료분은 업로드 시도
    if upload:
        # run_label = query_id (정수) → 파일 prefix
        keyword = str(query_id)

        # 2-a. article_features 업로드
        print(f"\n  [FeatureMap] DB 업로드 시작 (keyword={keyword!r})...")
        upload_article_cmd = [
            sys.executable,
            str(_UPLOAD_TO_DB),
            "--mode",       "articles",
            "--keyword",    keyword,
            "--label_dir",  out_dir,
            "--batch_size", str(upload_batch_size),
        ]
        print(f"  [FeatureMap/Upload] {' '.join(upload_article_cmd)}")
        res_a = subprocess.run(upload_article_cmd, capture_output=False, text=True)
        upload_articles_ok = res_a.returncode == 0
        print(f"  [FeatureMap/Upload] article_features: "
              f"{'✓ 완료' if upload_articles_ok else f'✕ 실패 (exit {res_a.returncode})'}")

        # 2-b. comments 업로드 (skip_comments 옵션이면 건너뜀)
        if not skip_comments:
            upload_comment_cmd = [
                sys.executable,
                str(_UPLOAD_TO_DB),
                "--mode",       "comments",
                "--keyword",    keyword,
                "--label_dir",  out_dir,
                "--batch_size", str(upload_batch_size),
            ]
            print(f"  [FeatureMap/Upload] {' '.join(upload_comment_cmd)}")
            res_c = subprocess.run(upload_comment_cmd, capture_output=False, text=True)
            upload_comments_ok = res_c.returncode == 0
            print(f"  [FeatureMap/Upload] comments: "
                  f"{'✓ 완료' if upload_comments_ok else f'✕ 실패 (exit {res_c.returncode})'}")
        else:
            print("  [FeatureMap/Upload] 댓글 업로드 건너뜀 (--skip_comments)")

    return {
        "exit_code":       label_result.returncode,
        "success":         labeling_ok,
        "features":        features or ["all"],
        "upload_articles": upload_articles_ok,
        "upload_comments": upload_comments_ok,
    }
