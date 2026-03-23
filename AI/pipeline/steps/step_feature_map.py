"""
Feature Map Step
───────────────────────────────────────────────────────────────
AI/labeling/run_labeling.py 를 subprocess로 호출하여
article_features 테이블에 피처 라벨링 결과를 저장.

run_labeling.py 내부에서 Supabase2 article_features 테이블 upsert 가
이미 구현되어 있으므로, 이 스텝은 올바른 인자로 실행만 담당.

반환:
    {
        "exit_code": int,
        "success":   bool,
        "features":  list[str],
    }
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_RUN_LABELING = Path(__file__).resolve().parents[2] / "labeling" / "run_labeling.py"


def run_feature_map_step(
    query_id: int | str,
    features: list[str] | None = None,
    *,
    limit: int = 0,
    resume: bool = True,
    skip_comments: bool = False,
    out_dir: str = "./label_results",
    extra_args: list[str] | None = None,
) -> dict:
    """
    Args:
        query_id:      queries 테이블 PK
        features:      라벨링할 피처 목록 (None이면 all)
        limit:         최대 기사 수 (0=전체)
        resume:        중단된 라벨링 재개 여부
        skip_comments: 댓글 라벨링 건너뜀 여부
        out_dir:       라벨 JSON 저장 디렉토리
        extra_args:    추가 CLI 인자 (e.g. ["--no_parallel"])

    Returns: 실행 결과 dict
    """
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

    result = subprocess.run(
        cmd,
        capture_output=False,   # stdout/stderr 실시간 출력
        text=True,
    )

    success = result.returncode == 0
    status  = "✓ 완료" if success else f"✕ 실패 (exit {result.returncode})"
    print(f"  [FeatureMap] {status}")

    return {
        "exit_code": result.returncode,
        "success":   success,
        "features":  features or ["all"],
    }
