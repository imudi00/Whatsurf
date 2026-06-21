# labeling/experiments/utils/report.py
"""
실험 결과 저장 / 비교 출력

사용:
    from labeling.experiments.utils.report import ExperimentResult, save_result, compare_results
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────
# 결과 컨테이너
# ──────────────────────────────────────────────

@dataclass
class ExperimentResult:
    name:        str                          # 실험 이름 (config 식별자)
    feature:     str                          # 평가 피처 ("body_depth", "frame", ...)
    config:      dict[str, Any]               # 실험 설정 파라미터
    metrics:     dict[str, float | str]       # 평가 결과 (메트릭 이름 → 값)
    n_samples:   int                          # 평가 샘플 수
    notes:       str = ""
    timestamp:   str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    predictions: list[dict] = field(default_factory=list)  # 개별 예측 (선택)

    def summary(self) -> str:
        """한 줄 요약"""
        m_str = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in self.metrics.items())
        return f"[{self.name}] n={self.n_samples}  {m_str}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("predictions", None)  # 저장 시 예측값은 별도 파일
        return d


# ──────────────────────────────────────────────
# 저장 / 로드
# ──────────────────────────────────────────────

def save_result(result: ExperimentResult, out_dir: str) -> str:
    """
    결과를 JSON으로 저장.
    Returns: 저장 경로
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{result.feature}_{result.name}_{result.timestamp}.json"
    path  = os.path.join(out_dir, fname)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    # 예측값이 있으면 별도 파일
    if result.predictions:
        pred_path = path.replace(".json", "_predictions.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for p in result.predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"  → 결과 저장: {path}")
    return path


def load_results(out_dir: str, feature: str | None = None) -> list[ExperimentResult]:
    """
    results/ 디렉토리에서 JSON 파일 로드.
    feature 지정 시 해당 피처만 필터.
    """
    results = []
    for path in sorted(Path(out_dir).glob("*.json")):
        if "_predictions" in path.name:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if feature and d.get("feature") != feature:
                continue
            results.append(ExperimentResult(
                name=d["name"], feature=d["feature"], config=d["config"],
                metrics=d["metrics"], n_samples=d["n_samples"],
                notes=d.get("notes",""), timestamp=d.get("timestamp",""),
            ))
        except Exception as e:
            print(f"  [경고] {path.name} 로드 실패: {e}")
    return results


# ──────────────────────────────────────────────
# 비교 출력
# ──────────────────────────────────────────────

def compare_results(results: list[ExperimentResult], sort_by: str | None = None) -> str:
    """
    여러 실험 결과를 표 형태로 비교 출력.

    Args:
        results : ExperimentResult 리스트
        sort_by : 정렬 기준 메트릭 이름 (None이면 timestamp 순)
    """
    if not results:
        return "(결과 없음)"

    if sort_by:
        results = sorted(results,
                         key=lambda r: r.metrics.get(sort_by, float("-inf")),
                         reverse=True)

    # 모든 메트릭 키 수집
    all_metric_keys: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r.metrics:
            if k not in seen:
                all_metric_keys.append(k)
                seen.add(k)

    # 컬럼 폭 계산
    name_w   = max(len(r.name) for r in results) + 2
    metric_w = 10

    header = "NAME".ljust(name_w) + "N".rjust(6) + "".join(
        k.rjust(metric_w) for k in all_metric_keys
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in results:
        row = r.name.ljust(name_w) + str(r.n_samples).rjust(6)
        for k in all_metric_keys:
            v = r.metrics.get(k, "")
            if isinstance(v, float):
                cell = f"{v:.4f}"
            else:
                cell = str(v)
            row += cell.rjust(metric_w)
        lines.append(row)

    lines.append(sep)
    if sort_by:
        lines.append(f"  ↑ {sort_by} 기준 내림차순 정렬")
    return "\n".join(lines)


def print_compare(results: list[ExperimentResult], sort_by: str | None = None):
    """compare_results를 출력"""
    print(compare_results(results, sort_by))


# ──────────────────────────────────────────────
# 베스트 결과 선택
# ──────────────────────────────────────────────

def best_result(results: list[ExperimentResult], metric: str) -> ExperimentResult | None:
    """특정 메트릭 기준 최고 결과 반환"""
    valid = [r for r in results if metric in r.metrics and
             isinstance(r.metrics[metric], (int, float))]
    if not valid:
        return None
    return max(valid, key=lambda r: r.metrics[metric])
