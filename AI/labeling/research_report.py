# labeling/research_report.py
"""
연구용 데이터 수집 및 저장 모듈

저장 파일 (reports/ 폴더):
  run_report_{ts}.json      ← 전체 리포트 (시간/API/품질/텍스트통계)
  label_dist_{ts}.json      ← 레이블 분포 (빠른 확인용)
  feature_timing_{ts}.json  ← 피처별 소요시간 상세 (성능 분석용)
  sample_labels_{ts}.json   ← 피처별 샘플 결과 (품질 육안 확인용)
  comments_stats_{ts}.json  ← 댓글 감정/편향 분포 (댓글 분석용)
"""
import json
import time
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────
# ResearchReport
# ──────────────────────────────────────────────────────────

class ResearchReport:
    def __init__(self, keyword: str, limit: int, batch_size: int):
        self.keyword    = keyword
        self.limit      = limit
        self.batch_size = batch_size
        self.started_at = datetime.now().isoformat()
        self._run_start = time.time()

        # ── 시간 측정 ──────────────────────────────────────
        # {feature: [(elapsed_sec, n_items), ...]}
        self._timing_raw: dict[str, list[tuple[float, int]]] = defaultdict(list)

        # ── API 사용 ───────────────────────────────────────
        self.api_calls:   dict[str, int]  = defaultdict(int)
        self.api_retries: dict[str, int]  = defaultdict(int)
        self.api_errors:  list[dict]      = []
        self.key_rotations: list[dict]    = []   # Gemini 키 로테이션 이벤트

        # ── 라벨 수집 ──────────────────────────────────────
        self.labels: dict[str, list] = defaultdict(list)

        # 샘플 (피처별 처음 10건) ── 품질 육안 확인용
        self._samples: dict[str, list] = defaultdict(list)
        self._SAMPLE_LIMIT = 10

        # ── fallback 카운터 ────────────────────────────────
        self.fallbacks: dict[str, int] = defaultdict(int)

        # ── 에러 로그 (기사/댓글 단위) ─────────────────────
        self.errors: list[dict] = []

        # ── 텍스트 통계 ────────────────────────────────────
        self.text_lengths:   list[int] = []
        self.comment_counts: list[int] = []

        # ── 댓글 통계 ──────────────────────────────────────
        self._cmt_emotion_all:  list[str]   = []
        self._cmt_words_flat:   list[str]   = []   # 출현한 편향 단어 전체
        self._cmt_total:        int         = 0
        self._cmt_with_emotion: int         = 0
        self._cmt_with_words:   int         = 0

    # ── 시간 측정 ──────────────────────────────────────────

    def time_feature(self, feature: str):
        """with report.time_feature("frame")(n=5) 형태로 사용"""
        return _FeatureTimer(self, feature)

    def record_timing(self, feature: str, elapsed: float, n: int = 1):
        self._timing_raw[feature].append((round(elapsed, 4), max(n, 1)))

    # ── 라벨 수집 ──────────────────────────────────────────

    def record_labels(self, feature: str, values: list):
        self.labels[feature].extend(values)

    def record_sample(self, feature: str, article_id: str, value: Any):
        """품질 육안 확인용 샘플 (처음 _SAMPLE_LIMIT건)"""
        if len(self._samples[feature]) < self._SAMPLE_LIMIT:
            self._samples[feature].append({
                "article_id": article_id,
                "value":      value,
                "ts":         _now_iso(),
            })

    def record_fallback(self, feature: str, count: int = 1):
        self.fallbacks[feature] += count

    # ── API 사용 ───────────────────────────────────────────

    def record_api_call(self, provider: str, n: int = 1):
        self.api_calls[provider] += n

    def record_retry(self, provider: str, reason: str = ""):
        self.api_retries[provider] += 1
        self.api_errors.append({
            "provider": provider,
            "reason":   reason[:200],
            "ts":       _now_iso(),
        })

    def record_key_rotation(self, from_key: int, to_key: int, reason: str):
        """Gemini 키 로테이션 이벤트 기록"""
        self.key_rotations.append({
            "from_key": from_key,
            "to_key":   to_key,
            "reason":   reason,
            "ts":       _now_iso(),
        })

    def record_error(self, feature: str, article_id: str, error: str):
        """기사/피처 단위 에러 기록"""
        self.errors.append({
            "feature":    feature,
            "article_id": article_id,
            "error":      error[:300],
            "ts":         _now_iso(),
        })

    # ── 텍스트 통계 ────────────────────────────────────────

    def record_texts(self, texts: list[str], comment_counts: list[int] | None = None):
        self.text_lengths.extend([len(t) for t in texts])
        if comment_counts:
            self.comment_counts.extend(comment_counts)

    # ── 댓글 통계 ──────────────────────────────────────────

    def record_comment_labels(self, emotion_label: str | None,
                               cmt_words: list[str]):
        """댓글 1건의 라벨 결과 기록"""
        self._cmt_total += 1
        if emotion_label:
            self._cmt_emotion_all.append(emotion_label)
            self._cmt_with_emotion += 1
        if cmt_words:
            self._cmt_words_flat.extend(cmt_words)
            self._cmt_with_words += 1

    # ── 내부 통계 함수 ─────────────────────────────────────

    def _dist(self, values: list) -> dict:
        if not values:
            return {}
        c = Counter(str(v) for v in values)
        total = len(values)
        return {
            k: {"count": v, "pct": round(v / total * 100, 1)}
            for k, v in c.most_common()
        }

    def _stats(self, values: list[float]) -> dict:
        if not values:
            return {}
        sv = sorted(values)
        n  = len(sv)
        return {
            "mean":   round(sum(sv) / n, 4),
            "min":    round(sv[0], 4),
            "max":    round(sv[-1], 4),
            "median": round(sv[n // 2], 4),
            "p25":    round(sv[n // 4], 4),
            "p75":    round(sv[3 * n // 4], 4),
            "n":      n,
        }

    def _timing_stats(self) -> dict:
        """피처별 타이밍 통계 계산"""
        out = {}
        for feat, records in self._timing_raw.items():
            total_sec   = sum(e for e, _ in records)
            total_items = sum(n for _, n in records)
            per_item    = [e / n for e, n in records]
            out[feat] = {
                **self._stats(per_item),
                "total_sec":    round(total_sec, 3),
                "total_items":  total_items,
                "total_batches": len(records),
                "throughput_per_min": round(total_items / total_sec * 60, 1)
                    if total_sec > 0 else 0,
                "batches_detail": [
                    {"elapsed_sec": e, "n_items": n, "sec_per_item": round(e / n, 4)}
                    for e, n in records
                ],
            }
        return out

    # ── 전체 리포트 빌드 ───────────────────────────────────

    def build(self) -> dict:
        total_elapsed = round(time.time() - self._run_start, 2)
        timing        = self._timing_stats()

        NUMERIC = {"stance_score", "bias_x", "bias_y",
                   "emotion_intensity", "loaded_word_density"}
        label_dist  = {}
        label_stats = {}

        def _to_float(v):
            try:    return float(v)
            except: return None

        for feat, vals in self.labels.items():
            if feat in NUMERIC:
                label_stats[feat] = self._stats(
                    [x for x in (_to_float(v) for v in vals) if x is not None]
                )
            else:
                label_dist[feat] = self._dist(vals)

        # 편향 단어 빈도 (상위 20개)
        art_words_all = self.labels.get("art_words_flat", [])
        top_bias_words = dict(Counter(art_words_all).most_common(20))

        # 댓글 통계
        cmt_stats = {
            "total_comments":     self._cmt_total,
            "with_emotion_label": self._cmt_with_emotion,
            "with_loaded_words":  self._cmt_with_words,
            "emotion_dist":       self._dist(self._cmt_emotion_all),
            "top_cmt_bias_words": dict(Counter(self._cmt_words_flat).most_common(20)),
        }

        # 전체 throughput
        total_items = sum(
            sum(n for _, n in records)
            for records in self._timing_raw.values()
        )
        overall_throughput = round(total_items / total_elapsed * 60, 1) \
            if total_elapsed > 0 else 0

        return {
            "meta": {
                "keyword":           self.keyword,
                "limit":             self.limit,
                "batch_size":        self.batch_size,
                "started_at":        self.started_at,
                "finished_at":       _now_iso(),
                "total_elapsed_sec": total_elapsed,
                "total_articles":    len(self.text_lengths),
                "total_comments":    self._cmt_total,
            },

            "timing": {
                "overall_throughput_per_min": overall_throughput,
                "per_feature": {
                    feat: {
                        "mean_sec_per_item":   s["mean"],
                        "total_sec":           s["total_sec"],
                        "total_items":         s["total_items"],
                        "throughput_per_min":  s["throughput_per_min"],
                        "batches":             s["total_batches"],
                    }
                    for feat, s in timing.items()
                },
                "summary_table": [
                    f"{feat:20s} | {s['mean_sec_per_item']:.3f}s/건 "
                    f"| 총 {s['total_sec']:.1f}s "
                    f"| {s['throughput_per_min']:.1f}건/분"
                    for feat, s in {
                        feat: {
                            "mean_sec_per_item":  timing[feat]["mean"],
                            "total_sec":          timing[feat]["total_sec"],
                            "throughput_per_min": timing[feat]["throughput_per_min"],
                        }
                        for feat in timing
                    }.items()
                ],
            },

            "api_usage": {
                "calls":             dict(self.api_calls),
                "retries":           dict(self.api_retries),
                "total_llm_calls":   sum(self.api_calls.values()),
                "key_rotations":     self.key_rotations,
                "key_rotation_count": len(self.key_rotations),
                "errors":            self.api_errors,
            },

            "label_quality": {
                "distribution":   label_dist,
                "numeric_stats":  label_stats,
                "fallbacks":      dict(self.fallbacks),
                "fallback_total": sum(self.fallbacks.values()),
                "top_bias_words": top_bias_words,
            },

            "comment_stats": cmt_stats,

            "text_stats": {
                "article_length": self._stats(self.text_lengths),
                "comment_count":  self._stats(
                    [float(c) for c in self.comment_counts]
                ),
            },

            "errors": self.errors,
            "error_count": len(self.errors),
        }

    # ── 저장 ───────────────────────────────────────────────

    def save(self, out_dir: str) -> tuple[str, str]:
        report_dir = os.path.join(out_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)

        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = self.build()
        timing = self._timing_stats()

        def _w(name: str, obj: dict) -> str:
            p = os.path.join(report_dir, name)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            return p

        # 1. 전체 리포트
        report_path = _w(f"run_report_{ts}.json", report)

        # 2. 레이블 분포 (빠른 확인)
        dist_path = _w(f"label_dist_{ts}.json", {
            "meta":          report["meta"],
            "distribution":  report["label_quality"]["distribution"],
            "numeric_stats": report["label_quality"]["numeric_stats"],
            "top_bias_words": report["label_quality"]["top_bias_words"],
            "fallbacks":     report["label_quality"]["fallbacks"],
            "comment_stats": report["comment_stats"],
        })

        # 3. 피처별 소요시간 상세 (성능 분석용)
        _w(f"feature_timing_{ts}.json", {
            "meta": {
                "keyword": self.keyword,
                "started_at": self.started_at,
                "total_elapsed_sec": report["meta"]["total_elapsed_sec"],
            },
            "timing": timing,
            "summary_table": report["timing"]["summary_table"],
            "api_usage": report["api_usage"],
        })

        # 4. 샘플 라벨 (품질 육안 확인용)
        _w(f"sample_labels_{ts}.json", {
            "meta":    report["meta"],
            "samples": dict(self._samples),
        })

        # 5. 댓글 통계 (댓글 분석용)
        _w(f"comments_stats_{ts}.json", {
            "meta":          report["meta"],
            "comment_stats": report["comment_stats"],
        })

        return report_path, dist_path

    # ── 콘솔 요약 ──────────────────────────────────────────

    def print_summary(self):
        r = self.build()
        m = r["meta"]
        t = r["timing"]

        print(f"\n{'='*60}")
        print(f"  [리포트 요약]  keyword={m['keyword']}")
        print(f"  기사 {m['total_articles']}건  댓글 {m['total_comments']}건")
        print(f"  총 소요시간: {m['total_elapsed_sec']}s  "
              f"전체 처리량: {t['overall_throughput_per_min']}건/분")

        print(f"\n  ── 피처별 소요시간 ──────────────────────")
        for feat, s in t["per_feature"].items():
            print(f"    {feat:20s}: {s['mean_sec_per_item']:.3f}s/건 "
                  f"| 총 {s['total_sec']:.1f}s "
                  f"| {s['throughput_per_min']:.1f}건/분 "
                  f"({s['batches']}배치)")

        print(f"\n  ── API 호출 ─────────────────────────────")
        for p, n in r["api_usage"]["calls"].items():
            ret = r["api_usage"]["retries"].get(p, 0)
            print(f"    {p:15s}: {n}회 호출  {ret}회 재시도")
        if r["api_usage"]["key_rotations"]:
            print(f"    Gemini 키 로테이션: {r['api_usage']['key_rotation_count']}회")

        print(f"\n  ── 레이블 분포 (상위 3) ─────────────────")
        for feat, dist in r["label_quality"]["distribution"].items():
            top3 = list(dist.items())[:3]
            s = "  |  ".join(f"{k}: {v['pct']}%" for k, v in top3)
            print(f"    {feat:20s}: {s}")

        top_words = list(r["label_quality"]["top_bias_words"].items())[:5]
        if top_words:
            print(f"\n  ── 상위 편향 단어 ───────────────────────")
            print(f"    " + "  ".join(f"{w}({c})" for w, c in top_words))

        if r["comment_stats"]["total_comments"] > 0:
            cs = r["comment_stats"]
            print(f"\n  ── 댓글 통계 ────────────────────────────")
            print(f"    총 {cs['total_comments']}개  "
                  f"감정 라벨 {cs['with_emotion_label']}개  "
                  f"편향 단어 {cs['with_loaded_words']}개")
            top_emo = list(cs["emotion_dist"].items())[:3]
            if top_emo:
                print(f"    감정 분포: " +
                      "  ".join(f"{k}:{v['pct']}%" for k, v in top_emo))

        if r["errors"]:
            print(f"\n  ── 에러 ({r['error_count']}건) ──────────────────────")
            for e in r["errors"][:3]:
                print(f"    [{e['feature']}] id={e['article_id']}: {e['error'][:60]}")

        if r["label_quality"]["fallbacks"]:
            print(f"\n  ── Fallback 발동 ─────────────────────────")
            for feat, cnt in r["label_quality"]["fallbacks"].items():
                print(f"    {feat:20s}: {cnt}건")

        print(f"{'='*60}\n")


# ── with 블록용 타이머 ────────────────────────────────────

class _FeatureTimer:
    def __init__(self, report: ResearchReport, feature: str):
        self._report  = report
        self._feature = feature
        self._start   = 0.0
        self._n       = 1

    def __call__(self, n: int):
        self._n = n
        return self

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._start
        self._report.record_timing(self._feature, elapsed, self._n)


def _now_iso() -> str:
    return datetime.now().isoformat()
