# labeling/api_client_base.py
"""
Groq / Gemini 클라이언트 공통 유틸리티.
  - API 키 / 모델 목록 로드
  - 키별 RPD 카운터 (파일 영속)
  - retry-delay 파싱
  - LLM JSON 출력 강건 파싱
  - JSON retry 호출 래퍼
"""
import json
import os
import re
from datetime import date, datetime
from pathlib import Path


# ── API 키 / 모델 로드 ───────────────────────────────────────

def _split_env(val: str) -> list[str]:
    return [s.strip().strip("'\"") for s in val.split(",") if s.strip().strip("'\"")]


def load_api_keys(env_multi: str, env_single: str) -> list[str]:
    """콤마 구분 다중 키 또는 단일 키 로드. 따옴표 자동 제거."""
    if keys := _split_env(os.getenv(env_multi, "")):
        return keys
    if single := os.getenv(env_single, "").strip().strip("'\""):
        return [single]
    raise EnvironmentError(
        f"API 키 미설정. .env에 {env_multi}=key1,key2 (따옴표 없이) "
        f"또는 {env_single}=key 를 추가하세요."
    )


def load_models(env_multi: str, env_single: str, default: str) -> list[str]:
    """콤마 구분 다중 모델 또는 단일 모델 로드."""
    if models := _split_env(os.getenv(env_multi, "")):
        return models
    return [os.getenv(env_single, default).strip()]


# ── RPD 카운터 ───────────────────────────────────────────────

class RpdCounter:
    """키별 일일 요청 수 추적 (JSON 파일 영속)."""

    def __init__(self, counter_file: Path, rpd_limit: int):
        self.file      = counter_file
        self.rpd_limit = rpd_limit

    def _load(self) -> dict:
        if self.file.exists():
            try:
                return json.loads(self.file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save(self, data: dict):
        self.file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _remaining_from(self, entry: dict) -> int:
        if entry.get("date") != str(date.today()):
            return self.rpd_limit
        return max(0, self.rpd_limit - entry.get("count", 0))

    def remaining(self, key_idx: int) -> int:
        return self._remaining_from(self._load().get(f"key_{key_idx}", {}))

    def increment(self, key_idx: int):
        data  = self._load()
        today = str(date.today())
        k     = f"key_{key_idx}"
        entry = data.get(k, {})
        if entry.get("date") != today:
            entry = {"date": today, "count": 0}
        entry["count"] += 1
        data[k] = entry
        self._save(data)

    def max_remaining(self, n_keys: int) -> int:
        data = self._load()
        return max(
            (self._remaining_from(data.get(f"key_{i}", {})) for i in range(n_keys)),
            default=0,
        )

    def status(self, n_keys: int) -> dict:
        data = self._load()
        result = {}
        for i in range(n_keys):
            rem = self._remaining_from(data.get(f"key_{i}", {}))
            result[f"key_{i}"] = {
                "remaining": rem,
                "used":      self.rpd_limit - rem,
                "limit":     self.rpd_limit,
            }
        return result


# ── 유틸 ─────────────────────────────────────────────────────

def parse_retry_delay(msg: str, default: float = 30.0) -> float:
    """에러 메시지에서 retry-after 초를 파싱. 없으면 default 반환."""
    m = re.search(r"retry[_ ]?(?:in|delay|after)[:\s'\"]*([0-9.]+)", msg, re.IGNORECASE)
    return float(m.group(1)) + 1 if m else default


def now_iso() -> str:
    return datetime.now().isoformat()


# ── JSON 파싱 ─────────────────────────────────────────────────

def fix_json_string(text: str) -> str:
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()  # reasoning 모델
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text).strip()
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = re.sub(r',\s*([\]\}])', r'\1', text)
    text = re.sub(r'\}\s*\n\s*\{', '},\n{', text)
    return text


def robust_json_parse(raw: str):
    """
    LLM 출력에서 JSON 배열/객체를 강건하게 파싱.
    처리: markdown/think 블록, trailing comma, 전각 따옴표, NDJSON-like 응답.
    """
    text = fix_json_string(raw)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for pattern in (r'\[[\s\S]*\]', r'\{[\s\S]*\}'):
        m = re.search(pattern, text)
        if m:
            candidate = fix_json_string(m.group(0))
            try:
                result = json.loads(candidate)
                return result if isinstance(result, list) else [result]
            except json.JSONDecodeError:
                pass

    objects = []
    for line in text.splitlines():
        line = line.strip().rstrip(',')
        if line.startswith('{') and line.endswith('}'):
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if objects:
        return objects

    raise json.JSONDecodeError(
        f"JSON 파싱 전략 모두 실패. 원본 (앞 200자): {raw[:200]}", raw, 0
    )


# ── JSON retry 래퍼 ───────────────────────────────────────────

_RETRY_SUFFIX = (
    "\n\n⚠ 이전 응답이 유효한 JSON이 아니었습니다. "
    "반드시 유효한 JSON 배열만 출력하세요. 설명이나 마크다운 없이."
)


def retry_json_call(call_fn, prompt: str, *, _retry_json: int = 2, label: str = "LLM"):
    """
    call_fn(prompt) → str 을 호출하고 JSON 파싱.
    파싱 실패 시 suffix를 붙인 프롬프트로 최대 _retry_json회 재시도.
    """
    last_raw = ""
    for attempt in range(1, _retry_json + 2):
        p = prompt if attempt == 1 else prompt + _RETRY_SUFFIX
        if attempt > 1:
            print(f"  [{label} JSON] 파싱 실패 → 재시도 ({attempt - 1}/{_retry_json})")
        raw = call_fn(p)
        last_raw = raw
        try:
            return robust_json_parse(raw)
        except (json.JSONDecodeError, ValueError):
            if attempt <= _retry_json:
                continue
    raise json.JSONDecodeError(
        f"{label} JSON 파싱 {_retry_json + 1}회 시도 모두 실패.\n원본: {last_raw[:300]}",
        last_raw, 0,
    )
