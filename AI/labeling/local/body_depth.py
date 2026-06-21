"""
body_depth.py — 본문 정보 깊이 점수 계산
수식 기반 (모델 미사용). WEIGHTS / 정규화 상수로 튜닝 가능.

점수 구성 (0.0 ~ 1.0):
    length_score    : 본문 길이     (로그 정규화)
    diversity_score : 어휘 다양성   (어절 TTR)
    quote_score     : 인용문 밀도   (한국어 따옴표 포함)
    numeric_score   : 수치/통계 밀도 (숫자+단위 패턴)
    structure_score : 문단 구조     (줄바꿈 기준 문단 수)
"""
import math
import re 
from typing import Optional

# ──────────────────────────────────────────────
# 튜닝 파라미터 — 이 값만 수정하면 됨
# ──────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "length":    0.20,   # 본문 길이가 정보량의 기저
    "diversity": 0.25,   # 새로운 어휘 = 새로운 정보 신호
    "quotes":    0.20,   # 인용/출처 = 검증된 정보 포함
    "numerics":  0.20,   # 수치/통계 = 구체적 사실 포함
    "structure": 0.15,   # 문단 다양성 = 주제 다양성
}

# 정규화 상수: 이 값일 때 해당 점수 ≈ 1.0
LENGTH_SOFT_MAX      = 2000   # 자수(char). 2000자 이상이면 포화
DIVERSITY_TTR_REF    = 0.55   # 어절 기준 type-token ratio 기준값
QUOTE_RATIO_REF      = 0.12   # 전체 자수 대비 인용 문자 비율
NUMERIC_DENSITY_REF  = 0.06   # 전체 어절 대비 수치 패턴 비율
STRUCTURE_PARA_MAX   = 8      # 이 문단 수 이상이면 structure_score = 1.0

# ──────────────────────────────────────────────
# 정규식 패턴
# ──────────────────────────────────────────────

# 한국어에서 사용되는 다양한 인용 부호
_QUOTE_RE = re.compile(
    r'["\u201c\u201d\u300c\u300d\u300e\u300f\u2018\u2019]'
    r'[^"\u201c\u201d\u300c\u300d\u300e\u300f\u2018\u2019]{3,}'
    r'["\u201c\u201d\u300c\u300d\u300e\u300f\u2018\u2019]'
)

# 숫자+단위 / 퍼센트 / 날짜 / 금액 등
_NUMERIC_RE = re.compile(
    r'\d[\d,]*(?:\.\d+)?'
    r'(?:\s*(?:%|억|만|천|원|개|명|건|회|배|㎞|km|m|㎡|℃|년|월|일|분|초|위))?'
)


# ──────────────────────────────────────────────
# 개별 점수 계산
# ──────────────────────────────────────────────

def _length_score(text: str) -> float:
    """본문 길이 → 로그 정규화 [0, 1]"""
    n = len(text)
    if n == 0:
        return 0.0
    return min(math.log(n + 1) / math.log(LENGTH_SOFT_MAX + 1), 1.0)


def _diversity_score(words: list[str]) -> float:
    """어절 type-token ratio → 정규화 [0, 1]"""
    n = len(words)
    if n == 0:
        return 0.0
    ttr = len(set(words)) / n
    return min(ttr / DIVERSITY_TTR_REF, 1.0)


def _quote_score(text: str) -> float:
    """인용 문자 비율 → [0, 1]"""
    if not text:
        return 0.0
    quoted_chars = sum(len(m.group()) for m in _QUOTE_RE.finditer(text))
    ratio = quoted_chars / max(len(text), 1)
    return min(ratio / QUOTE_RATIO_REF, 1.0)


def _numeric_score(words: list[str], text: str) -> float:
    """수치 패턴 밀도 → [0, 1]"""
    n_words = max(len(words), 1)
    n_nums  = len(_NUMERIC_RE.findall(text))
    return min((n_nums / n_words) / NUMERIC_DENSITY_REF, 1.0)


def _structure_score(text: str) -> float:
    """비어있지 않은 줄 수 → 문단 다양성 [0, 1]"""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return min(len(lines) / STRUCTURE_PARA_MAX, 1.0)


# ──────────────────────────────────────────────
# 메인 함수
# ──────────────────────────────────────────────

def compute_body_depth(text: str) -> float:
    """
    본문 정보 깊이 점수 계산 (0.0 ~ 1.0, 모델 미사용).

    Args:
        text: 기사 제목 + 본문 (결합 텍스트). 빈 문자열이면 0.0 반환.

    Returns:
        float: body_depth 점수 (소수점 4자리)

    세부 점수 및 가중치:
        length    × 0.20  (로그 정규화 길이)
        diversity × 0.25  (어절 TTR)
        quotes    × 0.20  (인용문 밀도)
        numerics  × 0.20  (수치/단위 패턴 밀도)
        structure × 0.15  (문단 구조)
    """
    if not text or not text.strip():
        return 0.0

    words = text.split()

    l = _length_score(text)
    d = _diversity_score(words)
    q = _quote_score(text)
    n = _numeric_score(words, text)
    s = _structure_score(text)

    score = (
        WEIGHTS["length"]    * l
        + WEIGHTS["diversity"] * d
        + WEIGHTS["quotes"]    * q
        + WEIGHTS["numerics"]  * n
        + WEIGHTS["structure"] * s
    )
    return round(score, 4)


def describe_body_depth(score: float) -> str:
    """점수 → 레이블 (high / medium / low)"""
    if score >= 0.65:
        return "high"
    elif score >= 0.35:
        return "medium"
    else:
        return "low"


def get_body_depth_detail(text: str) -> dict:
    """
    디버깅/분석용: 세부 점수 반환.

    Returns:
        {
            "body_depth": float,
            "label": str,
            "detail": {
                "length": float, "diversity": float,
                "quotes": float, "numerics": float, "structure": float
            }
        }
    """
    if not text or not text.strip():
        return {"body_depth": 0.0, "label": "low",
                "detail": {k: 0.0 for k in WEIGHTS}}

    words = text.split()
    detail = {
        "length":    round(_length_score(text), 4),
        "diversity": round(_diversity_score(words), 4),
        "quotes":    round(_quote_score(text), 4),
        "numerics":  round(_numeric_score(words, text), 4),
        "structure": round(_structure_score(text), 4),
    }
    score = sum(WEIGHTS[k] * v for k, v in detail.items())
    score = round(score, 4)
    return {"body_depth": score, "label": describe_body_depth(score), "detail": detail}
