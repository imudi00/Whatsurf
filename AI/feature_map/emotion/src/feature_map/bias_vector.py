# bias_vector.py
"""
BiasLab Dual-Axis Bias Vector
정치적 편향을 두 축(민주/공화 친밀도, 좌-우 점수)으로 표현
"""
from dataclasses import dataclass


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class BiasVectorOutput:
    tone_dem: float   # 민주 성향 점수
    tone_rep: float   # 공화 성향 점수
    lr_score: float   # 좌(-) ↔ 우(+) 종합 점수
    rationale: str    # 판단 근거 레이블


# ──────────────────────────────────────────────
# 편향 신호 사전 (확장 가능)
# ──────────────────────────────────────────────

_DEM_SIGNALS = ["policy", "welfare", "progressive", "climate", "equity"]
_REP_SIGNALS = ["regulation", "tax cut", "conservative", "border", "security"]


# ──────────────────────────────────────────────
# 계산 함수
# ──────────────────────────────────────────────

def _count_signals(text_lower: str, signals: list) -> float:
    """신호 단어 등장 횟수 합산"""
    return sum(1.0 for s in signals if s in text_lower)


def compute_bias_vector(text: str) -> BiasVectorOutput:
    """
    텍스트에서 정치적 편향 벡터를 계산한다.

    Returns:
        BiasVectorOutput: tone_dem, tone_rep, lr_score, rationale
    """
    text_lower = text.lower()

    tone_dem = _count_signals(text_lower, _DEM_SIGNALS)
    tone_rep = _count_signals(text_lower, _REP_SIGNALS)
    lr_score = tone_rep - tone_dem

    if lr_score > 0:
        rationale = "right_leaning"
    elif lr_score < 0:
        rationale = "left_leaning"
    else:
        rationale = "neutral"

    return BiasVectorOutput(
        tone_dem=tone_dem,
        tone_rep=tone_rep,
        lr_score=lr_score,
        rationale=rationale,
    )


def normalize_bias_vector(output: BiasVectorOutput, scale: float = 5.0) -> BiasVectorOutput:
    """lr_score를 [-1, 1] 범위로 정규화"""
    normalized_lr = max(-1.0, min(1.0, output.lr_score / scale))
    return BiasVectorOutput(
        tone_dem=output.tone_dem,
        tone_rep=output.tone_rep,
        lr_score=normalized_lr,
        rationale=output.rationale,
    )
