from dataclasses import dataclass

@dataclass
class BiasVectorOutput:
    tone_dem: float
    tone_rep: float
    lr_score: float
    rationale: str


def compute_bias_vector(text: str) -> BiasVectorOutput:
    """
    BiasLab Dual-Axis Bias Vector
    """

    # 현재는 테스트용 heuristic
    tone_dem = 0.0
    tone_rep = 0.0
    rationale = "neutral"

    if "policy" in text.lower():
        tone_dem += 1.0
        rationale = "policy_support"

    if "regulation" in text.lower():
        tone_rep += 1.0
        rationale = "anti_regulation"

    lr_score = tone_rep - tone_dem

    return BiasVectorOutput(
        tone_dem=tone_dem,
        tone_rep=tone_rep,
        lr_score=lr_score,
        rationale=rationale
    )