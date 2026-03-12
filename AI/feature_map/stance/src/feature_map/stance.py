# stance.py
"""
뉴스 기사 논조(stance) 점수 분석 [-1.0 ~ +1.0]
frame, logic 결과를 맥락으로 활용해 정확도를 높임
"""
from .llm_client import call_llm, parse_json_response

# ──────────────────────────────────────────────
# 프롬프트 빌더
# ──────────────────────────────────────────────

def _build_stance_prompt(struct: dict, frame: str, logic: str) -> str:
    return f"""뉴스 기사의 주요 사안에 대한 논조(stance)를 분석하세요.

## 점수 기준
- +1.0: 사안을 전적으로 옹호/긍정
- +0.5: 주로 우호적, 일부 비판 포함
-  0.0: 중립 또는 양쪽 균형
- -0.5: 주로 비판적, 일부 긍정 포함
- -1.0: 사안을 강하게 비난/반대

## 이미 분석된 맥락
- 프레임: {frame}
- 논거 유형: {logic}

## 분석 근거 (우선순위)
1. 판단/감정 동사의 방향과 강도: {struct['judgment_words']}
2. 인용 주체의 포지션: {struct['quotes']}
3. 리드문의 방향성: {struct['lead'][:150]}
4. 수치가 긍정/부정 맥락인가: {struct['numbers']}

JSON만 출력:
{{"stance_score": 0.0, "dominant_tone": "비판적/우호적/중립", "key_evidence": "근거 한 줄"}}"""


# ──────────────────────────────────────────────
# 분석 함수
# ──────────────────────────────────────────────

def extract_stance(struct: dict, frame: str, logic: str) -> dict:
    """
    기사 구조체 + 프레임 + 논거 유형을 받아 논조 점수를 반환

    Returns:
        dict: stance_score, dominant_tone, key_evidence
    """
    prompt = _build_stance_prompt(struct, frame, logic)
    raw = call_llm(prompt)
    result = parse_json_response(raw)

    # stance_score 범위 클램핑
    result["stance_score"] = max(-1.0, min(1.0, float(result["stance_score"])))
    return result


def get_stance_score(struct: dict, frame: str, logic: str) -> float:
    """논조 점수만 간단히 반환"""
    return extract_stance(struct, frame, logic)["stance_score"]


def get_dominant_tone(struct: dict, frame: str, logic: str) -> str:
    """지배적 논조 레이블만 반환"""
    return extract_stance(struct, frame, logic)["dominant_tone"]
