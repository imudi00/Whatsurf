# stance_combined.py
"""
frame / logic / stance 를 LLM 호출 1회로 한번에 분석
Free Tier RPM 절약용 (기사당 3회 → 1회)
"""
from .llm_client import call_llm, parse_json_response
from .frame import FRAME_LABELS
from .logic import LOGIC_LABELS

_COMBINED_DEFINITIONS = """
## 프레임 (7종 중 하나)
- 인과: 원인 규명, 책임 귀속
- 대립: 집단 간 갈등/충돌 강조
- 개인: 특정 인물의 감정/사례 중심
- 경제적가치: 비용/수익/시장 영향
- 도덕: 윤리적/종교적 판단
- 안보: 국가안보/생존권 위협
- 권리: 인권/젠더/소수자 보호

## 논거 유형 (6종 중 하나)
- 정책적 비난: 정부/기관 정책을 직접 공격
- 전문가 견해: 전문가/연구 데이터로 주장 뒷받침
- 피해자 서사: 피해 당사자의 경험/감정 중심
- 파급효과: 사건이 미치는 사회/경제적 영향 분석
- 해결책 제시: 대안/정책 제안 중심
- 사실/정보 전달: 판단 없이 팩트 나열

## 논조 점수
- +1.0: 사안을 전적으로 옹호/긍정
- +0.5: 주로 우호적, 일부 비판 포함
-  0.0: 중립 또는 양쪽 균형
- -0.5: 주로 비판적, 일부 긍정 포함
- -1.0: 사안을 강하게 비난/반대
"""


def _build_combined_prompt(struct: dict) -> str:
    return f"""뉴스 기사를 분석해 아래 세 가지를 한번에 출력하세요.

{_COMBINED_DEFINITIONS}

## 분석 대상
헤드라인: {struct['headline']}
리드: {struct['lead']}
인용구: {struct['quotes']}
수치: {struct['numbers']}
판단어: {struct['judgment_words']}
출처: {struct['sources']}
문장 샘플: {struct['sampled_sentences']}

반드시 아래 JSON만 출력 (다른 텍스트 금지):
{{
  "frame": "...",
  "frame_reason": "한 줄 근거",
  "logic": "...",
  "logic_reason": "한 줄 근거",
  "stance_score": 0.0,
  "dominant_tone": "비판적/우호적/중립",
  "key_evidence": "근거 한 줄"
}}"""


def extract_stance_all(struct: dict) -> dict:
    """
    frame / logic / stance 를 LLM 1회 호출로 한번에 분석

    Returns:
        dict: frame, frame_reason, logic, logic_reason,
              stance_score, dominant_tone, key_evidence
    """
    prompt = _build_combined_prompt(struct)
    raw = call_llm(prompt)
    result = parse_json_response(raw)

    # 유효하지 않은 레이블 fallback
    if result.get("frame") not in FRAME_LABELS:
        result["frame"] = "인과"
    if result.get("logic") not in LOGIC_LABELS:
        result["logic"] = "사실/정보 전달"

    result["stance_score"] = max(-1.0, min(1.0, float(result.get("stance_score", 0.0))))
    return result
