# frame.py
"""
뉴스 기사 주제 접근 프레임 분류 (7종)
"""
from .llm_client import call_llm, parse_json_response

# ──────────────────────────────────────────────
# 레이블 & 퓨샷
# ──────────────────────────────────────────────

FRAME_LABELS = ["인과", "대립", "개인", "경제적가치", "도덕", "안보", "권리"]

_FEW_SHOT = """
예시1)
헤드라인: 의대 정원 확대, 의사협회 강력 반발
리드: 정부의 의대 증원 방침에 의사협회가 총파업을 예고했다.
판단어: 반발, 규탄, 촉구
→ frame: 대립

예시2)
헤드라인: 전기요금 8% 인상, 중소기업 원가 부담 우려
리드: 산업부가 전기요금 인상안을 발표하며 산업계 파장이 예상된다.
수치: 8%, 3조원
→ frame: 경제적가치

예시3)
헤드라인: 성범죄 피해자 2차 가해 근절 촉구
리드: 피해자 단체가 가해자 신상공개 확대를 요구했다.
판단어: 촉구, 요구, 규탄
→ frame: 권리
"""

_FRAME_DEFINITIONS = """
- 인과: 원인 규명, 책임 귀속
- 대립: 집단 간 갈등/충돌 강조
- 개인: 특정 인물의 감정/사례 중심
- 경제적가치: 비용/수익/시장 영향
- 도덕: 윤리적/종교적 판단
- 안보: 국가안보/생존권 위협
- 권리: 인권/젠더/소수자 보호
"""


# ──────────────────────────────────────────────
# 프롬프트 빌더
# ──────────────────────────────────────────────

def _build_frame_prompt(struct: dict) -> str:
    return f"""뉴스 기사의 주제 접근 프레임을 분류하세요.

## 프레임 정의
{_FRAME_DEFINITIONS}

## 판단 근거 (중요도 순)
1. 헤드라인과 리드의 핵심 주제
2. 판단/감정 동사의 방향
3. 인용된 주체(정부/피해자/전문가 등)

{_FEW_SHOT}

## 분석 대상
헤드라인: {struct['headline']}
리드: {struct['lead']}
인용구: {struct['quotes']}
판단어: {struct['judgment_words']}
출처: {struct['sources']}

반드시 다음 중 하나만 JSON으로 출력:
{{"frame": "...", "reason": "한 줄 근거"}}"""


# ──────────────────────────────────────────────
# 분류 함수
# ──────────────────────────────────────────────

def extract_frame(struct: dict) -> dict:
    """기사 구조체에서 프레임 레이블과 근거를 반환"""
    prompt = _build_frame_prompt(struct)
    raw = call_llm(prompt)
    result = parse_json_response(raw)

    if result.get("frame") not in FRAME_LABELS:
        result["frame"] = "인과"  # 기본값 fallback
    return result


def get_frame_label(struct: dict) -> str:
    """프레임 레이블만 간단히 반환"""
    return extract_frame(struct)["frame"]
