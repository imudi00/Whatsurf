# labeling/groq/frame_logic_labeler.py
"""
Groq Llama 3.3 70B — frame / logic 자동 라벨링
few-shot 프롬프트 포함 (중간 난이도 피처)
"""
from .groq_client import call_groq_json

FRAME_LABELS = ["인과", "대립", "개인", "경제적가치", "도덕", "안보", "권리"]
LOGIC_LABELS = ["정책적 비난", "전문가 견해", "피해자 서사", "파급효과", "해결책 제시", "사실/정보 전달"]

_FEW_SHOT = """
예시1)
헤드라인: 의대 정원 확대, 의사협회 강력 반발
판단어: 반발, 규탄, 촉구 / 출처: 야당, 시민단체
→ frame: 대립, logic: 정책적 비난

예시2)
헤드라인: 전기요금 8% 인상, 중소기업 원가 부담 우려
수치: 8%, 3조원 / 출처: 경제연구원, 교수
→ frame: 경제적가치, logic: 전문가 견해

예시3)
헤드라인: 성범죄 피해자 2차 가해 근절 촉구
판단어: 촉구, 요구 / 출처: 피해자 가족
→ frame: 권리, logic: 피해자 서사
"""


def label_frame_logic_batch(items: list) -> list:
    """
    items: [{"id": ..., "headline": ..., "lead": ...,
              "quotes": [...], "numbers": [...],
              "judgment_words": [...], "sources": [...]}]
    반환:  [{"id": ..., "frame": ..., "logic": ...,
              "frame_reason": ..., "logic_reason": ...}]
    """
    blocks = "\n\n".join(
        f"--- 항목 {i} (id={it['id']}) ---\n"
        f"헤드라인: {it['headline']}\n리드: {it.get('lead','')}\n"
        f"판단어: {it.get('judgment_words','')}\n수치: {it.get('numbers','')}\n"
        f"인용구: {it.get('quotes','')}\n출처: {it.get('sources','')}"
        for i, it in enumerate(items)
    )

    prompt = f"""아래 뉴스 기사들의 frame과 logic을 분류하세요.

## 프레임 (7종): 인과/대립/개인/경제적가치/도덕/안보/권리
## 논거 유형 (6종): 정책적 비난/전문가 견해/피해자 서사/파급효과/해결책 제시/사실/정보 전달

{_FEW_SHOT}

{blocks}

JSON 배열만 출력:
[
  {{"idx": 0, "id": "...", "frame": "...", "frame_reason": "한줄", "logic": "...", "logic_reason": "한줄"}},
  ...
]"""

    results = call_groq_json(prompt)
    results.sort(key=lambda r: r.get("idx", 0))

    out = []
    for r in results:
        if r.get("frame") not in FRAME_LABELS: r["frame"] = "인과"
        if r.get("logic") not in LOGIC_LABELS: r["logic"] = "사실/정보 전달"
        out.append(r)
    return out
