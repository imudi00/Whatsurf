# labeling/groq/frame_labeler.py
"""
frame / logic 자동 라벨링 — Groq Llama 3.3 70B

배치 크기에 따라 기사 내용을 자동 압축:
  ≤5개  : 리드 200자, 판단어 8개, 인용구 3개
  6~15개: 리드 120자, 판단어 5개, 인용구 2개
  16+개 : 리드  70자, 판단어 3개, 인용구 1개
"""
from .groq_client import call_groq_json

FRAME_LABELS = ["인과", "대립", "개인", "경제적가치", "도덕", "안보", "권리"]
LOGIC_LABELS = ["정책적 비난", "전문가 견해", "피해자 서사", "파급효과", "해결책 제시", "사실/정보 전달"]

_FEW_SHOT = """예시1) 헤드라인: 의대 정원 확대, 의사협회 강력 반발 → frame: 대립, logic: 정책적 비난
예시2) 헤드라인: GDP 0.3% 하락 전망, 전문가들 경고 → frame: 경제적가치, logic: 전문가 견해
예시3) 헤드라인: 성범죄 피해자 2차 가해 근절 촉구 → frame: 권리, logic: 피해자 서사"""


def _truncate_params(n: int) -> dict:
    """배치 크기 n에 따라 압축 파라미터 반환"""
    if n <= 5:
        return {"lead": 200, "judgment": 8, "quotes": 3}
    elif n <= 15:
        return {"lead": 120, "judgment": 5, "quotes": 2}
    else:
        return {"lead": 70,  "judgment": 3, "quotes": 1}


def _fmt_list(lst, max_items: int) -> str:
    if not lst:
        return "없음"
    items = lst[:max_items] if isinstance(lst, list) else [str(lst)]
    return ", ".join(str(x) for x in items)


def label_frame_logic_batch(structs: list[dict]) -> list[dict]:
    """
    기사 구조체 N개를 Groq 1회 호출로 frame/logic 라벨링.
    배치 크기에 따라 기사 내용 자동 압축.

    Args:
        structs: build_article_struct() 결과 리스트

    Returns:
        [{"frame": "...", "frame_reason": "...", "logic": "...", "logic_reason": "..."}, ...]
    """
    p = _truncate_params(len(structs))
    items = []
    for i, s in enumerate(structs):
        lead = (s.get("lead") or "")[:p["lead"]]
        items.append(
            f"[{i}] {s.get('headline','')}\n"
            f"리드: {lead}\n"
            f"판단어: {_fmt_list(s.get('judgment_words'), p['judgment'])}\n"
            f"인용: {_fmt_list(s.get('quotes'), p['quotes'])}"
        )

    prompt = (
        f"아래 {len(structs)}개 기사를 각각 frame/logic으로 분류하세요.\n\n"
        f"frame(7종): 인과/대립/개인/경제적가치/도덕/안보/권리\n"
        f"logic(6종): 정책적 비난/전문가 견해/피해자 서사/파급효과/해결책 제시/사실/정보 전달\n\n"
        f"예시:\n{_FEW_SHOT}\n\n"
        + "\n\n".join(items)
        + "\n\nJSON 배열만 출력:\n"
        "[{\"idx\":0,\"frame\":\"...\",\"frame_reason\":\"한줄\",\"logic\":\"...\",\"logic_reason\":\"한줄\"},...]"
    )

    results = call_groq_json(prompt)
    if isinstance(results, dict):          # 단일 객체로 반환된 경우
        results = [results]
    results.sort(key=lambda r: r.get("idx", 0))

    out = []
    for r in results:
        if r.get("frame") not in FRAME_LABELS:
            r["frame"] = "인과"
        if r.get("logic") not in LOGIC_LABELS:
            r["logic"] = "사실/정보 전달"
        out.append(r)

    # 결과 수가 맞지 않으면 부족분 채움
    while len(out) < len(structs):
        out.append({"idx": len(out), "frame": "인과", "frame_reason": "", "logic": "사실/정보 전달", "logic_reason": ""})

    return out[:len(structs)]


def label_frame_logic(struct: dict) -> dict:
    return label_frame_logic_batch([struct])[0]
