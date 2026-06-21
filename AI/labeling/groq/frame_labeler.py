# labeling/groq/frame_labeler.py
"""
frame / logic 자동 라벨링 — Groq Llama 3.3 70B

배치 크기에 따라 기사 내용을 자동 압축:
  ≤5개  : 리드 200자, 판단어 8개, 인용구 3개
  6~15개: 리드 120자, 판단어 5개, 인용구 2개
  16+개 : 리드  70자, 판단어 3개, 인용구 1개
"""
from difflib import SequenceMatcher
from .groq_client import call_groq_json

FRAME_LABELS = [
    "사건 원인 집중",
    "갈등/대립 강조",
    "개인 사례 중심",
    "경제적 영향 강조",
    "윤리/도덕 판단",
    "안전/안보 위협",
    "권리/인권 강조",
]

LOGIC_LABELS = [   
    "정책적 비난",
    "전문가 견해",
    "피해자 서사",
    "파급효과",
    "해결책 제시",
    "사실/정보 전달"]

_FEW_SHOT = """예시1) 헤드라인: 의대 정원 확대, 의사협회 강력 반발 → frame: 2, logic: 1
예시2) 헤드라인: GDP 0.3% 하락 전망, 전문가들 경고 → frame: 4, logic: 2
예시3) 헤드라인: 성범죄 피해자 2차 가해 근절 촉구 → frame: 7, logic: 3"""


_FUZZY_THRESHOLD = 0.40   # 유사도 임계값 (이 이상이면 가장 가까운 레이블로 매핑)


def _match_label(val: str, labels: list[str], field: str = "") -> str | None:
    """
    LLM 출력값을 정해진 레이블 목록과 매칭.
    1순위: exact match
    2순위: 숫자 index (1-based)
    3순위: partial match (부분문자열 포함)
    4순위: 문자 유사도 (SequenceMatcher ≥ _FUZZY_THRESHOLD)
           → 한글 오자 모델(llama-4-scout 등) 대응
    5순위: None
    """
    if not val:
        return None
    val_str = str(val).strip()

    # 1. exact
    if val_str in labels:
        return val_str

    # 2. 숫자 index
    try:
        idx = int(val_str) - 1
        if 0 <= idx < len(labels):
            return labels[idx]
    except ValueError:
        pass

    # 3. partial
    for label in labels:
        if val_str in label or label in val_str:
            return label

    # 4. 문자 유사도 (오자 보정)
    best_label = max(labels, key=lambda l: SequenceMatcher(None, val_str, l).ratio())
    best_ratio = SequenceMatcher(None, val_str, best_label).ratio()
    if best_ratio >= _FUZZY_THRESHOLD:
        print(f"  [frame_labeler] {field} 유사도 매핑 ({best_ratio:.2f}): {val_str!r} → {best_label!r}")
        return best_label

    print(f"  [frame_labeler 경고] {field} 매핑 실패 (유사도 {best_ratio:.2f}): {val_str!r}")
    return None


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

    frame_legend = "\n".join(f"  {i+1}={l}" for i, l in enumerate(FRAME_LABELS))
    logic_legend = "\n".join(f"  {i+1}={l}" for i, l in enumerate(LOGIC_LABELS))

    prompt = (
        f"아래 {len(structs)}개 기사를 각각 frame/logic 번호로 분류하세요.\n\n"
        f"frame (1~7):\n{frame_legend}\n\n"
        f"logic (1~6):\n{logic_legend}\n\n"
        f"예시:\n{_FEW_SHOT}\n\n"
        + "\n\n".join(items)
        + "\n\n반드시 frame과 logic을 정수(숫자)로만 출력하세요. JSON 배열만 출력:\n"
        "[{\"idx\":0,\"frame\":1,\"frame_reason\":\"한줄\",\"logic\":3,\"logic_reason\":\"한줄\"},...]"
    )

    results = call_groq_json(prompt)
    if isinstance(results, dict):
        results = [results]
    results.sort(key=lambda r: r.get("idx", 0))

    out = []
    for r in results:
        r["frame"] = _match_label(r.get("frame", ""), FRAME_LABELS, "frame")
        r["logic"] = _match_label(r.get("logic", ""), LOGIC_LABELS, "logic")
        out.append(r)

    # 결과 수가 맞지 않으면 부족분 채움
    while len(out) < len(structs):
        out.append({"idx": len(out), "frame": None, "frame_reason": "", "logic": None, "logic_reason": ""})

    return out[:len(structs)]


def label_frame_logic(struct: dict) -> dict:
    return label_frame_logic_batch([struct])[0]
