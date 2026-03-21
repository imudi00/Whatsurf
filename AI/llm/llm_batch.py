# llm/llm_batch.py
"""
뉴스/댓글 배치 프롬프트 빌더 + 호출 (Gemini)

뉴스 배치 : N개 기사 → frame / logic / stance 동시 분석 (LLM 1회)
댓글 배치 : 기사당 전체 댓글 → emotion_label / loaded_words 동시 분석 (LLM 1회)
"""
from llm_client import call_llm_json

FRAME_LABELS   = ["인과", "대립", "개인", "경제적가치", "도덕", "안보", "권리"]
LOGIC_LABELS   = ["정책적 비난", "전문가 견해", "피해자 서사", "파급효과", "해결책 제시", "사실/정보 전달"]
EMOTION_LABELS = ["anger","disgust","fear","anticipation","sadness","surprise","prediction","trust","neutral"]


# ── 뉴스 배치 ──────────────────────────────────────────────

def analyze_news_batch(structs: list) -> list:
    """뉴스 구조체 리스트 → frame/logic/stance 배치 분석"""
    items = "\n\n".join(
        f"--- 기사 {i} ---\n"
        f"헤드라인: {s['headline']}\n리드: {s['lead']}\n"
        f"인용구: {s['quotes']}\n수치: {s['numbers']}\n"
        f"판단어: {s['judgment_words']}\n출처: {s['sources']}\n"
        f"문장샘플: {s['sampled_sentences']}"
        for i, s in enumerate(structs)
    )
    prompt = f"""아래 {len(structs)}개 뉴스 기사를 각각 분석하세요.

## 프레임 (7종): 인과/대립/개인/경제적가치/도덕/안보/권리
## 논거 유형 (6종): 정책적 비난/전문가 견해/피해자 서사/파급효과/해결책 제시/사실/정보 전달
## 논조 점수: -1.0(강비판) ~ 0.0(중립) ~ +1.0(강옹호)

{items}

JSON 배열만 출력 (다른 텍스트 금지):
[
  {{"idx":0,"frame":"...","frame_reason":"한줄","logic":"...","logic_reason":"한줄","stance_score":0.0,"dominant_tone":"비판적/우호적/중립","key_evidence":"한줄"}},
  ...
]"""
    results = call_llm_json(prompt)
    results.sort(key=lambda r: r.get("idx", 0))
    out = []
    for r in results:
        if r.get("frame") not in FRAME_LABELS:   r["frame"] = "인과"
        if r.get("logic") not in LOGIC_LABELS:   r["logic"] = "사실/정보 전달"
        r["stance_score"] = max(-1.0, min(1.0, float(r.get("stance_score", 0.0))))
        out.append(r)
    return out


# ── 댓글 배치 ──────────────────────────────────────────────

def analyze_comments_batch(comments: list) -> list:
    """댓글 텍스트 리스트 → emotion_label / loaded_words 배치 분석"""
    if not comments:
        return []
    items = "\n".join(f'댓글 {i}: "{c}"' for i, c in enumerate(comments))
    prompt = f"""아래 {len(comments)}개 댓글을 각각 분석하세요.

## 감정 레이블 (9종): {" / ".join(EMOTION_LABELS)}
## loaded_words: 편향·자극적 단어 (radical/racist/extremist/illegal/dangerous/propaganda/fake/biased 등)

{items}

JSON 배열만 출력:
[
  {{"idx":0,"emotion_label":"...","emotion_intensity":0.0,"loaded_words":["단어1","단어2","단어3"]}},
  ...
]
규칙: emotion_intensity=0.0~1.0, loaded_words 최대 3개 (없으면 [])"""
    results = call_llm_json(prompt)
    results.sort(key=lambda r: r.get("idx", 0))
    out = []
    for r in results:
        if r.get("emotion_label") not in EMOTION_LABELS: r["emotion_label"] = "neutral"
        r["emotion_intensity"] = max(0.0, min(1.0, float(r.get("emotion_intensity", 0.5))))
        r["loaded_words"] = r.get("loaded_words", [])[:3]
        out.append(r)
    return out
