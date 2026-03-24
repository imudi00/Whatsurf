# labeling/groq/bias_labeler.py
"""
Groq Llama 3.3 70B — bias_x / bias_y 자동 라벨링
bias_x: 좌(-1.0) ↔ 우(+1.0) 정치 성향
bias_y: 감성(-1.0: 부정) ↔ 사실(+1.0: 중립/사실)

배치 크기에 따라 본문 발췌 길이 자동 조절:
  ≤5개  : 본문 300자
  6~15개: 본문 180자
  16+개 : 본문 100자
"""
from .groq_client import call_groq_json


def _snippet_len(n: int) -> int:
    if n <= 5:   return 300
    elif n <= 15: return 180
    else:         return 100


def label_bias_batch(items: list) -> list:
    """
    items: build_article_struct() 결과 또는 {"id","headline","lead","body_snippet"} dict 리스트
    반환:  [{"idx","id","bias_x":float,"bias_y":float,"bias_reason":str}]
    """
    slen = _snippet_len(len(items))

    blocks = "\n\n".join(
        f"[{i}] id={it.get('id', i)}\n"
        f"헤드라인: {it.get('headline', it.get('title', ''))}\n"
        f"리드: {(it.get('lead') or '')[:120]}\n"
        f"본문: {str(it.get('body_snippet', it.get('sampled_sentences', '')))[:slen]}"
        for i, it in enumerate(items)
    )

    prompt = (
        f"아래 {len(items)}개 기사의 정치/감성 편향을 수치로 평가하세요.\n\n"
        "bias_x: -1.0(강한 진보) ~ +1.0(강한 보수)  예) 진보성향 -0.6 / 약한 보수 +0.3\n"
        "bias_y: -1.0(강한 감성·선동) ~ +1.0(순수 사실 보도)  예) 다소 감성적 -0.4 / 균형 +0.1\n\n"
        "주의: 0.0은 완전히 중립일 때만 사용. 미세한 편향도 ±0.1~±0.4로 반드시 수치화.\n\n"
        + blocks
        + "\n\nJSON 배열만 출력 (0.0 남용 금지):\n"
        "[{\"idx\":0,\"id\":\"...\",\"bias_x\":-0.3,\"bias_y\":0.5,\"bias_reason\":\"한줄\"},...]"
    )

    results = call_groq_json(prompt)
    if isinstance(results, dict):
        results = [results]
    results.sort(key=lambda r: r.get("idx", 0))

    out = []
    for r in results:
        try:
            r["bias_x"] = max(-1.0, min(1.0, float(r.get("bias_x", 0.0))))
            r["bias_y"] = max(-1.0, min(1.0, float(r.get("bias_y", 0.0))))
        except (TypeError, ValueError):
            r["bias_x"], r["bias_y"] = 0.0, 0.0
        out.append(r)

    while len(out) < len(items):
        out.append({"idx": len(out), "id": str(items[len(out)].get("id", len(out))),
                    "bias_x": 0.0, "bias_y": 0.0, "bias_reason": ""})
    return out[:len(items)]
