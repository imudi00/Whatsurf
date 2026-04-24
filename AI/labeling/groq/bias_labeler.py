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
        "bias_x: -1.0(강한 진보) ~ +1.0(강한 보수)\n"
        "  예) 강한 진보 -0.8 / 약한 진보 -0.3 / 약한 보수 +0.3 / 강한 보수 +0.8\n"
        "bias_y: -1.0(강한 감성·선동) ~ +1.0(순수 사실 보도)\n"
        "  예) 선동적 -0.7 / 다소 감성적 -0.3 / 다소 사실적 +0.3 / 사실 보도 +0.7\n\n"
        "⚠️ 규칙:\n"
        "  - 반드시 소수점 한 자리 이상의 수치를 사용하세요 (예: -0.3, +0.5)\n"
        "  - 0.0은 절대로 사용 금지. 미세한 편향도 ±0.1 이상으로 표현하세요.\n"
        "  - 한국 언론 특성상 완전한 중립 기사는 거의 없습니다.\n\n"
        + blocks
        + "\n\nJSON 배열만 출력:\n"
        "[{\"idx\":0,\"id\":\"...\",\"bias_x\":-0.3,\"bias_y\":0.5,\"bias_reason\":\"한줄\"},...]"
    )

    results = call_groq_json(prompt)
    if isinstance(results, dict):
        results = [results]
    results.sort(key=lambda r: r.get("idx", 0))

    out = []
    for r in results:
        try:
            bx = float(r.get("bias_x") or 0.0)
            by = float(r.get("bias_y") or 0.0)
            # LLM이 0.0을 반환하면 None으로 처리 (파싱 실패와 동일하게 DB에 NULL)
            r["bias_x"] = max(-1.0, min(1.0, bx)) if bx != 0.0 else None
            r["bias_y"] = max(-1.0, min(1.0, by)) if by != 0.0 else None
        except (TypeError, ValueError):
            r["bias_x"], r["bias_y"] = None, None
        out.append(r)

    # 배치 결과 부족분 보충 (None으로 채워 DB에 NULL 저장)
    while len(out) < len(items):
        out.append({"idx": len(out), "id": str(items[len(out)].get("id", len(out))),
                    "bias_x": None, "bias_y": None, "bias_reason": ""})
    return out[:len(items)]
