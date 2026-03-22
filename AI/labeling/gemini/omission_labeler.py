# labeling/gemini/omission_labeler.py
"""
Gemini 2.5 Pro — omission_risk 자동 라벨링
맥락 추론이 필요한 고난도 피처. RPD 한도 내에서만 실행.

omission_risk: 동종 클러스터 대비 핵심 정보 누락 정도
  - low  : 핵심 사실·주체·수치 대부분 포함
  - mid  : 일부 중요 관점이나 수치 누락
  - high : 핵심 사실 혹은 주요 당사자 관점 대부분 누락

배치 크기별 본문 압축:
  ≤5개  : 본문 400자
  6~10개: 본문 250자
  11+개 : 본문 150자
"""
from .gemini_client import call_gemini_pro_json, rpd_remaining


def _snippet_len(n: int) -> int:
    if n <= 5:    return 400
    elif n <= 10: return 250
    else:         return 150


def label_omission_batch(target_articles: list, cluster_summaries: list) -> list:
    """
    target_articles  : [{"id","title","body_snippet"}]
    cluster_summaries: 동종 이슈 핵심 엔티티/요약 목록 (str list)
    반환: [{"id","omission_risk":"low/medium/high","omission_reason":"..."}]
    """
    remaining = rpd_remaining()
    if remaining < len(target_articles):
        print(f"  [Gemini] ⚠ 잔여 RPD {remaining}건 → {remaining}개만 처리")
        target_articles = target_articles[:remaining]

    if not target_articles:
        return []

    slen = _snippet_len(len(target_articles))
    cluster_block = "\n".join(f"- {s}" for s in cluster_summaries[:10])
    target_block  = "\n\n".join(
        f"[{i}] id={a.get('id', i)}\n"
        f"제목: {a.get('title', a.get('headline', ''))}\n"
        f"본문: {str(a.get('body_snippet', a.get('sampled_sentences', '')))[:slen]}"
        for i, a in enumerate(target_articles)
    )

    prompt = (
        f"동종 이슈 기사들과 비교해 {len(target_articles)}개 기사의 omission_risk를 판단하세요.\n\n"
        "omission_risk: low(핵심 대부분 포함) / mid(일부 누락) / high(핵심 다수 누락)\n\n"
        f"## 클러스터 핵심 내용:\n{cluster_block}\n\n"
        f"## 분석 대상:\n{target_block}\n\n"
        "JSON 배열만 출력:\n"
        "[{\"idx\":0,\"id\":\"...\",\"omission_risk\":\"low/mid/high\",\"omission_reason\":\"한줄\"},...]"
    )

    results = call_gemini_pro_json(prompt)
    if isinstance(results, dict):
        results = [results]
    results.sort(key=lambda r: r.get("idx", 0))

    out = []
    for r in results:
        if r.get("omission_risk") not in ("low", "med", "high"):
            r["omission_risk"] = "med"
        out.append(r)

    while len(out) < len(target_articles):
        out.append({"idx": len(out), "id": str(target_articles[len(out)].get("id", len(out))),
                    "omission_risk": "med", "omission_reason": ""})
    return out[:len(target_articles)]
