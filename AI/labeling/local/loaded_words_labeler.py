# labeling/local/loaded_words_labeler.py
"""
로컬 규칙 기반 loaded_words 추출

3단계 보장 (항상 최소 2개 반환):
  Tier-1: BIAS_WORDS 사전 (정치/이념 편향어)
  Tier-2: OPINION_WORDS 사전 (강한 평가/감정/신조어 포함 ㅋㅋ ㄷㄷ 등)
  Tier-3: 텍스트에서 직접 의미 있는 단어 추출 (salient fallback)
"""
import re
from typing import List, Optional

# ── features/loaded_words 사전 로드 ──────────────────────────
from labeling.features.loaded_words import (
    detect_loaded_words, detect_opinion_words,
    detect_informal_patterns, BIAS_WORDS, OPINION_WORDS,
)


# ── Tier-3 Fallback: 텍스트에서 직접 추출 ─────────────────────

# 매우 흔한 조사/어미/기능어 (의미가 낮은 것들) — 너무 길면 성능 저하
_STOPWORDS = {
    "이","가","은","는","을","를","에","의","와","과","도","만","로","으로",
    "에서","에게","부터","까지","이다","있다","없다","하다","된다","되다",
    "그","이","저","그것","이것","저것","때","그래서","그런데","하지만",
    "따라서","또한","그리고","하지만","그러나","그래도","그러면","그래",
    "아","어","네","예","응","음","오","우","이제","이미","아직","매우",
    "더","덜","너무","조금","많이","잘","못","안","좀","제발","다시",
    "새로운","모든","각","어떤","이런","그런","저런","같은","어떻게",
}

def _extract_salient_fallback(text: str, min_needed: int) -> List[str]:
    """
    BIAS_WORDS + OPINION_WORDS로 부족할 때
    텍스트에서 의미 있어 보이는 단어를 직접 추출.
    우선순위: 자음반복 패턴 > 긴 단어 > 반복 등장 단어
    """
    # 기본 토큰화: 공백/특수문자로 분리
    tokens = re.split(r'[\s,!?.\'"·…\-~ㆍ]+', text)
    candidates = []
    freq: dict[str, int] = {}
    for t in tokens:
        t = t.strip()
        if len(t) < 2:
            continue
        # 자음 반복 패턴 (ㅋㅋ, ㄷㄷ, ㅠㅠ 등) — 높은 우선순위
        if re.fullmatch(r'[ㄱ-ㅎ]{2,}', t) or re.fullmatch(r'[ㅏ-ㅣ]{2,}', t):
            freq[t] = freq.get(t, 0) + 3   # 가중치 3
            if t not in candidates:
                candidates.append(t)
        elif t.lower() not in _STOPWORDS and not t.isdigit():
            freq[t] = freq.get(t, 0) + 1
            if t not in candidates:
                candidates.append(t)

    # 정렬: 가중치 × 길이 (긴 단어 우선)
    candidates.sort(key=lambda w: (freq.get(w, 0) * len(w)), reverse=True)
    return candidates[:min_needed]


# ── 핵심 추출 함수 ─────────────────────────────────────────────

def _extract_all(text: str, max_words: int = 10,
                 min_guaranteed: int = 2) -> dict:
    """
    3단계 fallback으로 최소 min_guaranteed개 반환.
    반환: {"words": [...], "tier_used": int, "density": float}
    """
    # Tier-1: BIAS_WORDS (정치/이념 편향어)
    tier1 = detect_loaded_words(text)

    # Tier-2: OPINION_WORDS + 자음반복 패턴
    tier2 = detect_opinion_words(text)
    tier2 += [w for w in detect_informal_patterns(text) if w not in tier2]

    combined = list(dict.fromkeys(tier1 + tier2))  # 중복 제거, 순서 유지

    tier_used = 1 if tier1 else (2 if tier2 else 3)

    # Tier-3 fallback: 아직 부족하면 텍스트에서 직접 추출
    if len(combined) < min_guaranteed:
        need = min_guaranteed - len(combined)
        fallback = [w for w in _extract_salient_fallback(text, need + 5)
                    if w not in combined]
        combined += fallback[:need]
        if not tier1 and not tier2:
            tier_used = 3

    words = combined[:max_words]
    total = max(len(text.split()), 1)
    density = round(sum(text.count(w) for w in tier1) / total, 4)  # density는 tier1 기준

    return {"words": words, "tier_used": tier_used, "density": density}


# ── 퍼블릭 인터페이스 ──────────────────────────────────────────

def label_loaded_words_batch(
    texts: List[str],
    max_words: int = 10,
    min_guaranteed: int = 2,
) -> List[dict]:
    """
    run_labeling.py 인터페이스.
    texts = [str, ...]
    반환: [{"loaded_words": [...], "loaded_word_density": float, "is_biased": bool, "tier": int}]

    tier=1 : 정치/이념 편향어 감지 (가장 강한 신호)
    tier=2 : 강한 평가/감정/신조어 감지
    tier=3 : 텍스트 자체에서 추출 (fallback)
    """
    results = []
    for text in texts:
        r = _extract_all(text, max_words=max_words, min_guaranteed=min_guaranteed)
        results.append({
            "loaded_words":        r["words"],
            "loaded_word_density": r["density"], 
            "is_biased":           r["tier_used"] <= 2 and len(r["words"]) > 0, 
            "tier":                r["tier_used"],
        })
    return results


# ── 하위 호환 ──────────────────────────────────────────────────

def label_loaded_words_rule(items: List[dict], max_words: int = 10) -> List[dict]:
    """items: [{"id":..., "text":...}]"""
    results = []
    for item in items:
        r = _extract_all(item["text"], max_words=max_words)
        results.append({"id": str(item["id"]), "loaded_words": r["words"], "density": r["density"]})
    return results


def label_loaded_words(items: List[dict], mode: str = "rule",
                       model_name: Optional[str] = None, max_words: int = 10) -> List[dict]:
    if mode == "rule":
        return label_loaded_words_rule(items, max_words)
    raise ValueError(f"mode='{mode}' 미지원 (현재 'rule'만 가능)")
