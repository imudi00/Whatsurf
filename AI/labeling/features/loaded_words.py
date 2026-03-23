# labeling/features/loaded_words.py
"""
한국어 뉴스 편향 단어(Loaded Words) 감지
(구 feature_map/emotion/src/feature_map/loaded_words.py)

- BIAS_WORDS   : 정치/이념/위기 편향어 (tier-1, 강함)
- OPINION_WORDS: 강한 평가/감정 표현 + 인터넷 신조어 (tier-2, 폭넓음)

\b 경계는 한국어에 작동 안 하므로 단순 포함 검색 사용.
"""
import re
from typing import List

# ──────────────────────────────────────────────
# TIER 1 — 정치/이념 편향어
# ──────────────────────────────────────────────
BIAS_WORDS: List[str] = [
    # 정치적 낙인 / 이념
    "극우", "극좌", "극단주의", "과격파", "급진주의",
    "빨갱이", "종북", "친일파", "토착왜구", "좌빨", "수꼴",
    "독재", "독재자", "파시스트", "전체주의", "폭군",
    "매국", "매국노", "역적", "반역", "반역자",
    # 선동
    "선동", "선동꾼", "선동세력", "선동적", "선동하다",
    "폭동", "봉기", "반란", "扇動",
    # 공격적 비하
    "꼴통", "수구꼴통", "개돼지", "쓰레기", "찌꺼기",
    "떼거리", "패거리", "일당",
    # 위기·재앙 과장
    "재앙", "파국", "대란", "붕괴", "파멸", "멸망",
    "폭락", "폭등", "대혼란", "대위기", "망국",
    # 공분·분노 유발
    "경악", "공분", "격분", "격노", "참담", "참혹",
    "처참", "처절", "비극", "참극",
    # 척결·타도
    "척결", "타도", "근절", "청산", "박멸", "섬멸",
    "투쟁", "항쟁", "결사",
    # 음모·조작 프레임
    "조작", "날조", "가짜뉴스", "왜곡", "은폐",
    "밀실", "야합", "결탁", "카르텔", "적폐", "기득권",
    # 공포 조장
    "협박", "겁박", "압박", "위협", "공포정치",
]

# ──────────────────────────────────────────────
# TIER 2 — 강한 평가/감정/신조어 (의견 강도 높음)
# ──────────────────────────────────────────────
OPINION_WORDS: List[str] = [
    # 강한 부정 평가
    "최악", "형편없", "엉망", "한심", "황당", "어이없",
    "기가막", "말도안", "뻔뻔", "뻔뻔스", "무능", "무능력",
    "실망", "충격", "경악", "분노", "화나", "짜증",
    "헛소리", "개소리", "거짓말", "사기꾼",
    "쓸모없", "무책임", "무능한",
    # 강한 긍정 평가
    "최고", "대박", "훌륭", "완벽", "멋지", "대단",
    "완전히", "진짜로", "정말로",
    # 강한 의견 표현
    "절대", "반드시", "당연히", "무조건", "확실히",
    "솔직히", "솔까말", "진심으로", "진심",
    # 강한 감탄/놀람
    "헐", "대박", "실화냐", "미쳤다", "미친", "이게뭐야",
    "어쩌라고", "어이없다", "황당하다",
    # 인터넷 슬랭 / 신조어 (강한 의견 신호)
    "레알", "리얼", "노답", "개판", "난리", "폭발",
    "핵", "존나", "완전", "개-", "갑자기왜",
    "ㄹㅇ", "ㅇㅈ", "ㅂㄷㅂㄷ", "분노유발",
    # 감정 표현 자음 (인터넷)
    "ㅋㅋ", "ㅎㅎ", "ㄷㄷ", "ㅠㅠ", "ㅜㅜ", "ㅡㅡ",
    "ㅋㅋㅋ", "ㅠㅠㅠ", "ㄷㄷㄷ",
    # 강한 비판/냉소
    "웃기다", "웃기지", "한마디로", "결국에는",
    "이러니까", "그러니까", "당연하지", "역시나",
    "또또", "맨날", "항상이래", "이번에도",
    # 정치·사회 강한 의견
    "정치놀음", "국민무시", "세금도둑", "나랏돈",
    "민심이반", "민심폭발", "심각하다", "위험하다",
]

# ──────────────────────────────────────────────
# 감지 함수
# ──────────────────────────────────────────────

def detect_loaded_words(text: str) -> List[str]:
    """BIAS_WORDS(tier-1)에서 감지된 단어 목록 반환 (중복 제거, 출현 순)"""
    found, seen = [], set()
    for w in BIAS_WORDS:
        if w not in seen and w in text:
            found.append(w)
            seen.add(w)
    return found


def detect_opinion_words(text: str) -> List[str]:
    """OPINION_WORDS(tier-2)에서 감지된 단어 목록 반환"""
    found, seen = [], set()
    for w in OPINION_WORDS:
        if w not in seen and w in text:
            found.append(w)
            seen.add(w)
    return found


# 자음 반복 패턴 (ㅋㅋ, ㄷㄷ, ㅠ 등 2글자 이상)
_CONSONANT_RE = re.compile(r'[ㄱ-ㅎㅏ-ㅣ]{2,}')

def detect_informal_patterns(text: str) -> List[str]:
    """ㅋㅋ, ㄷㄷ, ㅠㅠ 같은 자음/모음 반복 패턴 추출 (중복 제거)"""
    matches = _CONSONANT_RE.findall(text)
    seen, result = set(), []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def count_loaded_words(text: str) -> int:
    return sum(text.count(w) for w in BIAS_WORDS)


def loaded_word_density(text: str) -> float:
    """총 단어 수 대비 편향 단어 비율 (0~1)"""
    total_words = max(len(text.split()), 1)
    return count_loaded_words(text) / total_words


def mask_loaded_words(text: str, mask_token: str = "[MASK]") -> str:
    masked = text
    for w in BIAS_WORDS:
        masked = masked.replace(w, mask_token)
    return masked


def replace_loaded_words(text: str, replacement: str = "***") -> str:
    return mask_loaded_words(text, mask_token=replacement)
