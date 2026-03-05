# body_depth.py
import re

def compute_body_depth(text: str, features: dict) -> float:
    """
    body_depth = 0.4 * (문단수 정규화) + 0.3 * (정보문장 비율) + 0.3 * (배경키워드 존재)
    """
    # 문단수 정규화 (5문단 이상이면 1.0)
    para_score = min(features["paragraph_count"] / 5.0, 1.0)
    
    # 정보문장 비율 (수치/인용/출처 포함 문장 비율)
    info_keywords = re.compile(r'\d+[%억만원개명]|"[^"]{3,}"|에 따르면|관계자|밝혔다|분석|연구|조사')
    info_sentences = sum(1 for s in features["sentences"] if info_keywords.search(s))
    info_ratio = info_sentences / max(features["sentence_count"], 1)
    
    # 배경 키워드 존재 여부
    background_keywords = ['배경', '원인', '역사', '맥락', '이유', '경위', '전후', '과정']
    bg_score = 1.0 if any(kw in text for kw in background_keywords) else 0.0
    
    depth = 0.4 * para_score + 0.3 * info_ratio + 0.3 * bg_score
    return round(depth, 4)