# preprocessor.py
import re

def preprocess(text: str) -> dict:
    """
    본문을 헤드라인/리드/본문/인용/수치로 구조 분해
    LLM에게 raw text 대신 이 구조체를 넘김
    """
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    
    # 헤드라인: 첫 줄 (또는 별도 전달)
    headline = lines[0] if lines else ""
    
    # 리드문: 첫 문단 (보통 핵심 요약)
    paragraphs = text.strip().split('\n\n')
    lead = paragraphs[0].strip() if paragraphs else ""
    
    # 인용구 추출
    quotes = re.findall(r'["\'\"](.*?)["\'\"]', text)
    quotes += re.findall(r'(.{5,30}?)(?:라고|이라고|고)\s(?:밝혔|말했|전했|주장했|강조했)', text)
    quotes = list(set(quotes))[:5]
    
    # 수치/통계 추출
    numbers = re.findall(r'\d+[\.,]?\d*\s*[%억만원명개건회%]', text)
    
    # 주요 술어 추출 (감정/판단 동사)
    judgment_words = re.findall(
        r'[\w]+(?:비판|우려|촉구|반발|환영|지적|경고|강조|주장|요구|반대|찬성|규탄|촉구)[\w]*',
        text
    )
    
    # 출처 표현
    sources = re.findall(r'([가-힣\w]+(?:부|처|청|원|원|회|단|협|측|관계자|전문가|교수|연구원))', text)
    sources = list(set(sources))[:5]
    
    # 본문 전체 문장 리스트
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
    
    # 앞/중/뒤 문장 샘플링 (토큰 절약하면서 전체 흐름 포착)
    n = len(sentences)
    sampled = []
    if n > 0: sampled.append(sentences[0])
    if n > 2: sampled.append(sentences[n//2])
    if n > 1: sampled.append(sentences[-1])
    
    return {
        "headline": headline,
        "lead": lead[:200],
        "quotes": quotes,
        "numbers": numbers[:5],
        "judgment_words": judgment_words[:5],
        "sources": sources,
        "sampled_sentences": sampled,
        "total_sentences": n,
    }