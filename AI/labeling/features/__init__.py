# labeling/features/__init__.py
"""
feature_map에서 통합된 로컬 텍스트 피처 모듈.

주요 모듈:
    preprocessor      — LLM 입력용 기사 구조체 생성 (build_article_struct)
    text_utils        — 텍스트 전처리 유틸 (preprocessor 의존)
    loaded_words      — 편향 단어 사전 + 감지 함수 (BIAS_WORDS, detect_loaded_words 등)
    keyword_extractor — KPF-BERT NER 기반 키워드/엔티티 추출 (extract_features)
"""
