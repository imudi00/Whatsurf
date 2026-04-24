# 모듈 상세 문서 — `labeling/features/`

> 구 `feature_map/*/src/feature_map/`에서 통합된 **로컬 텍스트 피처 모듈**
> LLM·HuggingFace 모델을 사용하지 않으며, 순수 규칙/정규식/NER 기반으로 동작합니다.

---

## 1. 모듈 개요

LLM 호출 전 기사 구조체를 준비하거나, 규칙 기반으로 편향 단어·피처를 추출하는 **전처리/피처 유틸** 집합.
`labeling/run_labeling.py`와 `labeling/local/loaded_words_labeler.py`가 직접 의존합니다.

---

## 2. 파일 구성

| 파일 | 역할 | 의존성 |
|---|---|---|
| `preprocessor.py` | 기사 본문 → LLM 입력용 구조체 생성 | `text_utils.py` |
| `text_utils.py` | 텍스트에서 헤드라인·인용·수치·판단어 추출 | 없음 |
| `loaded_words.py` | 편향 단어 사전 + 감지·계수·밀도 함수 | 없음 |
| `keyword_extractor.py` | KPF-BERT NER 기반 엔티티·키워드 추출 | `transformers` |
| `bias_vector.py` | 좌-우 편향 벡터 계산 (규칙 기반) | 없음 |
| `omission_risk.py` | 클러스터 대비 엔티티 누락도 계산 (규칙 기반) | `keyword_extractor.py` |

---

## 3. `preprocessor.py` — LLM 입력 구조체 생성

기사 텍스트를 LLM이 이해하기 좋은 요소로 분해합니다.
`run_labeling.py`의 `_run_groq_block()`이 Groq 배치 전 호출합니다.

### 주요 함수

| 함수명 | 인수 | 반환 타입 | 설명 |
|---|---|---|---|
| `build_article_struct(text)` | `str` | `dict` | 기사 본문 → 구조체 변환 |

### 반환 구조체 예시

```python
{
    "headline":          "의대 증원 강행…교육부, 대학에 최후통첩",
    "lead":              "정부가 2025학년도 의대 정원을 2000명 늘리기로 확정하면서...",
    "quotes":            ["전례 없는 일이다", "협의 없이 일방 통보했다"],
    "numbers":           ["2000명", "30%", "40억원"],
    "judgment_words":    ["강행비판", "우려촉구", "반발"],
    "sources":           ["교육부", "의협", "대학 관계자"],
    "sampled_sentences": ["정부가 결정했다.", "학교 측은 반대했다.", "사태는 장기화될 전망이다."],
    "total_sentences":   24
}
```

---

## 4. `text_utils.py` — 텍스트 전처리 유틸

`preprocessor.py`가 의존하는 정규식 기반 추출 함수 모음.

| 함수명 | 반환 타입 | 설명 |
|---|---|---|
| `extract_headline(text)` | `str` | 첫 번째 비어있지 않은 줄 |
| `extract_lead(text, max_len=200)` | `str` | 첫 문단 (최대 200자) |
| `extract_quotes(text, max_count=5)` | `List[str]` | 인용구 + "라고/이라고" 패턴 |
| `extract_numbers(text, max_count=5)` | `List[str]` | 숫자+단위 패턴 (%, 억, 원, 명 등) |
| `extract_judgment_words(text, max_count=5)` | `List[str]` | 판단·감정 술어 (비판·촉구·반발 등) |
| `extract_sources(text, max_count=5)` | `List[str]` | 기관명·직함 패턴 (부·처·청·관계자·교수 등) |
| `sample_sentences(text)` | `List[str]` | 앞·중·뒤 문장 3개 샘플 (토큰 절약용) |
| `normalize_text(s)` | `str` | 제로폭 공백 제거 + 연속 공백 정규화 |

---

## 5. `loaded_words.py` — 편향 단어 사전

한국어 뉴스의 감정·이념적 색깔이 강한 단어 사전과 감지 함수.
`labeling/local/loaded_words_labeler.py`가 이 파일에서 import합니다.

### 사전 구성

| 사전 | 상수명 | 설명 | 단어 수 |
|---|---|---|---|
| Tier-1 | `BIAS_WORDS` | 정치·이념 편향어 (극우, 빨갱이, 재앙, 척결 등) | ~60개 |
| Tier-2 | `OPINION_WORDS` | 강한 평가·감정·신조어 (최악, ㅋㅋ, 노답 등) | ~80개 |

### 함수

| 함수명 | 반환 타입 | 설명 |
|---|---|---|
| `detect_loaded_words(text)` | `List[str]` | BIAS_WORDS에서 감지된 단어 목록 (순서 유지) |
| `detect_opinion_words(text)` | `List[str]` | OPINION_WORDS에서 감지된 단어 목록 |
| `detect_informal_patterns(text)` | `List[str]` | ㅋㅋ·ㄷㄷ·ㅠㅠ 자음/모음 반복 패턴 |
| `count_loaded_words(text)` | `int` | BIAS_WORDS 총 등장 횟수 |
| `loaded_word_density(text)` | `float` | 전체 어절 대비 편향 단어 비율 (0~1) |
| `mask_loaded_words(text, mask_token="[MASK]")` | `str` | 편향 단어를 [MASK]로 치환 |
| `replace_loaded_words(text, replacement="***")` | `str` | 편향 단어를 지정 문자열로 치환 |

### 입출력 예시

```python
text = "극우 세력이 재앙적인 정책을 선동하고 있다."
detect_loaded_words(text)
# → ["극우", "재앙", "선동"]

loaded_word_density(text)
# → 0.375  (전체 8어절 중 BIAS_WORDS 3개 × 1회)
```

---

## 6. `keyword_extractor.py` — KPF-BERT NER

**KPF/KPF-bert-ner** 모델로 한국어 뉴스 도메인 엔티티를 추출합니다.
`run_labeling.py`의 `_run_gemini_block()`이 omission_risk 전처리용으로 호출합니다.

> ⚠️ 첫 실행 시 HuggingFace에서 모델 다운로드 (~500MB)

### 함수

| 함수명 | 반환 타입 | 설명 |
|---|---|---|
| `get_ner_pipeline()` | `transformers.Pipeline` | 싱글턴 NER 파이프라인 |
| `extract_ner_entities(text, token_limit=512)` | `List[dict]` | NER 엔티티 목록 |
| `extract_features(text)` | `dict` | 통합 피처 추출 (entities + 통계) |

### `extract_features` 반환 구조체 예시

```python
{
    "sentences":       ["정부가 결정했다", "의협이 반발했다", ...],
    "entities":        [{"word": "교육부", "entity_group": "ORG", "score": 0.99}, ...],
    "entity_count":    8,
    "entity_density":  0.33,   # entities / sentences
    "num_density":     0.12,   # 수치 토큰 / sentences
    "quote_count":     3,
    "source_count":    4,
    "paragraph_count": 6,
    "sentence_count":  24
}
```

---

## 7. `bias_vector.py` — 정치 편향 벡터 (규칙 기반)

영어 편향 신호 키워드로 좌-우 스펙트럼 점수를 계산하는 경량 모듈.
현재 파이프라인에서는 직접 사용되지 않으나 향후 확장을 위해 보관.

### 출력 필드 (`BiasVectorOutput`)

| 필드 | 타입 | 범위 | 설명 |
|---|---|---|---|
| `tone_dem` | `float` | 0~1 | 민주·진보 성향 신호 비율 |
| `tone_rep` | `float` | 0~1 | 보수·공화 성향 신호 비율 |
| `lr_score` | `float` | -1~+1 | 좌(−) ↔ 우(+) 종합 점수 |
| `rationale` | `str` | 좌편향/중립/우편향 | 방향 레이블 |

| 함수명 | 반환 타입 | 설명 |
|---|---|---|
| `compute_bias_vector(text)` | `BiasVectorOutput` | 원시 편향 벡터 계산 |
| `normalize_bias_vector(bv)` | `BiasVectorOutput` | 정규화 (현재 pass-through) |

---

## 8. `omission_risk.py` — 클러스터 대비 누락도 (규칙 기반)

동종 기사 클러스터에서 핵심 엔티티가 얼마나 빠졌는지 계산.
현재 고품질 omission_risk는 `labeling/gemini/omission_labeler.py`가 담당하며,
이 모듈은 규칙 기반 경량 버전.

| 함수명 | 반환 타입 | 설명 |
|---|---|---|
| `collect_cluster_entities(articles, features_fn)` | `Counter` | 클러스터 전체 엔티티 빈도 집계 |
| `compute_omission_risk(text, cluster_articles, features_fn, threshold=0.5)` | `str` | `"low"` / `"mid"` / `"high"` |

---

## 9. 테스트 실행

```bash
# loaded_words 단독 테스트
python -c "
from labeling.features.loaded_words import detect_loaded_words
print(detect_loaded_words('극우 세력의 재앙적 선동이다'))
"

# preprocessor 단독 테스트
python -c "
from labeling.features.preprocessor import build_article_struct
import json, pprint
struct = build_article_struct('의대 증원 강행\n\n정부가 2000명 증원을 확정했다. 의협은 비판했다.')
pprint.pprint(struct)
"

# body_depth 분포 분석 (output 있을 때)
python AI/labeling/experiments/rule_based/body_depth_tuner.py \
    --labeled_dir AI/labeling/output --mode distribution
```
