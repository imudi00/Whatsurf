# 모듈 상세 문서 — `labeling/local/`

> **로컬 HuggingFace 모델 래퍼 + 규칙 기반 피처**
> 외부 API 호출 없이 로컬에서 즉시 실행. GPU 없으면 CPU 자동 전환.

---

## 1. 모듈 개요

기사·댓글에서 논조 점수, 감정, 편향 단어를 **로컬 모델 또는 규칙**으로 라벨링합니다.
`run_labeling.py`가 로컬 블록 순서대로 (`body_depth` → `stance` → `loaded_words` → `emotion`) 호출합니다.

---

## 2. 파일 구성

| 파일 | 피처 | 방식 | 의존 모델 |
|---|---|---|---|
| `body_depth.py` | `body_depth` | 수식 기반 | 없음 |
| `stance_labeler.py` | `stance_score`, `stance_label` | HuggingFace | snunlp/KR-FinBert-SC |
| `loaded_words_labeler.py` | `art_words`, `loaded_word_density`, `is_biased` | 규칙 기반 | 없음 |
| `emotion_labeler.py` | `primary_emotion`, `emotion_intensity`, `emotion_probs` | HuggingFace | hun3359/klue-bert-base-sentiment |

---

## 3. `body_depth.py` — 본문 정보 깊이 점수

5가지 텍스트 지표를 가중합으로 계산하는 **수식 기반** 피처.
모델 없이 즉시 실행되며 WEIGHTS 상수로 튜닝 가능.

### 튜닝 파라미터

| 상수 | 기본값 | 설명 |
|---|---|---|
| `WEIGHTS["length"]` | 0.20 | 본문 길이 (로그 정규화) |
| `WEIGHTS["diversity"]` | 0.25 | 어절 type-token ratio |
| `WEIGHTS["quotes"]` | 0.20 | 인용문 비율 |
| `WEIGHTS["numerics"]` | 0.20 | 수치·단위 패턴 밀도 |
| `WEIGHTS["structure"]` | 0.15 | 비어있지 않은 줄 수 (문단 다양성) |
| `LENGTH_SOFT_MAX` | 2000 | 길이 포화점 (char) |
| `DIVERSITY_TTR_REF` | 0.55 | TTR 기준값 |
| `QUOTE_RATIO_REF` | 0.12 | 인용 비율 기준값 |
| `NUMERIC_DENSITY_REF` | 0.06 | 수치 밀도 기준값 |
| `STRUCTURE_PARA_MAX` | 8 | 문단 수 포화점 |

### 주요 함수

| 함수명 | 인수 | 반환 타입 | 설명 |
|---|---|---|---|
| `compute_body_depth(text)` | `str` | `float` | body_depth 점수 계산 (0.0~1.0) |
| `describe_body_depth(score)` | `float` | `str` | "high"(≥0.65) / "medium"(≥0.35) / "low" |
| `get_body_depth_detail(text)` | `str` | `dict` | 세부 점수 반환 (디버깅용) |

### 입출력 예시

```python
# Input
text = "의대 정원 2000명 증원 확정\n\n정부는 2025학년도부터 의대 정원을 2000명 늘리기로 했다..."

# Output
compute_body_depth(text)
# → 0.6123

describe_body_depth(0.6123)
# → "medium"

get_body_depth_detail(text)
# → {
#     "body_depth": 0.6123,
#     "label": "medium",
#     "detail": {
#         "length":    0.8941,   # 로그 정규화
#         "diversity": 0.7200,   # TTR 기반
#         "quotes":    0.4167,   # 인용 비율
#         "numerics":  0.8333,   # 수치 밀도
#         "structure": 0.6250    # 문단 수
#     }
# }
```

---

## 4. `stance_labeler.py` — 기사 논조 점수

**snunlp/KR-FinBert-SC** 모델로 기사의 감성(긍정·부정·중립)을 측정하고
`stance_score` (연속값)와 `stance_label` (3분류)로 변환합니다.

### 모델 설정

```
모델: snunlp/KR-FinBert-SC
라벨: label_0=negative, label_1=neutral, label_2=positive
최대 토큰: 256
```

### stance_score 계산

```
stance_score = P(positive) - P(negative)  ∈ [-1.0, +1.0]
```

| 범위 | `stance_label` |
|---|---|
| score > 0.15 | `"우호적"` |
| score < -0.15 | `"비판적"` |
| 그 외 | `"중립"` |

### 주요 함수

| 함수명 | 인수 | 반환 타입 | 설명 |
|---|---|---|---|
| `label_stance_batch(texts)` | `List[str]` | `List[dict]` | run_labeling.py 인터페이스 |
| `label_stance(items)` | `List[{id, text}]` | `List[dict]` | id 포함 버전 |
| `get_labeler(model_name)` | `str` | `LocalStanceLabeler` | 싱글턴 반환 |

### 입출력 예시

```python
# Input
texts = ["정부의 의대 증원 결정은 매우 잘못되었다.", "새 정책이 발표되었다."]

# Output
label_stance_batch(texts)
# → [
#     {"stance_score": -0.6821, "stance_label": "비판적"},
#     {"stance_score":  0.0312, "stance_label": "중립"}
# ]
```

---

## 5. `loaded_words_labeler.py` — 편향 단어 추출

`labeling/features/loaded_words.py`의 사전을 사용하여 3단계 폴백으로 최소 단어 수를 보장합니다.

### 3단계 폴백

| Tier | 소스 | 신호 강도 |
|---|---|---|
| Tier-1 | `BIAS_WORDS` 사전 | 가장 강함 (정치·이념 편향) |
| Tier-2 | `OPINION_WORDS` + 자음반복패턴 | 중간 (강한 감정·신조어) |
| Tier-3 | 텍스트 자체에서 빈도 기반 추출 | 폴백 (항상 최소 2개 보장) |

### 주요 함수

| 함수명 | 인수 | 반환 타입 | 설명 |
|---|---|---|---|
| `label_loaded_words_batch(texts, max_words=10, min_guaranteed=2)` | `List[str]` | `List[dict]` | run_labeling.py 인터페이스 |

### 입출력 예시

```python
# Input
texts = ["정부의 독재적 선동이 재앙을 부르고 있다.", "오늘 국회에서 예산안을 논의했다."]

# Output
label_loaded_words_batch(texts)
# → [
#     {
#         "loaded_words":        ["독재", "선동", "재앙"],
#         "loaded_word_density": 0.1667,
#         "is_biased":           True,
#         "tier":                1
#     },
#     {
#         "loaded_words":        ["오늘", "국회"],   # Tier-3 폴백
#         "loaded_word_density": 0.0,
#         "is_biased":           False,
#         "tier":                3
#     }
# ]
```

---

## 6. `emotion_labeler.py` — 댓글 감정 분류

**hun3359/klue-bert-base-sentiment** 모델로 댓글의 감정을 분류합니다.
`run_labeling.py`의 댓글 파이프라인 (`_process_one_article_comments`)에서 호출합니다.

### 모델 설정

```
모델: hun3359/klue-bert-base-sentiment (기본, .env EMOTION_MODEL로 변경 가능)
라벨: 모델 config.id2label에서 자동 로드 (한국어 레이블)
최대 토큰: 256
```

### 감정 레이블 매핑 (한국어 → 영어)

| 원본 | 변환 | 원본 | 변환 |
|---|---|---|---|
| 기쁨·행복·즐거움 | joy | 슬픔·우울 | sadness |
| 분노·화남 | anger | 두려움·불안·공포 | fear |
| 놀람·당혹감 | surprise | 혐오·싫음 | disgust |
| 상처 | hurt | 중립·보통 | neutral |

### 주요 함수

| 함수명 | 인수 | 반환 타입 | 설명 |
|---|---|---|---|
| `label_emotion_batch(texts, model_name)` | `List[str]` | `List[dict]` | run_labeling.py 인터페이스 |
| `_get_labeler(model_name)` | `str` | `LocalEmotionLabeler` | 싱글턴 반환 |

### 입출력 예시

```python
# Input
texts = ["이게 말이 되냐고 진짜 화난다", "좋은 결과가 나왔으면 좋겠다"]

# Output
label_emotion_batch(texts)
# → [
#     {
#         "primary_emotion":   "anger",
#         "raw_label":         "분노",
#         "emotion_intensity": 0.8312,
#         "emotion_probs": {
#             "anger": 0.8312, "disgust": 0.0921, "fear": 0.0412,
#             "joy": 0.0180, "neutral": 0.0175, ...
#         }
#     },
#     {
#         "primary_emotion":   "joy",
#         "raw_label":         "기쁨",
#         "emotion_intensity": 0.6240,
#         "emotion_probs": {"joy": 0.6240, "neutral": 0.2100, ...}
#     }
# ]
```

### `upload_to_db.py`에서의 감정 처리

`emotion_probs`에서 상위 10개 감정만 JSON으로 저장 (`cmt_emotion`):
```python
# cmt_emotion 예시 (top 10)
{
    "anger":   0.8312,
    "disgust": 0.0921,
    "fear":    0.0412,
    "joy":     0.0180,
    "neutral": 0.0175
}
```

---

## 7. 실행 환경 요구사항

| 패키지 | 용도 |
|---|---|
| `torch` | 모델 추론 (CPU/GPU 자동) |
| `transformers` | AutoTokenizer, AutoModelForSequenceClassification, pipeline |

```bash
pip install torch transformers
```

모델 파일은 첫 실행 시 HuggingFace Hub에서 자동 다운로드:
- `snunlp/KR-FinBert-SC` (~400MB)
- `hun3359/klue-bert-base-sentiment` (~400MB)

> `.env`의 `EMOTION_MODEL`을 변경하면 다른 감정 모델로 교체 가능.
