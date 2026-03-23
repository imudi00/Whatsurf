# 모듈 상세 문서 — `labeling/groq/` + `labeling/gemini/`

> **LLM API 클라이언트 모듈**
> 고복잡도 피처(frame/logic/bias/omission)를 외부 LLM API로 라벨링합니다.
> 모든 클라이언트는 키 로테이션, 모델 폴백, JSON 파싱 보정을 내장합니다.

---

## 1. 모듈 개요

| 디렉토리 | 담당 피처 | API | 모델 |
|---|---|---|---|
| `groq/` | frame, logic, bias_x, bias_y | Groq Cloud | Llama 3.3 70B (기본) |
| `gemini/` | omission_risk | Google Gemini | Gemini 2.5 Pro (기본) |

---

## 2. `labeling/groq/` — Groq LLM

### 파일 구성

| 파일 | 역할 |
|---|---|
| `groq_client.py` | API 키·모델 로테이션, JSON 파싱, `<think>` 블록 제거 |
| `frame_labeler.py` | frame / logic 분류 (배치 처리) |
| `bias_labeler.py` | bias_x / bias_y 회귀 (배치 처리) |

---

### 2.1 `groq_client.py` — Groq API 클라이언트

#### 핵심 기능

| 기능 | 설명 |
|---|---|
| 키 로테이션 | `GROQ_API_KEYS` 환경변수에서 콤마 구분 복수 키. RPM/TPM 한도 초과 시 자동 전환 |
| 모델 폴백 | `GROQ_MODELS` 환경변수에서 순서대로 폴백. 모델 한도 소진 시 다음 모델 전환 |
| `<think>` 블록 제거 | Qwen3·llama-4-scout 등 reasoning 모델의 `<think>...</think>` 출력 자동 제거 |
| JSON 파싱 보정 | 코드 펜스(```json) 제거, 부분 JSON 복구, trailing comma 처리 |

#### 주요 함수 / 전역 변수

| 이름 | 타입 | 설명 |
|---|---|---|
| `call_groq_json(prompt, system, model, max_tokens, temperature)` | `dict` | 프롬프트 → JSON dict 반환 |
| `last_call_info` | `dict` | 마지막 호출 모델명 등 메타 |
| `rotation_log` | `list` | 키/모델 로테이션 이벤트 기록 |

#### `.env` 설정

```env
GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3
GROQ_MODELS=meta-llama/llama-3.3-70b-versatile,qwen/qwen3-32b,meta-llama/llama-3.1-8b-instant
```

---

### 2.2 `frame_labeler.py` — frame / logic 분류

7개 프레임 + 6개 논리 유형을 **숫자(1~7 / 1~6)로만 출력**하도록 프롬프트 설계.
한글 오자 문제(llama-4-scout 등)를 근본적으로 우회합니다.

#### 레이블 정의

**Frame (1~7)**:
| 숫자 | 레이블 |
|---|---|
| 1 | 사건 원인 집중 |
| 2 | 갈등/대립 강조 |
| 3 | 개인 사례 중심 |
| 4 | 경제적 영향 강조 |
| 5 | 윤리/도덕 판단 |
| 6 | 안전/안보 위협 |
| 7 | 권리/인권 강조 |

**Logic (1~6)**:
| 숫자 | 레이블 |
|---|---|
| 1 | 정책적 비난 |
| 2 | 전문가 견해 |
| 3 | 피해자 서사 |
| 4 | 파급효과 |
| 5 | 해결책 제시 |
| 6 | 사실/정보 전달 |

#### `_match_label()` — 숫자/텍스트 이중 파싱

```
1순위: 정확 일치 (숫자 또는 텍스트)
2순위: 숫자 index 분기 (LLM이 숫자만 반환한 경우)
3순위: 부분 포함 일치
4순위: fuzzy matching (SequenceMatcher, threshold=0.40) → 한글 오자 보정
```

#### 배치 처리 파라미터

| 배치 크기 | 리드 길이 | 판단어 수 | 인용구 수 |
|---|---|---|---|
| 1~5개 | 200자 | 8개 | 3개 |
| 6~15개 | 120자 | 5개 | 2개 |
| 16개+ | 70자 | 3개 | 1개 |

#### 주요 함수

| 함수명 | 인수 | 반환 타입 | 설명 |
|---|---|---|---|
| `label_frame_logic_batch(structs)` | `List[dict]` | `List[dict]` | 배치 프레임·논리 분류 |

#### 입출력 예시

```python
# Input (build_article_struct 결과 리스트)
structs = [
    {
        "headline": "의대 증원 강행…교육부, 대학에 최후통첩",
        "lead": "정부가 2025학년도 의대 정원을 2000명 늘리기로 확정하면서...",
        "quotes": ["전례 없는 일이다", "협의 없이 일방 통보했다"],
        "numbers": ["2000명", "30%"],
        "judgment_words": ["강행비판", "우려촉구"],
        ...
    }
]

# Output
label_frame_logic_batch(structs)
# → [
#     {
#         "frame":        "갈등/대립 강조",
#         "frame_reason": "정부-의협 간 갈등 구도를 전면에 배치",
#         "logic":        "정책적 비난",
#         "logic_reason": "정부 정책의 부당성을 중심으로 논거 구성"
#     }
# ]
```

---

### 2.3 `bias_labeler.py` — bias_x / bias_y 회귀

2차원 편향 좌표를 `-1.0 ~ +1.0` 실수로 출력.

| 축 | 범위 | 의미 |
|---|---|---|
| `bias_x` | -1.0 ~ +1.0 | -1.0=진보, 0=중립, +1.0=보수 |
| `bias_y` | -1.0 ~ +1.0 | -1.0=감성적, 0=균형, +1.0=사실적 |

#### 배치 처리 파라미터

| 배치 크기 | 본문 길이 |
|---|---|
| 1~5개 | 300자 |
| 6~15개 | 180자 |
| 16개+ | 100자 |

#### 입출력 예시

```python
# Input
structs = [{"headline": "...", "lead": "...", ...}]

# Output
label_bias_batch(structs)
# → [
#     {
#         "bias_x":      -0.35,
#         "bias_y":       0.20,
#         "bias_reason": "정부 정책에 비판적, 피해자 감성 중심이나 일부 수치 인용"
#     }
# ]
```

---

## 3. `labeling/gemini/` — Gemini LLM

### 파일 구성

| 파일 | 역할 |
|---|---|
| `gemini_client.py` | API 키·모델 로테이션, RPD(일일 요청 한도) 추적, JSON 파싱 |
| `omission_labeler.py` | omission_risk 3분류 (low/mid/high) |

---

### 3.1 `gemini_client.py` — Gemini API 클라이언트

#### 핵심 기능

| 기능 | 설명 |
|---|---|
| RPD 추적 | 하루 요청 수 카운트. `rpd_remaining()` 으로 잔여량 조회 가능 |
| 키 로테이션 | `GEMINI_API_KEYS`에서 복수 키. 한도 초과 시 자동 전환 |
| 모델 폴백 | `GEMINI_PRO_MODELS`에서 순서대로 폴백 |

#### `.env` 설정

```env
GEMINI_API_KEYS=AIza_key1,AIza_key2
GEMINI_PRO_MODELS=gemini-2.5-pro,gemini-2.0-flash
```

#### 주요 함수 / 전역 변수

| 이름 | 타입 | 설명 |
|---|---|---|
| `call_gemini_json(prompt, system, model, max_tokens)` | `dict` | 프롬프트 → JSON dict |
| `rpd_remaining()` | `int` | 오늘 남은 요청 수 |
| `last_call_info` | `dict` | 마지막 호출 메타 |
| `rotation_log` | `list` | 키/모델 로테이션 이벤트 |

---

### 3.2 `omission_labeler.py` — omission_risk 분류

동종 클러스터 기사 엔티티 목록과 비교하여 누락 위험도를 평가합니다.
**하루 100~200건 RPD 제한** 때문에 `--max_gemini` 옵션으로 사용량 제어.

#### 레이블 정의

| 레이블 | DB 저장값 | 의미 |
|---|---|---|
| `low` | -1 | 핵심 사실·주체·수치 대부분 포함 |
| `mid` | 0 | 일부 중요 관점이나 수치 누락 |
| `high` | 1 | 핵심 사실 또는 주요 당사자 관점 대부분 누락 |

#### 배치 처리 파라미터

| 배치 크기 | 본문 스니펫 길이 |
|---|---|
| 1~5개 | 400자 |
| 6~10개 | 250자 |
| 11개+ | 150자 |

#### 주요 함수

| 함수명 | 인수 | 반환 타입 | 설명 |
|---|---|---|---|
| `label_omission_batch(articles, cluster_entities)` | `List[dict], List[str]` | `List[dict]` | 배치 누락도 평가 |

#### 입출력 예시

```python
# Input
articles = [
    {
        "id": "123",
        "title": "의대 증원 강행",
        "body_snippet": "정부가 2000명 증원을 강행하기로 했다..."
    }
]
cluster_entities = ["교육부", "의협", "대학", "학생", "의사"]

# Output
label_omission_batch(articles, cluster_entities)
# → [
#     {
#         "omission_risk":   "mid",
#         "omission_reason": "의협의 반발 입장이 언급되지 않았고, 학생 피해 관련 수치 누락"
#     }
# ]
```

---

## 4. 배치 실패 폴백 (`run_labeling.py`)

배치 전체 JSON 파싱 실패 시 **1개씩 재시도** 로직이 모든 Groq/Gemini 블록에 공통 적용:

```python
try:
    results = label_frame_logic_batch(batch_structs)   # 배치 시도
except Exception as e:
    results = []
    for single_struct in batch_structs:
        try:
            res = label_frame_logic_batch([single_struct])
            results.append(res[0])
        except Exception as e2:
            results.append({"frame": None, "logic": None,
                            "frame_reason": "api_error", "logic_reason": "api_error"})
```

---

## 5. 모델 선택 가이드

| 모델 | 장점 | 단점 |
|---|---|---|
| `llama-3.3-70b-versatile` | 한국어 안정적, frame/logic 정확도 높음 | RPM 제한 낮음 |
| `qwen/qwen3-32b` | RPM 여유, 수치·논리 분석 강함 | `<think>` 블록 생성 (자동 제거됨) |
| `llama-4-scout-17b` | 빠름 | 한국어 오자 발생 → 숫자 출력 방식으로 대응 |
| `gemini-2.5-pro` | omission 분석 품질 최상 | RPD 제한 엄격 (하루 ~100건) |
| `gemini-2.0-flash` | RPD 여유, 빠름 | omission 품질 다소 낮음 |
