# Issue Log 02 — API 클라이언트 (Groq / Gemini) 문제

## 배경

Groq(frame/logic/bias)과 Gemini(omission)는 무료 티어 기준
RPM·RPD 한도가 낮아 대용량 처리 시 잦은 오류 발생.

---

## 이슈 1: `.env` 다중 키 파싱 실패 — 따옴표 형식 오류

### 증상
```
RuntimeError: Gemini Pro: 모든 키(1개) 한도 소진
RPD 현황: {'key_0': {'remaining': 100, 'used': 0, 'limit': 100}}
# .env에 키 2개 설정했는데 1개로 인식
```

### 원인
`.env`에서 각 키를 따옴표로 감싸면 `python-dotenv`가 따옴표 자체를 값의 일부로 파싱.

```env
# ❌ 잘못된 형식
GEMINI_API_KEYS='AIzaSyB...','AIzaSyC...'
# → 파싱 결과: "'AIzaSyB...','AIzaSyC...'" (문자열 1개)

# ✅ 올바른 형식
GEMINI_API_KEYS=AIzaSyB...,AIzaSyC...
# → 파싱 결과: split(",") → 2개 키
```

### 해결
`_get_api_keys()`에서 `.strip("'\"")` 처리를 추가해 실수 방지.

```python
keys = [k.strip().strip("'\"") for k in multi.split(",") if k.strip().strip("'\"")]
```

---

## 이슈 2: 키 로테이션 시 역방향 복귀 (0→1→0)

### 증상
key_0 한도 초과 → key_1으로 전환 → 다시 key_0으로 복귀
동일 오류 반복.

### 원인
로테이션 인덱스를 매 호출마다 0부터 순환(cycle) 방식으로 구현했기 때문.

### 해결
**모듈 레벨 전역 변수**로 `_cur_key`, `_cur_model`을 관리.
한 번 전진하면 세션 내에서 절대 이전으로 돌아가지 않음.

```python
_cur_model: int = 0
_cur_key:   int = 0

# 키 소진 시
_cur_key += 1   # 절대 감소하지 않음
# 모델 소진 시
_cur_model += 1
_cur_key = 0    # 다음 모델은 key_0부터
```

---

## 이슈 3: Groq 모델 로테이션이 llama-3.1에서 멈춤

### 증상
`GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-70b-versatile,llama-3.1-8b-instant,...`
설정 시 3.1만 계속 사용하다 멈춤.

### 원인 분석
1. **RPM 소진 패턴**: 3.3-70b가 RPM 한도(30 RPM)에 빠르게 걸려 key_0 → key_1 소진 후 3.1로 전환. 사용자 눈에는 "3.3은 안 씀"처럼 보임.
2. **모델 에러 미감지**: deprecated 모델(`mixtral-8x7b-32768` 등)이 404를 반환해도 rate_limit 패턴(`429`)과 달라 `raise`로 바로 전파됨 → 다음 모델로 전환 안 됨.
3. **다른 에러도 같은 (model, key)로 재시도**: 일시적 서버 오류(500)가 발생하면 해당 (model, key)를 계속 재시도.

### 해결 — 에러 유형별 로테이션

```python
def _is_rate_limit(e):
    return isinstance(e, _GroqRateLimit) or "429" in str(e) or "rate_limit" in str(e).lower()

def _is_model_error(e):
    return (
        isinstance(e, _GroqNotFound) or
        "model_not_found" in str(e).lower() or
        "deprecated" in str(e).lower() or
        "decommissioned" in str(e).lower()
    )

# 처리 분기
except Exception as e:
    if _is_rate_limit(e):
        # MAX_RETRIES 후 → 다음 키
    elif _is_model_error(e):
        # 즉시 → 다음 모델
    else:
        # 기타 서버 오류 → 다음 키 (포기하지 않고 계속 시도)
```

---

## 이슈 4: Gemini RPD 한도가 20,000개 처리에 매우 부족

### 분석

| 티어 | 모델 | RPM | RPD |
|---|---|---|---|
| 무료 | Gemini 2.5 Pro | 2 | 100 |
| 무료 | Gemini 2.0 Flash | 15 | 1,500 |

**20,000개 기사 omission 처리 시**:
- batch_size=15 → 1,333 Gemini 호출 필요
- 키 2개 × 100 RPD = **하루 200개** → **완료까지 약 100일**

### 해결책

1. **`--max_gemini N`** 인자로 하루 처리량 제한 (기본 200)
2. **`--offset N`** 인자로 다음날 이어서 처리
3. `.env`에 `GEMINI_PRO_MODELS=gemini-2.5-pro,gemini-2.0-flash` 추가 → Flash로 자동 로테이션
4. 키를 여러 개 확보해 `GEMINI_API_KEYS=key1,key2,key3,...`

```bash
# 1일차: 0~200 처리
python run_labeling.py --keyword 의대정원 --max_gemini 200

# 2일차: 200~400 처리
python run_labeling.py --keyword 의대정원 --max_gemini 200 --offset 200
```

---

## 이슈 5: JSON 파싱 실패로 배치 전체 건너뜀

### 증상
```
[경고] frame/logic 배치 실패 (건너뜀): Expecting ',' delimiter: line 10 column 49
[경고] bias 배치 실패 (건너뜀): Expecting ':' delimiter: line 4 column 45
```

빈도가 너무 높아 결과의 상당 부분이 `null`.

### 원인
20개 기사를 한 번에 처리하는 긴 JSON 응답에서 LLM이 흔하게 저지르는 실수:

| 오류 종류 | 예시 |
|---|---|
| trailing comma | `{"a": 1,}` |
| 객체 간 쉼표 누락 | `} {` → `},{` 필요 |
| 전각 따옴표 | `"key"` → `"key"` |
| 앞뒤 자연어 혼입 | `"결과입니다: [...]"` |
| 콜론 누락 | `{"key" "value"}` |

### 해결 — 4단계 강건 파싱 + API 재시도

```python
def _fix_json_string(text):
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r',\s*([\]\}])', r'\1', text)          # trailing comma
    text = re.sub(r'\}\s*\n\s*\{', '},\n{', text)        # 누락 쉼표
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # 전각 따옴표
    return text

def _robust_json_parse(raw):
    # 전략 1: 전처리 후 직접 파싱
    # 전략 2: 배열 [ ... ] 추출
    # 전략 3: 객체 { ... } 추출 후 배열 감싸기
    # 전략 4: 줄 단위 { } 수집 (NDJSON-like)
    ...

def call_groq_json(prompt, _retry_json=2):
    for attempt in range(1, _retry_json + 2):
        raw = call_groq(retry_prompt if attempt > 1 else prompt)
        try:
            return _robust_json_parse(raw)
        except json.JSONDecodeError:
            continue   # API 재호출 (최대 2회)
    raise ...
```

**처리 흐름**:
```
LLM 응답
  → _fix_json_string (전처리: trailing comma, 쉼표 누락, 따옴표 등)
  → 전략 1~4 순차 시도
  → 실패 시 API 재호출 최대 2회 (프롬프트에 "유효한 JSON만 출력" 명시)
  → 모두 실패 시에만 배치 건너뜀
```
