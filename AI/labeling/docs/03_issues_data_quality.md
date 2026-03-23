# Issue Log 03 — 데이터 품질 문제

## 이슈 1: `loaded_words`가 항상 빈 배열 `[]`

### 증상
라벨링 결과에서 `art_words: []`, `cmt_words: []` 가 대부분.
이 피처가 의사결정에 가장 영향이 큰 단어 토큰을 뽑는 핵심 피처.

### 원인 1: BIAS_WORDS가 영어 단어로 구성됨 (초기 버전)

초기 `loaded_words.py`의 BIAS_WORDS:
```python
BIAS_WORDS = ["propaganda", "radical", "extreme", "fake news", ...]  # 영어!
```
한국어 텍스트에서 당연히 매칭 없음.

### 원인 2: `\b` 정규식 한국어 미지원

```python
# ❌ 한국어에서 작동 안 함
re.findall(r'\b선동\b', text)

# ✅ 단순 포함 검색 사용
word in text
```

`\b`는 ASCII 단어 경계 기준. 한국어 유니코드 문자는 모두 `\W`로 인식.

### 원인 3: 단어 목록이 너무 좁음 (정치 편향어만)

BIAS_WORDS를 한국어로 교체해도, 기사/댓글 대부분이 정치적 극단 표현을 쓰지 않음.
특히 **댓글**은 비공식 구어체 → 정치 편향어 거의 없음.

### 해결 — 2티어 사전 + 3단계 Fallback

#### 사전 구조

```python
# Tier-1: 정치/이념 편향어 (강한 신호)
BIAS_WORDS = [
    "극우", "극좌", "빨갱이", "종북", "독재", "선동",
    "조작", "가짜뉴스", "척결", "타도", "매국", ...
]

# Tier-2: 강한 평가/감정어 + 인터넷 신조어 (폭넓은 신호)
OPINION_WORDS = [
    "최악", "황당", "어이없", "기가막", "뻔뻔", "미쳤",
    "헛소리", "대박", "레알", "노답", "개판",
    "ㅋㅋ", "ㄷㄷ", "ㅠㅠ", "ㄹㅇ", "ㅂㄷ", ...
]
```

#### 3단계 Fallback 로직

```
Tier-1 BIAS_WORDS 검색
  ↓ 2개 미만이면
Tier-2 OPINION_WORDS + 자음반복 패턴(ㅋㅋ, ㄷㄷ 등) 검색
  ↓ 아직 2개 미만이면
Tier-3 텍스트에서 직접 salient 단어 추출
  (자음반복 > 긴 단어 > 반복 등장 순으로 점수 부여)
```

**결과에 `tier` 필드 포함** → 연구 시 신뢰도 구분 가능:

```json
{
  "art_words": ["선동", "가짜뉴스"],
  "loaded_word_density": 0.023,
  "is_biased": true,
  "tier": 1
}
```

| tier | 의미 | 신뢰도 |
|---|---|---|
| 1 | 정치/이념 편향어 직접 감지 | 높음 |
| 2 | 강한 평가/감정 표현 감지 | 중간 |
| 3 | 텍스트 통계적 추출 | 낮음 (참고용) |

---

## 이슈 2: 댓글 감정 분류가 Plutchik 8감정 체계와 맞지 않음

### 증상
연구자가 원하는 레이블:
```
분노, 역겨움, 두려움, 기대, 슬픔, 놀람, 예측, 믿음, 중립
```
(Plutchik의 감정의 바퀴 8종 + 중립)

실제 출력은 다른 레이블 체계로 나옴.

### 원인
사용 중인 모델 `hun3359/klue-bert-base-sentiment`의 레이블셋이 다름.
이 모델은 Plutchik 8감정으로 훈련된 것이 아니며, **`기대`·`믿음`·`예측` 3가지는 지원 안 함**.

또한 코드에서 모델 출력을 영어로 매핑(`기쁨→joy`)하여 원하는 한국어 레이블과 괴리.

### 핵심
> Plutchik 8감정 + 중립을 완전히 지원하려면 **그 레이블로 훈련된 모델**이 필요.
> 현재 모델로는 `기대`, `믿음`, `예측`은 나올 수 없음.

### 대응 방안

1. **현 모델 유지**: 모델이 지원하는 레이블 그대로 사용. `기대`·`믿음` 제외.
2. **모델 교체**: Plutchik 8감정으로 파인튜닝된 한국어 모델 사용.
3. **LLM 기반 분류**: 댓글을 Groq/Gemini에 보내어 9종 중 하나로 명시적 분류.
   (단: API 비용 발생, 댓글 수가 많으면 RPM 부담)

`EMOTION_MODEL=<모델명>` 환경변수로 모델 교체 가능.

---

## 이슈 3: `research_report.py` — `ValueError: could not convert string to float: 'unknown'`

### 증상
```
File "research_report.py", line 205, in build
    [float(v) for v in vals if v is not None]
ValueError: could not convert string to float: 'unknown'
```

### 원인
API 실패 시 기본값으로 `"unknown"` (문자열)을 저장했는데,
`bias_x`, `bias_y`는 수치형 피처라 `float()` 변환 시도 → 실패.

```python
# ❌ 잘못된 fallback
results = [{"bias_x": "unknown", "bias_y": "unknown"} ...]
```

### 해결
API 실패 fallback을 `"unknown"` → `None` 으로 변경.

```python
# ✅ None으로 저장 → float 변환 시 필터링됨
results = [{"bias_x": None, "bias_y": None, "bias_reason": "api_error"} ...]

# research_report.py 에서
def _to_float(v):
    try:    return float(v)
    except: return None   # None이면 통계에서 제외

label_stats[feat] = self._stats(
    [x for x in (_to_float(v) for v in vals) if x is not None]
)
```

---

## 이슈 4: Supabase 페이지네이션 누락 — 1000개 이상 데이터 미수집

### 증상
Supabase 기본 한도는 1000행. 그 이상의 데이터는 수집 안 됨.

### 해결

```python
PAGE_SIZE = 1000
rows: list = []
offset = 0
while True:
    fetch = PAGE_SIZE if limit == 0 else min(PAGE_SIZE, limit - len(rows))
    resp = (
        supabase.table("ai_test")
        .select("id, title, body, comments")
        .eq("keyword", keyword)
        .range(offset, offset + fetch - 1)
        .execute()
    )
    page = resp.data or []
    rows.extend(page)
    if len(page) < fetch or (limit > 0 and len(rows) >= limit):
        break
    offset += fetch
```

`limit=0` → 전체 데이터 수집.

---

## 이슈 5: 에러 발생 시 완료된 배치 결과 유실

### 증상
50개 기사 처리 중 30번째에서 오류 → 1~29번 결과가 메모리에만 있어 모두 유실.

### 해결 — 배치 완료 직후 즉시 저장

```python
def _save_partial(label_map, article_ids, out_dir, keyword):
    os.makedirs(out_dir, exist_ok=True)
    for aid in article_ids:
        path = os.path.join(out_dir, f"{keyword}_{aid}_labeled.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(label_map[aid], f, ensure_ascii=False, indent=2)

# 각 배치 루프 끝에서 호출
_save_partial(label_map, batch_ids, out_dir, keyword)
```

배치 단위로 저장 → 중간 실패해도 완료된 배치는 보존.
