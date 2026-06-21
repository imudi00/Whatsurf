# Issue Log 07 — 모델 견고성 및 배치 처리 안정화

## 배경

Groq API의 일일 RPD 한도 소진 이후 `llama-4-scout-17b-16e-instruct` 등 대체 모델로 전환하면서
발생한 모델별 출력 품질 차이 및 배치 전체 실패 문제를 기록.

---

## 이슈 1: llama-4-scout 한글 오자 출력

### 증상
```
[frame_labeler] 첫 번째 결과 샘플:
  frame='사간 원瞋 짓중', logic='사안/정분 발전'
```
`'사건 원인 집중'` → `'사간 원瞋 짓중'`
`'갈등/대립 강조'` → `'곱또/대력 초강'`

모든 frame/logic 값이 FRAME_LABELS / LOGIC_LABELS와 매칭 실패 → None.

### 원인
llama-4-scout 계열 모델은 한글 텍스트 레이블을 그대로 출력할 때
특수문자 혼입, 음절 순서 오류 등 오자(오탈자)를 빈번히 생성.
한국어 훈련 데이터 비중이 낮은 경량 다국어 모델의 구조적 한계.

### 시도 1: 단어 단순화
각 레이블을 더 짧게 바꾸는 방안 검토.
→ 레이블 자체가 이미 짧은 수준이며, 레이블 의미 손실 우려. 채택 안 함.

### 해결: 숫자 출력 방식으로 전환 (근본 해결)
frame/logic을 텍스트 레이블 대신 **정수 번호(1~7 / 1~6)** 로만 출력하도록 프롬프트 변경.
숫자는 어떤 모델도 안정적으로 출력 가능.

```python
# frame_labeler.py 프롬프트 변경
frame_legend = "\n".join(f"  {i+1}={l}" for i, l in enumerate(FRAME_LABELS))
logic_legend = "\n".join(f"  {i+1}={l}" for i, l in enumerate(LOGIC_LABELS))

prompt = (
    f"아래 {len(structs)}개 기사를 각각 frame/logic 번호로 분류하세요.\n\n"
    f"frame (1~7):\n{frame_legend}\n\n"
    f"logic (1~6):\n{logic_legend}\n\n"
    ...
    "반드시 frame과 logic을 정수(숫자)로만 출력하세요. JSON 배열만 출력:\n"
    '[{"idx":0,"frame":1,"frame_reason":"한줄","logic":3,"logic_reason":"한줄"},...]'
)
```

```python
# few-shot 예시도 숫자 형식으로 변경
_FEW_SHOT = """예시1) 헤드라인: 의대 정원 확대, 의사협회 강력 반발 → frame: 2, logic: 1
예시2) 헤드라인: GDP 0.3% 하락 전망, 전문가들 경고 → frame: 4, logic: 2
예시3) 헤드라인: 성범죄 피해자 2차 가해 근절 촉구 → frame: 7, logic: 3"""
```

---

## 이슈 2: `_match_label()` 한글 오자 부분 복구 (보조 수단)

### 배경
숫자 출력 전환 이전, 오자가 있어도 부분 복구하기 위한 fuzzy matching 도입.

### 해결
`_match_label()` 함수에 4단계 매칭 로직 구현.

```python
_FUZZY_THRESHOLD = 0.40  # 유사도 임계값

def _match_label(val: str, labels: list[str], field: str = "") -> str | None:
    """
    1순위: exact match
    2순위: 숫자 index (1-based)
    3순위: partial match (부분문자열 포함)
    4순위: 문자 유사도 (SequenceMatcher ≥ 0.40) — 한글 오자 모델 대응
    5순위: None
    """
    if not val:
        return None
    val_str = str(val).strip()

    # 1. exact
    if val_str in labels:
        return val_str

    # 2. 숫자 index (1-based)
    try:
        idx = int(val_str) - 1
        if 0 <= idx < len(labels):
            return labels[idx]
    except ValueError:
        pass

    # 3. partial
    for label in labels:
        if val_str in label or label in val_str:
            return label

    # 4. 유사도 (오자 보정)
    best_label = max(labels, key=lambda l: SequenceMatcher(None, val_str, l).ratio())
    best_ratio = SequenceMatcher(None, val_str, best_label).ratio()
    if best_ratio >= _FUZZY_THRESHOLD:
        print(f"  [frame_labeler] {field} 유사도 매핑 ({best_ratio:.2f}): {val_str!r} → {best_label!r}")
        return best_label

    return None
```

---

## 이슈 3: `<think>` 블록으로 인한 JSON 파싱 실패

### 증상
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```
또는 LLM 응답 전체가 JSON이 아닌 텍스트로 처리됨.

### 원인
Qwen3, llama-4-scout 등 reasoning 계열 모델은 응답 앞에
`<think>...</think>` 블록을 포함해 내부 추론 과정을 출력.
기존 `_fix_json_string()` 이 이를 처리하지 못해 JSON 추출 실패.

### 해결
`groq_client.py`의 `_fix_json_string()`에 `<think>` 블록 제거 로직 추가.

```python
def _fix_json_string(text: str) -> str:
    # <think>...</think> 블록 제거 (Qwen3, llama-4-scout reasoning 모델 대응)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()
    # ... 기존 JSON 추출 로직
```

---

## 이슈 4: 배치 실패 시 전체 api_error

### 증상
```json
[
  {"frame": null, "logic": null, "frame_reason": "api_error", "logic_reason": "api_error"},
  {"frame": null, "logic": null, "frame_reason": "api_error", "logic_reason": "api_error"},
  ...
]
```
배치 내 1개 기사의 문제(긴 본문, 특수문자 등)로 전체 배치가 실패하고
모든 결과가 `api_error`로 채워짐.

### 원인
`label_frame_logic_batch()` 호출 시 JSON 파싱 오류가 발생하면
`except` 블록에서 배치 전체를 빈 결과로 채우는 구조.

### 해결
배치 실패 시 1개씩 재시도하는 폴백 로직 추가 (frame/logic, bias, omission 공통 적용).

```python
try:
    results = label_frame_logic_batch(batch_structs)
except Exception as e:
    print(f"  [경고] frame/logic 배치 실패 → 1개씩 재시도: {e}")
    results = []
    for j, single_struct in enumerate(batch_structs):
        try:
            res = label_frame_logic_batch([single_struct])
            results.append(res[0])
        except Exception as e2:
            print(f"    [경고] 단건 실패 idx={j}: {e2}")
            results.append({
                "frame": None, "logic": None,
                "frame_reason": "api_error", "logic_reason": "api_error"
            })
```

---

## 이슈 5: cluster_entity_lists O(n²) 행 hang

### 증상
기사 100개 배치 처리 시 수 시간 후에도 완료되지 않음.
NER 호출 카운트가 비정상적으로 많음.

### 원인
`cluster_entity_lists` 생성 로직에서 기사 i의 클러스터 엔티티를
"i를 제외한 나머지 전체" 기사로부터 매번 NER 추출.
→ 기사 n개면 NER 호출 횟수 = n × (n-1) = O(n²)

n=100일 때 NER 호출 = 9,900회.

### 해결
각 기사별 NER을 1회만 실행하고, 전체에서 30% 이상 출현한 엔티티를
클러스터 공통 엔티티로 재사용. O(n) 변환.

```python
# 각 기사 엔티티를 1회만 추출
all_entity_sets = [set(extract_entities(a["body"])) for a in articles]

# 전체 대비 30% 이상 출현 엔티티 = 클러스터 공통 엔티티
from collections import Counter
entity_counter = Counter(e for es in all_entity_sets for e in es)
common_entities = [e for e, cnt in entity_counter.items()
                   if cnt / len(articles) >= 0.30]

# 각 기사의 클러스터 엔티티: 공통 엔티티 (자신 제외)
cluster_entity_lists = [
    [e for e in common_entities if e not in es]
    for es in all_entity_sets
]
```
