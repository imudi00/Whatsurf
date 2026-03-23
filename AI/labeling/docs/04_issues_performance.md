# Issue Log 04 — 성능 및 처리량 문제

## 이슈 1: 기본 batch_size=5 → RPM 과다 소비

### 증상
기사 50개를 처리할 때 Groq API 호출 횟수:
- frame/logic: 10회
- bias: 10회
- **총 20회** → 30 RPM 한도에서 약 40초/batch → 총 13분 이상

### 원인
`batch_size=5` 기본값. 기사 5개당 API 1회 호출.

### 해결
**`batch_size=20`** 으로 기본값 변경.

| batch_size | frame+bias 호출 | 시간(50기사) |
|---|---|---|
| 5 (기존) | 20회 | ~13분 |
| **20 (신규)** | **6회** | **~4분** |

배치 크기가 커질수록 기사 내용을 압축해야 토큰 초과 방지.
각 labeler에 동적 압축 적용:

```python
def _truncate_params(n: int) -> dict:
    if n <= 5:   return {"lead": 200, "judgment": 8,  "quotes": 3}
    elif n <= 15: return {"lead": 120, "judgment": 5,  "quotes": 2}
    else:         return {"lead": 70,  "judgment": 3,  "quotes": 1}
```

| 배치 크기 | frame 리드 | bias 본문 | omission 본문 |
|---|---|---|---|
| ≤5   | 200자 | 300자 | 400자 |
| 6~15 | 120자 | 180자 | 250자 |
| 16+  | 70자  | 100자 | 150자 |

---

## 이슈 2: Groq와 Gemini가 순차 실행 → 총 시간 합산

### 증상
기존 실행 순서:
```
로컬(2min) → frame/logic(15min) → bias(15min) → omission(10min)
                                              = 총 42분
```

Groq와 Gemini는 **완전히 다른 API** → 동시에 실행 불가능한 이유가 없음.

### 해결 — ThreadPoolExecutor로 병렬 실행

**핵심 설계 원칙**:
- Groq 스레드와 Gemini 스레드가 **서로 다른 dict**에 결과 저장 (공유 상태 없음)
- 전처리(`structs`, `cluster_entities`) 는 메인 스레드에서 먼저 수행
- 두 스레드 완료 후 메인 스레드에서 병합 + 저장

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    future_groq   = executor.submit(_run_groq_block, ...)
    future_gemini = executor.submit(_run_gemini_block, ...)
    groq_result   = future_groq.result()    # 완료 대기
    gemini_result = future_gemini.result()  # 완료 대기

# 메인 스레드에서 병합
for aid, data in groq_result.items():
    label_map[aid].update(data)
for aid, data in gemini_result.items():
    label_map[aid].update(data)
```

**병렬 실행 후 시간**:
```
로컬(2min) → [Groq(30min) ‖ Gemini(10min)] → 저장
                     max(30, 10) = 30min
              총 32분  (기존 42분 대비 24% 단축)
```

실제 효과는 omission 처리량이 많을수록 커짐.

**실행 옵션**:
```bash
python run_labeling.py --keyword 의대정원          # 병렬 (기본)
python run_labeling.py --keyword 의대정원 --no_parallel  # 순차 (디버깅)
```

---

## 이슈 3: 댓글 처리가 기사 순서대로 직렬 → 느림

### 증상
댓글은 로컬 모델(API 없음)이라 병렬화가 자유롭지만,
기존 코드는 기사 하나씩 순서대로 처리.

### 해결
`ThreadPoolExecutor`로 기사 단위 병렬 처리.

```python
with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
    futures = {
        pool.submit(_process_one_article_comments, row, ...): str(row["id"])
        for row in active_rows
    }
    for future in as_completed(futures):
        path = future.result()
```

기본 `--cmt_workers 4`. 기사가 많을수록 효과 큼.

**스레드 안전성**:
- `report.record_comment_labels()` 호출 시 `threading.Lock` 적용
- 각 기사의 결과 파일이 다른 경로이므로 파일 쓰기 충돌 없음

---

## 이슈 4: 20,000~30,000개 대용량 처리 전략

### 문제

| API | 무료 RPD | batch_size=20 기준 필요 호출 수 | 소요 일수 |
|---|---|---|---|
| Groq (2키) | 28,800/일 | 20k÷20 = 1,000회 | **1일** |
| Gemini Pro (2키) | 200/일 | 20k÷15 = 1,333회 | **7일** |

**Groq는 문제없음. Gemini가 병목.**

### 전략

#### A. 매일 나눠서 처리 (`--offset`)

```bash
# Day 1: 0~200
python run_labeling.py --keyword X --max_gemini 200

# Day 2: 200~400
python run_labeling.py --keyword X --max_gemini 200 --offset 200

# Day N: (N-1)*200 ~ N*200
python run_labeling.py --keyword X --max_gemini 200 --offset $((N*200))
```

#### B. Gemini Flash로 전환

```env
GEMINI_PRO_MODELS=gemini-2.5-pro,gemini-2.0-flash
```

Flash는 무료 티어 RPD가 약 1,500/일로 Pro 대비 15배 높음.
Pro 소진 시 자동으로 Flash로 로테이션.

#### C. omission 샘플링

연구 목적상 전체 기사의 omission을 구하는 대신,
키워드당 대표 기사 샘플(예: 500개)만 omission 처리.

```bash
python run_labeling.py --keyword X --features frame bias loaded_words stance
python run_labeling.py --keyword X --features omission --limit 500
```

#### D. Groq 기반 omission 대체

Groq는 충분한 RPD가 있으므로, omission도 Groq Llama로 처리하는 labeler 추가.
Gemini 대비 품질은 다소 낮지만 처리량 문제 해결.

---

## 요약: 현재 권장 처리 속도

기사 1,000개 기준 (batch_size=20, 병렬):

| 단계 | 시간 |
|---|---|
| 로컬 (stance + loaded_words) | ~2분 |
| Groq (frame + bias) 병렬 | ~30분 |
| Gemini (omission) 병렬 | max_gemini=200 제한 기준 ~30분 |
| 댓글 로컬 (cmt_workers=4) | 댓글 수에 따라 |

**Gemini 처리량이 핵심 병목.** 키를 여러 개 확보하거나 Flash 모델 사용 권장.
