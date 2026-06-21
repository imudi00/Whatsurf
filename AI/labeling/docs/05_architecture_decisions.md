# Architecture Decisions — 설계 의사결정 기록

## 1. 피처별 처리 분담 결정

### 배경
각 피처마다 복잡도·비용·속도가 달라 처리 방식 선택이 필요.

### 결정

| 피처 | 처리 방식 | 근거 |
|---|---|---|
| stance | 로컬 HuggingFace | API 비용 없음. KLUE-BERT로 충분. |
| art_words / cmt_words | 로컬 규칙 기반 | 사전 기반으로 빠르고 설명 가능. API 불필요. |
| frame / logic | Groq 70B | 맥락 이해 필요. 높은 추론 능력 요구. |
| bias_x / bias_y | Groq 70B | 정치 성향 판단은 대형 모델 권장. |
| omission_risk | Gemini Pro | 클러스터 비교 필요 → 긴 컨텍스트. Gemini가 적합. |
| cmt_emotion | 로컬 HuggingFace | 댓글 수가 많음 → API 비용 부담. 로컬로 처리. |

### 신뢰도 우선 원칙
> 속도나 비용보다 신뢰도를 우선. API 실패 시 `null` 저장 (결측값으로 처리).
> `"unknown"` 같은 임의 기본값은 연구 결과를 오염시킬 수 있어 금지.

---

## 2. loaded_words 설계: 사전 기반 vs 모델 기반

### 배경
처음에는 NER 모델로 편향 span을 추출하려 했으나,
한국어 편향 표현으로 파인튜닝된 공개 모델이 없음.

### 결정: 사전 기반 + Tier 구조

- **Tier-1 BIAS_WORDS**: 명시적 정치 편향어. 연구에서 가장 신뢰할 수 있는 신호.
- **Tier-2 OPINION_WORDS**: 강한 평가·감정어. 특히 댓글에서 유용.
- **Tier-3 통계 추출**: Fallback. 텍스트에서 두드러진 단어 직접 추출.

**`tier` 값을 출력에 포함**해 연구자가 신뢰도를 직접 판단 가능.

```json
{"art_words": ["선동"], "tier": 1}   // 정치 편향어 — 신뢰도 높음
{"art_words": ["최악", "ㅋㅋ"], "tier": 2}  // 평가/감정어 — 중간
{"art_words": ["경제성장"], "tier": 3}  // 통계 추출 — 참고용
```

### 최소 2개 보장 원칙
빈 배열 반환 시 연구에서 "단어 없음"과 "감지 실패"가 구분되지 않음.
→ 항상 최소 2개 반환. 단 `tier` 값으로 신뢰도 구분.

---

## 3. 댓글 감정 레이블 설계

### 배경
연구자가 원하는 레이블: Plutchik 8감정 + 중립 (9종).
현재 모델이 지원하는 레이블: 모델 훈련 데이터셋 기준 (7종 이하).

### 결정: 현 모델 유지 + raw_label 보존

```json
{
  "cmt_emotion": {
    "label": "anger",
    "raw_label": "분노",    // 모델 원본 출력 보존
    "intensity": 0.87,
    "probs": {"anger": 0.87, "sadness": 0.08, ...}
  }
}
```

- `primary_emotion`: 영어 정규화 레이블 (코드에서 사용)
- `raw_label`: 모델 원본 한국어 레이블 (연구 분석 시 참고)

**향후**: Plutchik 8감정 파인튜닝 모델 또는 LLM 분류로 교체 시 `EMOTION_MODEL` 환경변수만 변경.

---

## 4. ERD 기반 저장 구조 결정

### 배경
초기에는 기사와 댓글의 라벨을 하나의 JSON에 모두 저장.
Supabase ERD 분석 후 분리 저장으로 변경.

### ERD 기반 구조

```
article_features 테이블:
  art_words    JSONB    ← 기사 편향 단어

comments 테이블:
  cmt_emotion  JSONB    ← 댓글별 감정 (label, intensity, probs)
  cmt_words    JSONB    ← 댓글별 편향/의견 단어
```

### 파일 저장 구조

```
label_results/
  {keyword}_{id}_labeled.json   ← 기사 라벨 (art_words, frame, bias 등)
  {keyword}_{id}_comments.json  ← 댓글 라벨 (cmt_emotion, cmt_words)
  reports/
    run_report_{ts}.json        ← 전체 실행 요약
    label_dist_{ts}.json        ← 레이블 분포
    feature_timing_{ts}.json    ← 피처별 소요 시간
    sample_labels_{ts}.json     ← 샘플 출력 (10개)
    comments_stats_{ts}.json    ← 댓글 통계
```

---

## 5. 병렬 처리 설계

### 배경
Groq와 Gemini는 독립된 API. 순차 실행할 이유가 없음.

### 결정: 스레드별 독립 결과 dict → 메인 스레드 병합

```
# ❌ 위험한 방식 (두 스레드가 같은 dict 수정)
Thread-1: label_map[aid]["frame"] = ...
Thread-2: label_map[aid]["omission_risk"] = ...

# ✅ 채택 방식 (각 스레드가 자체 dict 반환)
Thread-1: groq_result[aid] = {"frame": ..., "bias_x": ...}
Thread-2: gemini_result[aid] = {"omission_risk": ...}
# 메인에서 병합
label_map[aid].update(groq_result[aid])
label_map[aid].update(gemini_result[aid])
```

### 병렬 모드에서의 트레이드오프

| 항목 | 병렬 (기본) | 순차 (`--no_parallel`) |
|---|---|---|
| 속도 | 빠름 | 느림 |
| 중간 저장 | 없음 (완료 후 1회) | 배치마다 즉시 저장 |
| 크래시 시 복구 | 어려움 | 완료 배치는 보존 |
| 디버깅 | 어려움 | 쉬움 |

**대용량 처리 시 권장**: `--no_parallel --offset N` 으로 구간별 순차 처리.

---

## 6. API 키 관리 원칙

### 다중 키 로테이션 설계

```
세션 시작: _cur_key=0, _cur_model=0

호출 실패 시:
  rate_limit (RPM) → MAX_RETRIES 재시도 → 실패 시 _cur_key += 1
  rpd_exhausted    → 즉시 _cur_key += 1
  model_error      → 즉시 _cur_model += 1, _cur_key = 0
  기타 서버 오류   → _cur_key += 1 (다음 키에서 재시도)

세션 종료 조건:
  _cur_model >= len(models) → RuntimeError
```

**원칙**: 한 번 전진한 키/모델은 같은 세션에서 다시 사용하지 않음 (0→1→0 방지).

### RPD 카운터 파일

```json
// .groq_usage.json
{
  "key_0": {"date": "2025-01-01", "count": 450},
  "key_1": {"date": "2025-01-01", "count": 230}
}
```

하루가 지나면 카운트 자동 리셋.

---

## 7. 연구 리포트 설계

### 목적
라벨링 결과의 신뢰도를 검증하고, 연구에 활용 가능한 통계를 자동 생성.

### 5개 파일 출력

| 파일 | 내용 | 용도 |
|---|---|---|
| `run_report_{ts}.json` | 전체 실행 정보 | 실험 재현 |
| `label_dist_{ts}.json` | 레이블 분포 + 상위 편향 단어 | 데이터 분포 확인 |
| `feature_timing_{ts}.json` | 피처별 배치 소요 시간 | 비용 분석 |
| `sample_labels_{ts}.json` | 피처별 첫 10개 결과 | 품질 육안 검토 |
| `comments_stats_{ts}.json` | 댓글 감정/단어 분포 | 댓글 분석 |

### 피처 타이밍 구조

```json
{
  "frame_logic": {
    "mean_sec": 4.2,
    "throughput_per_min": 285,
    "batches": [{"n": 20, "elapsed": 4.1}, ...]
  }
}
```
