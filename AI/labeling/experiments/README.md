# 피처 실험 프레임워크

피처의 **정확도 향상**(rule-based 튜닝)과 **객관적 성능 평가**(모델 기반)를 위한 실험 환경.

---

## 디렉토리 구조

```
experiments/
├── data/
│   └── ground_truth_template.jsonl   # 정답 레이블 템플릿 (작성 가이드 포함)
│
├── utils/
│   ├── metrics.py        # 평가 메트릭 (Pearson, F1, Cohen's Kappa, Precision@K 등)
│   ├── data_loader.py    # JSONL 로드 / labeled.json 로드 / 어노테이션 템플릿 생성
│   └── report.py         # 실험 결과 저장·비교 출력
│
├── rule_based/
│   ├── body_depth_tuner.py      # body_depth 파라미터 Grid Search
│   └── loaded_words_tuner.py    # loaded_words 사전/임계값 튜닝
│
├── model_eval/
│   ├── eval_runner.py            # 공통 평가 러너 (BaseEvaluator)
│   ├── frame_logic_eval.py       # frame / logic 정확도 평가
│   ├── bias_eval.py              # bias_x / bias_y 평가
│   └── stance_omission_eval.py   # stance_score / omission_risk 평가
│
└── results/              # (자동 생성) 실험 결과 JSON
```

---

## 사용 흐름

### Phase 1 — 정답 수집 (한 번만)

```bash
# labeled.json 결과물 → 어노테이션 템플릿 생성 (50개 샘플)
python -m labeling.experiments.utils.data_loader \
    --labeled_dir  AI/labeling/output \
    --out          AI/labeling/experiments/data/ground_truth.jsonl \
    --sample 50

# → 파일을 열어 labels 필드 값을 직접 수정 (틀린 것만 교정)
# → 저장 후 Phase 2~3 진행
```

**정답 파일 형식** (`ground_truth.jsonl`):
```json
{"id": "123", "text": "제목\n\n본문...",
 "labels": {"body_depth": 0.75, "frame": 2, "logic": 6,
             "bias_x": -0.1, "bias_y": 0.2, "omission_risk": 0,
             "stance_score": -0.3, "loaded_words": ["극우"]}}
```

---

### Phase 2 — Rule-Based 피처 튜닝

#### body_depth 튜닝

```bash
# 모드 1: 정답 없이 분포 분석 (어느 config가 고른 분포를 갖는지)
python AI/labeling/experiments/rule_based/body_depth_tuner.py \
    --labeled_dir AI/labeling/output \
    --mode distribution

# 모드 2: Grid Search (정답 필요)
python AI/labeling/experiments/rule_based/body_depth_tuner.py \
    --ground_truth AI/labeling/experiments/data/ground_truth.jsonl \
    --mode grid \
    --metric pearson \
    --n_random 200

# 모드 3: 특정 preset 파라미터 확인
python AI/labeling/experiments/rule_based/body_depth_tuner.py \
    --mode preset \
    --config_name diversity_quote_heavy
```

**사전 정의 preset 목록:**

| config_name | 특징 |
|---|---|
| `baseline` | 현재 기본값 |
| `diversity_quote_heavy` | 어휘 다양성·인용 중시 |
| `length_numeric_heavy` | 길이·수치 중시 (보도자료형) |
| `structure_heavy` | 문단 구조 중시 (기획기사) |
| `uniform` | 5개 지표 균등 가중치 |
| `short_article_focus` | 짧은 기사 차별화 |
| `long_article_focus` | 긴 기사 차별화 |
| `quote_sensitive` | 인용 많을수록 후한 점수 |
| `numeric_sensitive` | 수치 많을수록 후한 점수 |

#### loaded_words 튜닝

```bash
# 분포 분석 (bias 기사 비율, 평균 단어 수)
python AI/labeling/experiments/rule_based/loaded_words_tuner.py \
    --labeled_dir AI/labeling/output \
    --mode distribution

# 단어 빈도 분석 (어떤 단어가 자주/드물게 감지되는지)
python AI/labeling/experiments/rule_based/loaded_words_tuner.py \
    --labeled_dir AI/labeling/output \
    --mode word_freq

# 사전 평가 (정답 필요)
python AI/labeling/experiments/rule_based/loaded_words_tuner.py \
    --ground_truth AI/labeling/experiments/data/ground_truth.jsonl \
    --mode eval
```

**사전 정의 preset 목록:**

| config_name | 특징 |
|---|---|
| `baseline` | 현재 기본값 |
| `tier1_only` | BIAS_WORDS만 (정밀도 우선) |
| `tier1_2` | BIAS + OPINION (재현율 향상) |
| `max20` | 최대 단어 20개 추출 |
| `max5` | 핵심 5개만 |
| `no_guarantee` | 진짜 없으면 빈 리스트 |
| `denoise_common` | 흔한 오탐 단어 제거 |
| `politics_enhanced` | 정치 편향어 추가 |

---

### Phase 3 — 모델 성능 평가

```bash
# frame / logic 평가
python AI/labeling/experiments/model_eval/frame_logic_eval.py \
    --ground_truth AI/labeling/experiments/data/ground_truth.jsonl \
    --predictions  AI/labeling/output \
    --run_name     llama3.3_70b

# bias_x / bias_y 평가
python AI/labeling/experiments/model_eval/bias_eval.py \
    --ground_truth AI/labeling/experiments/data/ground_truth.jsonl \
    --predictions  AI/labeling/output

# stance_score 평가
python AI/labeling/experiments/model_eval/stance_omission_eval.py \
    --ground_truth AI/labeling/experiments/data/ground_truth.jsonl \
    --predictions  AI/labeling/output \
    --feature stance

# omission_risk 평가
python AI/labeling/experiments/model_eval/stance_omission_eval.py \
    --ground_truth AI/labeling/experiments/data/ground_truth.jsonl \
    --predictions  AI/labeling/output \
    --feature omission

# 이전 실험과 비교 (results/ 디렉토리 내 이전 결과 자동 불러옴)
python AI/labeling/experiments/model_eval/frame_logic_eval.py \
    --ground_truth AI/labeling/experiments/data/ground_truth.jsonl \
    --predictions  AI/labeling/output \
    --run_name     qwen3_32b \
    --compare_dir  AI/labeling/experiments/results
```

#### 빠른 단일 피처 평가

```bash
python AI/labeling/experiments/model_eval/eval_runner.py \
    --ground_truth AI/labeling/experiments/data/ground_truth.jsonl \
    --predictions  AI/labeling/output \
    --feature      frame
```

---

## 평가 메트릭 요약

| 피처 | 메트릭 |
|---|---|
| `body_depth` | Pearson r, Spearman ρ, MAE |
| `frame` | Accuracy, F1-macro, Cohen's Kappa, 혼동행렬 |
| `logic` | Accuracy, F1-macro, Cohen's Kappa, 혼동행렬 |
| `bias_x / bias_y` | Pearson r, Spearman ρ, MAE, 방향 정확도, 사분면 정확도 |
| `stance_score` | Pearson r, Spearman ρ, MAE, 논조 3분류 F1 |
| `omission_risk` | Accuracy, F1-macro, Cohen's Kappa |
| `loaded_words` | Precision@K, Recall@K, F1@K, Jaccard |

---

## 결과 파일 형식

`results/` 디렉토리에 자동 저장:
```json
{
  "name":      "diversity_quote_heavy",
  "feature":   "body_depth",
  "config":    {"w_diversity": 0.35, "w_quotes": 0.30, ...},
  "metrics":   {"pearson": 0.812, "spearman": 0.803, "mae": 0.091},
  "n_samples": 50,
  "timestamp": "20260322_143022"
}
```

---

## body_depth 튜닝 후 적용 방법

grid search에서 찾은 최적 파라미터를 `labeling/local/body_depth.py`에 반영:

```python
# labeling/local/body_depth.py 상단 상수를 수정
WEIGHTS = {
    "length":    0.10,   # ← 여기를 실험 결과 최적값으로 교체
    "diversity": 0.35,
    "quotes":    0.30,
    "numerics":  0.15,
    "structure": 0.10,
}
LENGTH_SOFT_MAX     = 2000    # ← 여기도
DIVERSITY_TTR_REF   = 0.55
QUOTE_RATIO_REF     = 0.06   # ← 여기도
```

## loaded_words 튜닝 후 적용 방법

`word_freq` 분석 결과를 보고 `labeling/features/loaded_words.py`의 `BIAS_WORDS` / `OPINION_WORDS` 수정:
- 감지 0건 단어 → 제거 또는 유지
- 자주 감지되는 유용 단어 → 상위에 배치
- 새로운 사회적 키워드 → 추가
