# 모듈 상세 문서 — `labeling/experiments/`

> **피처 실험 프레임워크**
> 규칙 기반 피처 파라미터 최적화 + 모델 기반 피처 객관적 성능 평가 환경.

---

## 1. 모듈 개요

| 하위 디렉토리 | 역할 |
|---|---|
| `utils/` | 공통 메트릭·데이터 로더·결과 리포트 |
| `rule_based/` | body_depth / loaded_words 파라미터 탐색 |
| `model_eval/` | frame/logic/bias/stance/omission 성능 평가 |
| `data/` | Ground truth JSONL 저장 위치 |
| `results/` | 실험 결과 JSON 자동 저장 (gitignore 권장) |

---

## 2. `utils/` — 공통 유틸

### 2.1 `metrics.py` — 평가 메트릭 모음

#### 연속값 메트릭

| 함수 | 반환 | 용도 |
|---|---|---|
| `pearson_r(y_true, y_pred)` | `float` | 피어슨 상관계수 (-1~1) |
| `spearman_rho(y_true, y_pred)` | `float` | 스피어만 순위 상관 (-1~1) |
| `mae(y_true, y_pred)` | `float` | 평균 절대 오차 |
| `rmse(y_true, y_pred)` | `float` | 루트 평균 제곱 오차 |
| `score_distribution(scores)` | `dict` | n/min/max/mean/std/low%/mid%/high% |

#### 범주형 메트릭

| 함수 | 반환 | 용도 |
|---|---|---|
| `accuracy(y_true, y_pred)` | `float` | 정확도 |
| `f1_macro(y_true, y_pred, labels)` | `float` | macro-average F1 |
| `f1_per_class(y_true, y_pred, labels)` | `dict` | 클래스별 precision/recall/F1/support |
| `cohen_kappa(y_true, y_pred)` | `float` | Cohen's Kappa (일치도) |
| `confusion_matrix_str(y_true, y_pred, labels)` | `str` | 텍스트 혼동행렬 |

#### 리스트형 메트릭

| 함수 | 반환 | 용도 |
|---|---|---|
| `precision_at_k(y_true_sets, y_pred_lists, k)` | `float` | Precision@K |
| `recall_at_k(y_true_sets, y_pred_lists, k)` | `float` | Recall@K |
| `f1_at_k(y_true_sets, y_pred_lists, k)` | `float` | F1@K |
| `jaccard(y_true_sets, y_pred_sets)` | `float` | Jaccard 유사도 |

---

### 2.2 `data_loader.py` — 데이터 로드 / 템플릿 생성

| 함수 | 설명 |
|---|---|
| `load_ground_truth(path)` | JSONL 파일 로드. `#` 주석 줄 무시 |
| `load_labeled_dir(directory)` | `*_labeled.json` 파일 전체 로드 |
| `make_annotation_template(records, out_path, sample_n)` | labeled.json → 어노테이션 템플릿 생성 |
| `extract_pairs(gt_records, pred_records, feature, pred_feature)` | id 매칭 후 (y_true, y_pred, ids) 반환 |
| `split_ground_truth(records, test_ratio=0.2)` | train/test 분할 |

#### Ground Truth JSONL 형식

```jsonl
{"id": "123",
 "text": "기사 제목\n\n기사 본문...",
 "labels": {
   "body_depth":     0.75,
   "frame":          2,
   "logic":          6,
   "bias_x":        -0.35,
   "bias_y":         0.20,
   "omission_risk": 0,
   "stance_score":  -0.45,
   "loaded_words":  ["독재", "재앙"]
 }}
```

---

### 2.3 `report.py` — 실험 결과 관리

#### `ExperimentResult` 데이터클래스

| 필드 | 타입 | 설명 |
|---|---|---|
| `name` | `str` | 실험 식별자 (config 이름 또는 모델명) |
| `feature` | `str` | 평가 피처명 |
| `config` | `dict` | 실험 파라미터 |
| `metrics` | `dict` | 메트릭명 → 값 |
| `n_samples` | `int` | 평가 샘플 수 |
| `timestamp` | `str` | 실행 시각 (YYYYMMDD_HHMMSS) |

| 함수 | 설명 |
|---|---|
| `save_result(result, out_dir)` | JSON으로 저장 (예측값은 별도 JSONL) |
| `load_results(out_dir, feature)` | 저장된 결과 로드 |
| `compare_results(results, sort_by)` | 텍스트 비교 테이블 반환 |
| `best_result(results, metric)` | 특정 메트릭 기준 최고 결과 반환 |

#### 저장 파일 형식

```json
{
  "name":      "diversity_quote_heavy",
  "feature":   "body_depth",
  "config":    {"w_diversity": 0.35, "w_quotes": 0.30, "length_soft_max": 2000, ...},
  "metrics":   {"pearson": 0.812, "spearman": 0.803, "mae": 0.091},
  "n_samples": 50,
  "timestamp": "20260322_143022"
}
```

---

## 3. `rule_based/` — 파라미터 튜닝

### 3.1 `body_depth_tuner.py`

#### 실행 모드

| 모드 | 정답 필요 | 설명 |
|---|---|---|
| `distribution` | ❌ | 9개 preset의 점수 분포·baseline 대비 Spearman ρ 비교 |
| `grid` | ✅ | 9 preset + 랜덤 서치(n_random회). Pearson/Spearman/MAE 기준 최적 config 선택 |
| `preset` | ❌ | 특정 config 파라미터 확인 출력 |

#### Preset Config 목록

| 이름 | 특징 |
|---|---|
| `baseline` | 현재 기본값 |
| `diversity_quote_heavy` | 어휘 다양성(0.35) + 인용(0.30) 중시 |
| `length_numeric_heavy` | 길이(0.35) + 수치(0.30) 중시 |
| `structure_heavy` | 문단 구조(0.30) 중시 |
| `uniform` | 5개 지표 균등(0.20) |
| `short_article_focus` | LENGTH_SOFT_MAX=1000 (짧은 기사 차별화) |
| `long_article_focus` | LENGTH_SOFT_MAX=4000 (긴 기사 차별화) |
| `quote_sensitive` | QUOTE_RATIO_REF=0.06 (인용 민감도 증가) |
| `numeric_sensitive` | NUMERIC_DENSITY_REF=0.03 (수치 민감도 증가) |

#### 탐색 공간 (Grid/Random Search)

```python
GRID_SPACE = {
    "w_length":           [0.10, 0.20, 0.30],
    "w_diversity":        [0.15, 0.25, 0.35],
    "w_quotes":           [0.15, 0.20, 0.30],
    "w_numerics":         [0.10, 0.20, 0.30],
    "length_soft_max":    [1000, 2000, 3000],
    "diversity_ttr_ref":  [0.45, 0.55, 0.65],
    "quote_ratio_ref":    [0.06, 0.12, 0.18],
    "numeric_density_ref":[0.03, 0.06, 0.09],
    "structure_para_max": [5, 8, 12],
}
# w_structure = 1.0 - (w_length + w_diversity + w_quotes + w_numerics)
```

---

### 3.2 `loaded_words_tuner.py`

#### 실행 모드

| 모드 | 정답 필요 | 설명 |
|---|---|---|
| `distribution` | ❌ | 8개 preset의 편향 기사 비율·평균 단어 수·density 분포 비교 |
| `word_freq` | ❌ | BIAS_WORDS / OPINION_WORDS 실제 감지 빈도 분석 |
| `eval` | ✅ | Precision@K / Recall@K / F1@K / Jaccard + is_biased Accuracy |

#### Preset Config 목록

| 이름 | 특징 |
|---|---|
| `baseline` | 현재 기본값 (tier1_2_3, max=10) |
| `tier1_only` | BIAS_WORDS만 사용 (정밀도 우선) |
| `tier1_2` | BIAS + OPINION (재현율 향상) |
| `max20` | 최대 20개 추출 |
| `max5` | 핵심 5개만 |
| `no_guarantee` | min_guaranteed=0 (없으면 빈 리스트) |
| `denoise_common` | 흔한 오탐 단어 제거 ("완전","절대","당연히" 등) |
| `politics_enhanced` | 정치 편향어 추가 ("내로남불","정치쇼" 등) |

---

## 4. `model_eval/` — 모델 성능 평가

### 4.1 `eval_runner.py` — 공통 BaseEvaluator

모든 feature별 Evaluator의 베이스 클래스.

| 메서드 | 설명 |
|---|---|
| `evaluate(ground_truth_path, predictions_dir, run_name)` | ground_truth + labeled.json 로드 → id 매칭 → 메트릭 계산 → 저장 |
| `compute_metrics(y_true, y_pred)` | 서브클래스에서 오버라이드 필수 |

단독 실행:
```bash
python labeling/experiments/model_eval/eval_runner.py \
    --ground_truth experiments/data/ground_truth.jsonl \
    --predictions  labeling/output \
    --feature      frame
```

---

### 4.2 `frame_logic_eval.py`

| Evaluator 클래스 | 담당 피처 | 주요 메트릭 |
|---|---|---|
| `FrameEvaluator` | `frame` (1~7) | Accuracy / F1-macro / Cohen's Kappa / 혼동행렬 / 클래스별 F1 |
| `LogicEvaluator` | `logic` (1~6) | 동일 |

모델 비교:
```python
compare_models(
    ground_truth_path="experiments/data/ground_truth.jsonl",
    prediction_dirs={
        "llama3.3_70b": "output/llama3",
        "qwen3_32b":    "output/qwen3",
    },
    out_dir="experiments/results",
)
```

---

### 4.3 `bias_eval.py`

| 메트릭 | 설명 |
|---|---|
| `bias_x_pearson` / `bias_x_spearman` / `bias_x_mae` | bias_x 연속값 평가 |
| `bias_y_pearson` / `bias_y_spearman` / `bias_y_mae` | bias_y 연속값 평가 |
| `dir_accuracy` | 진보/중립/보수 3분류 정확도 (threshold ±0.2) |
| `dir_f1_macro` | 방향 분류 F1 |
| `quadrant_accuracy` | (x부호 × y부호) 4개 사분면 정확도 |

---

### 4.4 `stance_omission_eval.py`

| Evaluator | 피처 | 주요 메트릭 |
|---|---|---|
| `StanceEvaluator` | `stance_score` | Pearson·Spearman·MAE + 논조(비판/중립/우호) 3분류 Accuracy·F1·Kappa |
| `OmissionEvaluator` | `omission_risk` (-1/0/1) | Accuracy / F1-macro / Cohen's Kappa / 혼동행렬 |

---

## 5. 사용 흐름 요약

```
[1] 정답 수집
    python -m labeling.experiments.utils.data_loader \
        --labeled_dir labeling/output \
        --out experiments/data/ground_truth.jsonl \
        --sample 50
    (→ 파일 열어서 틀린 레이블만 교정)

[2] Rule-based 튜닝 (정답 없이도 즉시 실행 가능)
    body_depth_tuner.py --mode distribution    # 분포 확인
    body_depth_tuner.py --mode grid            # 최적 파라미터 탐색 (정답 필요)
    loaded_words_tuner.py --mode word_freq     # 사전 활용도 분석

[3] 모델 평가
    frame_logic_eval.py                        # Accuracy/F1/Kappa
    bias_eval.py                               # Pearson + 사분면 정확도
    stance_omission_eval.py                    # 논조/누락도 평가

[4] 최적 결과 적용
    body_depth.py 상단 WEIGHTS/상수 교체
    loaded_words.py BIAS_WORDS 사전 수정
```
