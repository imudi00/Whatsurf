# Issue Log 08 — 디렉토리 리팩토링 & 실험 프레임워크 구축

---

## 이슈 1: `feature_map/` ↔ `labeling/` 이중 구조 유지 부담

### 증상
- `labeling/run_labeling.py`에서 `feature_map.stance.src.feature_map.preprocessor` 처럼 4단계 중첩 경로로 import
- `labeling/local/loaded_words_labeler.py`가 sys.path를 런타임에 조작하여 `feature_map/` 내부 파일을 직접 찾는 해킹 코드 존재 (30줄)
- `feature_map/context/src/feature_map/body_depth.py`와 `labeling/local/body_depth.py` 두 버전이 공존하며 혼란

### 원인
- 초기 설계 시 `feature_map/`과 `labeling/`을 독립 패키지로 분리했으나, 라벨링 파이프라인이 고도화되면서 `labeling/`이 `feature_map/`의 모듈을 직접 소비하는 구조로 수렴
- `feature_map/*/src/feature_map/` 중첩 패키지 구조 (Python 패키징 관례)가 오히려 import 경로를 불필요하게 길게 만듦

### 해결
`labeling/features/` 패키지를 신설하고 필요한 파일을 이동:

```
feature_map/stance/src/feature_map/preprocessor.py  → labeling/features/preprocessor.py
feature_map/stance/src/feature_map/text_utils.py    → labeling/features/text_utils.py
feature_map/emotion/src/feature_map/loaded_words.py → labeling/features/loaded_words.py
feature_map/context/src/feature_map/keyword_extractor.py → labeling/features/keyword_extractor.py
feature_map/emotion/src/feature_map/bias_vector.py  → labeling/features/bias_vector.py
feature_map/context/src/feature_map/omission_risk.py → labeling/features/omission_risk.py
```

`run_labeling.py` import 수정:
```python
# Before
from feature_map.stance.src.feature_map.preprocessor import build_article_struct
from feature_map.context.src.feature_map.keyword_extractor import extract_features

# After
from labeling.features.preprocessor import build_article_struct
from labeling.features.keyword_extractor import extract_features
```

`feature_map/__init__.py`에 DEPRECATED 안내 추가. `feature_map/` 디렉토리는 삭제하지 않고 유지(이전 파이프라인 참조용).

---

## 이슈 2: `loaded_words_labeler.py` sys.path 해킹

### 증상
```python
# 기존 코드 (30줄 분량)
for _p in [Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[3]]:
    _lw = _p / "feature_map" / "emotion" / "src" / "feature_map"
    if (_lw / "loaded_words.py").exists():
        sys.path.insert(0, str(_lw))
        from loaded_words import detect_loaded_words, ...
```
런타임에 파일 시스템을 탐색하고 sys.path를 변조하는 방식 — 실행 위치가 바뀌면 동작 불보장, IDE 정적 분석 불가.

### 원인
`labeling/`과 `feature_map/`이 별도 패키지였을 때, 패키지 경계를 우회하기 위해 임시방편으로 작성

### 해결
`labeling/features/loaded_words.py`로 이동 후 단순 import로 교체:
```python
# After (3줄)
from labeling.features.loaded_words import (
    detect_loaded_words, detect_opinion_words,
    detect_informal_patterns, BIAS_WORDS, OPINION_WORDS,
)
```
불필요해진 `import sys`, `from pathlib import Path`도 제거.

---

## 이슈 3: 규칙 기반 피처의 파라미터 최적화 방법 부재

### 증상
- `body_depth`의 WEIGHTS, LENGTH_SOFT_MAX 등 상수값이 임의로 설정되어 있으며, 성능을 객관적으로 비교할 방법이 없음
- `loaded_words` 사전에 어떤 단어가 실제로 감지에 기여하는지, 노이즈 단어는 무엇인지 파악 불가

### 해결
`labeling/experiments/rule_based/` 하위에 튜너 구축:

**`body_depth_tuner.py`**
- 9개 사전 정의 preset config (`baseline`, `diversity_quote_heavy`, `length_numeric_heavy`, …)
- 랜덤 서치 (기본 100회): WEIGHTS 5개 + 정규화 상수 5개 탐색
- 모드 1 (정답 없음): 점수 분포 분석 (mean/std/low%/mid%/high%), baseline 대비 Spearman ρ 출력
- 모드 2 (정답 있음): Pearson/Spearman/MAE 기준 최적 config 선택, 상위 5개 결과 저장

**`loaded_words_tuner.py`**
- 8개 preset config (`baseline`, `tier1_only`, `tier1_2`, `max5/20`, `no_guarantee`, `denoise_common`, `politics_enhanced`)
- `word_freq` 모드: BIAS_WORDS / OPINION_WORDS별 실제 감지 빈도 분석 → 노이즈 단어 / 미사용 단어 식별
- 정답 있을 때: Precision@K / Recall@K / F1@K / Jaccard + is_biased 분류 Accuracy

---

## 이슈 4: 모델 기반 피처의 객관적 성능 평가 체계 부재

### 증상
- frame/logic이 올바른지, bias_x/y가 실제 편향과 얼마나 일치하는지 수치로 알 수 없음
- 모델을 교체(llama → qwen 등)했을 때 성능 변화를 정량적으로 비교하는 방법이 없음

### 해결
`labeling/experiments/model_eval/` + 공통 유틸 구축:

| 파일 | 담당 피처 | 핵심 메트릭 |
|---|---|---|
| `eval_runner.py` | 공통 BaseEvaluator | — |
| `frame_logic_eval.py` | frame, logic | Accuracy / F1-macro / Cohen's Kappa / 혼동행렬 |
| `bias_eval.py` | bias_x, bias_y | Pearson·Spearman·MAE + 방향 정확도 + 사분면 정확도 |
| `stance_omission_eval.py` | stance_score, omission_risk | Pearson·Spearman + 논조 3분류 F1 + Kappa |

**정답 수집 플로우**:
```bash
# 1. labeled.json → 어노테이션 템플릿 자동 생성 (LLM 예측값 pre-fill)
python -m labeling.experiments.utils.data_loader \
    --labeled_dir labeling/output --out experiments/data/ground_truth.jsonl --sample 50
# 2. 파일 열어서 틀린 레이블만 수동 교정
# 3. 평가 실행
python labeling/experiments/model_eval/frame_logic_eval.py \
    --ground_truth experiments/data/ground_truth.jsonl \
    --predictions labeling/output --run_name llama3.3_70b
```

**모델 비교 실험**: `--compare_dir results/` 옵션으로 이전 결과와 자동 테이블 비교 가능.

---

## 관련 파일 변경 요약

| 파일 | 변경 유형 | 내용 |
|---|---|---|
| `labeling/features/__init__.py` | 신규 | 패키지 초기화 + 모듈 설명 |
| `labeling/features/preprocessor.py` | 신규 (이동) | `build_article_struct` |
| `labeling/features/text_utils.py` | 신규 (이동) | LLM 입력용 텍스트 유틸 |
| `labeling/features/loaded_words.py` | 신규 (이동) | BIAS_WORDS / OPINION_WORDS 사전 |
| `labeling/features/keyword_extractor.py` | 신규 (이동) | KPF-BERT NER `extract_features` |
| `labeling/features/bias_vector.py` | 신규 (이동) | 규칙 기반 편향 벡터 |
| `labeling/features/omission_risk.py` | 신규 (이동) | 규칙 기반 누락 위험도 |
| `labeling/local/loaded_words_labeler.py` | 수정 | sys.path 해킹 → 정상 import |
| `labeling/run_labeling.py` | 수정 | feature_map.* → labeling.features.* |
| `feature_map/__init__.py` | 수정 | DEPRECATED 안내 주석 추가 |
| `labeling/experiments/` | 신규 | 전체 실험 프레임워크 |
| `AI/README.md` | 신규 | 전체 실행 가이드 |
