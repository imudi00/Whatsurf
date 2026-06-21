# AI 통합 파이프라인

`queries` 테이블의 `query_id` 하나를 받아 **Clustering → Feature Map → Timeline** 세 단계를 순서대로 실행하고, 결과를 Supabase2에 저장합니다.

---

## 실행 방법

### 전체 실행 (기본)

```bash
# 프로젝트 루트(40.WhatSurf/)에서 실행
python -m AI.pipeline.run_pipeline --query_id 4
```

세 단계 모두 순서대로 실행됩니다.

---

### 기능별 실행

#### Clustering만

```bash
python -m AI.pipeline.run_pipeline --query_id 4 --steps clustering
```

- SBERT 임베딩 → UMAP 차원 축소 → HDBSCAN 클러스터링
- LLM으로 각 클러스터 제목/요약 생성
- `article_clusters` 테이블 저장 + `articles.cluster_label` 업데이트

#### Feature Map만

```bash
python -m AI.pipeline.run_pipeline --query_id 4 --steps feature_map
```

#### Timeline만

```bash
python -m AI.pipeline.run_pipeline --query_id 4 --steps timeline
```

#### 여러 단계 조합

```bash
python -m AI.pipeline.run_pipeline --query_id 4 --steps clustering timeline
```

---

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--query_id` | 필수 | `queries` 테이블 PK |
| `--steps` | 전체 | 실행할 단계 (`clustering` `feature_map` `timeline`) |
| `--features` | all | feature_map에서 실행할 피처 (`frame` `logic` `bias` `omission` `stance`) |
| `--top_timepoints` | 8 | 타임라인에 저장할 최대 날짜 수 |
| `--limit` | 0 (전체) | feature_map 최대 기사 수 |
| `--no_resume` | — | feature_map 중단 재개 비활성화 |
| `--skip_comments` | — | feature_map 댓글 라벨링 건너뜀 |
| `--out_dir` | `./label_results` | feature_map 라벨 JSON 저장 경로 |

#### 옵션 사용 예시

```bash
# 피처 일부만 지정
python -m AI.pipeline.run_pipeline --query_id 4 --steps feature_map --features frame logic stance

# 타임라인 날짜 수 조정
python -m AI.pipeline.run_pipeline --query_id 4 --steps timeline --top_timepoints 5

# feature_map 처음부터 다시 실행 (재개 비활성화)
python -m AI.pipeline.run_pipeline --query_id 4 --steps feature_map --no_resume
```

---

## 테스트

```bash
# DB 연결 + 모듈 임포트 + 기사 로드만 확인 (쓰기 없음)
python AI/pipeline/test_pipeline.py --query_id 4 --dry_run

# clustering + timeline 실제 실행 (DB에 저장)
python AI/pipeline/test_pipeline.py --query_id 4
```

테스트 항목:
1. 모듈 임포트 체인 확인
2. Supabase2 테이블 연결 확인
3. 기사 로드 확인
4. 실제 실행 (dry_run 아닐 때)

---

## DB 저장 테이블

| 단계 | 저장 테이블 | 주요 컬럼 |
|------|------------|----------|
| Clustering | `article_clusters` | `query_id`, `cluster_title`, `cluster_summary`, `rep_article_id` |
| Clustering | `articles` | `cluster_label` 업데이트 |
| Feature Map | `article_features` | `article_id`, `frame`, `logic`, `bias_x`, `omission_risk`, `stance_score` 등 |
| Timeline | `queries_timeline` | `query_id`, `timeline_date`, `top_art` |

---

## 폴더 구조

```
AI/pipeline/
├── run_pipeline.py       # 메인 진입점 (CLI)
├── test_pipeline.py      # 테스트 스크립트
├── steps/
│   ├── step_clustering.py   # Clustering 단계
│   ├── step_feature_map.py  # Feature Map 단계 (run_labeling.py 위임)
│   └── step_timeline.py     # Timeline 단계
└── db/
    ├── save_clusters.py     # article_clusters 저장
    └── save_timeline.py     # queries_timeline 저장
```
