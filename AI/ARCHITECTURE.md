# WhatSurf AI — 구조 및 플로우 문서

> 최종 업데이트: 2026-04-24

---

## 목차

1. [전체 폴더 구조](#1-전체-폴더-구조)
2. [아키텍처 개요](#2-아키텍처-개요)
3. [엔드-투-엔드 실행 플로우](#3-엔드-투-엔드-실행-플로우)
4. [모듈별 상세](#4-모듈별-상세)
   - 4.1 [pipeline — 통합 오케스트레이터](#41-pipeline--통합-오케스트레이터)
   - 4.2 [labeling — 피처 라벨링](#42-labeling--피처-라벨링)
   - 4.3 [clustering — 기사 군집화](#43-clustering--기사-군집화)
   - 4.4 [timeline — 타임라인 분석](#44-timeline--타임라인-분석)
   - 4.5 [source — 데이터 로더](#45-source--데이터-로더)
   - 4.6 [llm — LLM 유틸리티 (레거시)](#46-llm--llm-유틸리티-레거시)
5. [DB 스키마 & 업로드 플로우](#5-db-스키마--업로드-플로우)
6. [API 클라이언트 구조](#6-api-클라이언트-구조)
7. [피처별 모델 정리](#7-피처별-모델-정리)
8. [환경변수 레퍼런스](#8-환경변수-레퍼런스)
9. [CLI 진입점 정리](#9-cli-진입점-정리)

---

## 1. 전체 폴더 구조

```
AI/
├── ARCHITECTURE.md              ← 이 문서
│
├── pipeline/                    # 전체 파이프라인 오케스트레이터
│   ├── run_pipeline.py          # 진입점: clustering → labeling → timeline
│   ├── steps/
│   │   ├── step_clustering.py   # clustering 단계 래퍼
│   │   ├── step_feature_map.py  # labeling 단계 래퍼 (subprocess 호출)
│   │   └── step_timeline.py     # timeline 단계 래퍼
│   └── db/
│       ├── save_clusters.py     # article_clusters 테이블 저장
│       └── save_timeline.py     # queries_timeline 테이블 저장
│
├── labeling/                    # 피처 추출 & 라벨링 (핵심)
│   ├── run_labeling.py          # 진입점: 기사별 전체 피처 추출
│   ├── upload_to_db.py          # label_results/ → Supabase2 upsert
│   ├── auto_labeler.py          # (레거시) CSV 기반 단순 라벨링
│   ├── label_stats.py           # 결과 통계 시각화
│   ├── research_report.py       # 실행 리포트 생성 (5개 JSON)
│   │
│   ├── api_client_base.py       # Groq/Gemini 공통 유틸리티
│   │
│   ├── features/                # 텍스트 전처리 & 피처 추출 (로컬, 모델 미사용)
│   │   ├── preprocessor.py      # build_article_struct(): 기사 구조화
│   │   ├── keyword_extractor.py # KPF-BERT NER + 통계 피처
│   │   ├── loaded_words.py      # BIAS_WORDS / OPINION_WORDS 사전
│   │   └── text_utils.py        # 정규식 헬퍼 함수들
│   │
│   ├── local/                   # 로컬 HuggingFace 모델 라벨러
│   │   ├── stance_labeler.py    # snunlp/KR-FinBert-SC → stance_score
│   │   ├── body_depth.py        # 수식 기반 기사 깊이 점수
│   │   ├── loaded_words_labeler.py # 3-tier 편향어 추출
│   │   └── emotion_labeler.py   # hun3359/klue-bert-base-sentiment (댓글)
│   │
│   ├── groq/                    # Groq API (Llama 3.3 70B)
│   │   ├── groq_client.py       # 키 로테이션 + JSON 파싱
│   │   ├── frame_labeler.py     # frame(7종) + logic(6종) 분류
│   │   └── bias_labeler.py      # bias_x / bias_y 수치 산출
│   │
│   └── gemini/                  # Gemini Pro API
│       ├── gemini_client.py     # 키 로테이션 + JSON 파싱
│       └── omission_labeler.py  # omission_risk 평가
│
├── clustering/                  # 기사 군집화 (SBERT → UMAP → HDBSCAN)
│   └── src/clustering/
│       ├── service_pipeline.py  # run_service_clustering()
│       ├── embedding.py         # SBERT 임베딩
│       ├── reducer.py           # UMAP 차원 축소
│       ├── clustering.py        # HDBSCAN 군집화
│       ├── data_loader.py       # Supabase2에서 기사 로드
│       ├── evaluation.py        # Silhouette / Davies-Bouldin 평가
│       └── incremental.py       # 점진적 군집화
│
├── timeline/                    # 타임라인 분석
│   └── src/timeline/
│       ├── timepoint_rank.py    # 중요 시점 순위 산출 (TTP/ETP)
│       ├── burst.py             # 버스트 감지
│       ├── summarize_extractive.py # 핵심 요약 추출
│       ├── io_utils.py          # 파일 I/O
│       └── text_utils.py        # 텍스트 유틸
│
├── source/                      # DB 연결 & 데이터 로더
│   ├── config/
│   │   ├── supabase_client.py   # Supabase #1 (ai_test 테이블)
│   │   └── supabase2_client.py  # Supabase #2 (articles / comments / queries)
│   └── data_loader.py           # 범용 로더
│
└── llm/                         # Gemini Flash 클라이언트 (레거시 auto_labeler 전용)
    ├── llm_client.py            # call_llm(), call_llm_json()
    └── llm_batch.py             # 배치 처리
```

---

## 2. 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                   Supabase2 (운영 DB)                        │
│  articles │ comments │ queries │ article_clusters │          │
│  article_features │ queries_timeline                         │
└─────────────────────────────────────────────────────────────┘
          ↑ upsert          ↑ upsert         ↑ upsert
          │                 │                │
┌─────────┴──────┐  ┌───────┴──────┐  ┌─────┴──────────┐
│  labeling/     │  │  clustering/ │  │  timeline/     │
│  upload_to_db  │  │  step_       │  │  step_         │
│                │  │  clustering  │  │  timeline      │
└────────────────┘  └──────────────┘  └────────────────┘
        ↑                  ↑                  ↑
        └──────────────────┴──────────────────┘
                           │
                  pipeline/run_pipeline.py
                  (통합 오케스트레이터)
                           │
              ┌────────────┴────────────┐
              │  Supabase2 articles 로드 │
              └────────────────────────┘
```

### 피처 추출 레이어 구조

```
기사 텍스트
    │
    ├─[전처리]──────────────────────────────────────────────────
    │  features/preprocessor.py  → headline, lead, quotes,
    │                               judgment_words, sources
    │  features/keyword_extractor.py → KPF-BERT NER + 통계
    │
    ├─[로컬 모델]──────────────────────────────────────────────
    │  local/stance_labeler.py      → stance_score [-1, +1]
    │  local/body_depth.py          → body_depth [0, 1]
    │  local/loaded_words_labeler.py → art_words, density
    │
    ├─[Groq API — Llama 3.3 70B]───────────────────────────────
    │  groq/frame_labeler.py        → frame (7종), logic (6종)
    │  groq/bias_labeler.py         → bias_x, bias_y [-1, +1]
    │
    ├─[Gemini Pro API]─────────────────────────────────────────
    │  gemini/omission_labeler.py   → omission_risk (low/mid/high)
    │
    └─[댓글 전용 — 로컬]────────────────────────────────────────
       local/emotion_labeler.py     → cmt_emotion
       local/loaded_words_labeler.py → cmt_words
```

---

## 3. 엔드-투-엔드 실행 플로우

### 전체 파이프라인 실행

```bash
python -m AI.pipeline.run_pipeline --query_id 15 --steps clustering feature_map timeline
```

```
[1] 데이터 로드
    Supabase2.articles (query_id=15) → rows: [{id, title, body_text, published_at, ...}]

[2] step_clustering  (SBERT → UMAP → HDBSCAN → LLM 요약)
    ├─ embedding.py: sentence-transformers → 벡터
    ├─ reducer.py: UMAP 2차원 축소
    ├─ clustering.py: HDBSCAN (min_cluster_size=max(3, n//5))
    ├─ llm_client.call_llm_json: 클러스터별 제목 + 요약 생성
    ├─ save_clusters.save_cluster() → article_clusters 삽입
    └─ save_clusters.update_article_cluster_label() → articles.cluster_label 업데이트

[3] step_feature_map  (subprocess: run_labeling.py → upload_to_db.py)
    ├─ run_labeling.py (로컬 병렬)
    │   ├─ stance_labeler   → label_results/{qid}_{aid}_labeled.json
    │   ├─ body_depth       ↑
    │   ├─ loaded_words     ↑
    │   └─ [병렬 스레드]
    │       ├─ Groq: frame_labeler + bias_labeler  ↑
    │       └─ Gemini: omission_labeler             ↑
    │
    ├─ 댓글 처리 (병렬 workers=4)
    │   └─ emotion_labeler + loaded_words_labeler
    │       → label_results/{qid}_{aid}_comments.json
    │
    └─ upload_to_db.py
        ├─ *_labeled.json → article_features upsert
        └─ *_comments.json → comments upsert (cmt_emotion, cmt_words)

[4] step_timeline  (날짜별 중요도 순위 → DB 저장)
    ├─ timepoint_rank.rank_timepoints(): TF-IDF 기반 중요 날짜 선별
    ├─ 날짜별 대표 기사 (body 가장 긴 기사)
    ├─ save_timeline.clear_timeline_for_query()  ← 기존 삭제
    └─ save_timeline.save_timeline_entry()       ← 재삽입
```

---

### 라벨링만 단독 실행

```bash
# Supabase2 기준 (신규)
python AI/labeling/run_labeling.py --source supabase2 --query_id 15 --features all

# ai_test 기준 (구버전)
python AI/labeling/run_labeling.py --source ai_test --keyword 종소세 --features frame logic bias
```

```
[1] 데이터 로드
    --source supabase2 → load_by_query_id(query_id)
      ├─ Supabase2.articles (query_id 필터)
      └─ Supabase2.comments (article_id 배치 로드, 100개씩)

    --source ai_test → load_by_keyword(keyword)
      └─ Supabase.ai_test (keyword 필터)

[2] 피처별 처리 (label_map: {article_id → {피처결과}} 누적)

    ┌──── 로컬 (순차) ──────────────────────────────────────────┐
    │  stance  → KR-FinBert-SC                                   │
    │  body_depth → 수식 (즉시)                                  │
    │  loaded_words → BIAS_WORDS/OPINION_WORDS 사전              │
    └───────────────────────────────────────────────────────────┘
         ↓ [structs 사전계산: build_article_struct()]
    ┌──── API 병렬 스레드 ──────────────────────────────────────┐
    │  Thread-1 (Groq):                                          │
    │    frame_labeler → label_frame_logic_batch(structs)        │
    │    bias_labeler  → label_bias_batch(structs)               │
    │                                                            │
    │  Thread-2 (Gemini):                                        │
    │    [NER 전처리] extract_features() 기사별 1회              │
    │    cluster_entities = 전체에서 ≥30% 출현 엔티티             │
    │    omission_labeler → label_omission_batch()               │
    └───────────────────────────────────────────────────────────┘

[3] 배치 완료마다 즉시 저장 (_save_partial)
    label_results/{keyword}_{article_id}_labeled.json

[4] 댓글 라벨링 (ThreadPoolExecutor, workers=4)
    emotion_labeler + loaded_words_labeler
    → label_results/{keyword}_{article_id}_comments.json

[5] 리포트 저장 (label_results/reports/)
    run_report_{ts}.json
    label_dist_{ts}.json
    feature_timing_{ts}.json
    sample_labels_{ts}.json
    comments_stats_{ts}.json
```

---

## 4. 모듈별 상세

### 4.1 pipeline — 통합 오케스트레이터

#### `pipeline/run_pipeline.py`

진입점. 세 단계를 순서대로 실행하고 결과를 dict로 반환.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `query_id` | 필수 | queries 테이블 PK |
| `steps` | `["clustering","feature_map","timeline"]` | 실행할 단계 |
| `features` | `None` (=all) | labeling 피처 subset |
| `top_timepoints` | `8` | 타임라인 저장 날짜 수 |
| `limit` | `0` (=전체) | 최대 기사 수 |
| `resume` | `True` | 중단된 라벨링 재개 |
| `upload` | `True` | feature_map 후 자동 DB 업로드 |

#### `pipeline/steps/step_feature_map.py`

`run_labeling.py` → `upload_to_db.py` 를 **subprocess**로 호출.
라벨링 실패 시에도 완료분 업로드를 시도하는 내결함성 설계.

```python
result = {
    "exit_code":       int,
    "success":         bool,
    "upload_articles": bool,
    "upload_comments": bool,
}
```

---

### 4.2 labeling — 피처 라벨링

#### `labeling/run_labeling.py`

| 피처 | 담당 | 모델/방식 |
|---|---|---|
| `stance` | 로컬 | snunlp/KR-FinBert-SC |
| `body_depth` | 로컬 | 수식 (5가지 지표 가중합) |
| `loaded_words` | 로컬 | BIAS_WORDS/OPINION_WORDS 사전 |
| `frame` | Groq | Llama 3.3 70B |
| `logic` | Groq | Llama 3.3 70B (frame과 동일 호출) |
| `bias` | Groq | Llama 3.3 70B |
| `omission` | Gemini Pro | gemini-2.5-pro |
| `emotion` (댓글) | 로컬 | hun3359/klue-bert-base-sentiment |

**배치 크기:** 기본 100기사/배치 (API), 32댓글/배치 (로컬)  
**Resume 지원:** `--resume` 플래그로 기존 JSON 유지 + 누락 피처만 추가  
**Retry:** `--retry_errors` 플래그로 `api_error` / `null` 항목만 재처리

#### `labeling/features/preprocessor.py`

```python
build_article_struct(text: str) -> dict
# {
#   "headline": str,         # 첫 번째 줄
#   "lead": str,             # 첫 번째 단락
#   "quotes": list[str],     # 인용문 목록
#   "numbers": list[str],    # 수치/통계
#   "judgment_words": list[str],  # 판단·촉구 표현
#   "sources": list[str],    # 출처 표현
#   "sampled_sentences": list[str],
#   "total_sentences": int,
# }
```

#### `labeling/features/keyword_extractor.py`

KPF/KPF-bert-ner (한국어 뉴스 특화 NER). 싱글턴 로드.

```python
extract_features(text: str) -> dict
# {
#   "entities": [{"word": str, "entity": str}, ...],
#   "entity_count": int,
#   "entity_density": float,
#   "num_density": float,
#   "quote_count": int,
#   "source_count": int,
# }
```

#### `labeling/local/body_depth.py`

| 지표 | 가중치 | 산출 방식 |
|---|---|---|
| length | 0.20 | log 정규화 (soft_max=2000자) |
| diversity | 0.25 | 타입-토큰 비율 (TTR, ref=0.55) |
| quotes | 0.20 | 인용문 문자 비율 (ref=0.12) |
| numerics | 0.20 | 수치 패턴 밀도 (ref=0.06) |
| structure | 0.15 | 단락 수 정규화 (max=8단락) |

출력: `score ∈ [0.0, 1.0]` / 레이블: `"high" ≥0.65`, `"medium" ≥0.35`, `"low"`

#### `labeling/local/loaded_words_labeler.py`

3-tier 폴백 시스템으로 최소 2단어 보장:

```
Tier 1: BIAS_WORDS  (정치·이념 편향어 50+종)
    → 있으면 반환, density 계산
Tier 2: OPINION_WORDS + 자음패턴 (ㅋㅋ, ㄷㄷ 등)
    → Tier 1 결과 없을 때
Tier 3: 빈도×길이 순위 직접 추출
    → Tier 2도 없을 때 (is_biased=False)
```

#### `labeling/groq/frame_labeler.py`

**Frame 레이블 (7종)**

| ID | 레이블 |
|---|---|
| 1 | 사건 원인 집중 |
| 2 | 갈등/대립 강조 |
| 3 | 개인 사례 중심 |
| 4 | 경제적 영향 강조 |
| 5 | 윤리/도덕 판단 |
| 6 | 안전/안보 위협 |
| 7 | 권리/인권 강조 |

**Logic 레이블 (6종)**

| ID | 레이블 |
|---|---|
| 1 | 정책적 비난 |
| 2 | 전문가 견해 |
| 3 | 피해자 서사 |
| 4 | 파급효과 |
| 5 | 해결책 제시 |
| 6 | 사실/정보 전달 |

배치 n에 따라 프롬프트 내 lead/judgment_words/quotes 길이 동적 압축.  
레이블 매칭: 정확일치 → 숫자 인덱스 → 부분 포함 → Fuzzy (임계값 0.40) 순서로 5단계 폴백.

#### `labeling/groq/bias_labeler.py`

| 축 | 범위 | 의미 |
|---|---|---|
| `bias_x` | -1.0 ~ +1.0 | -1=강한 진보, +1=강한 보수 |
| `bias_y` | -1.0 ~ +1.0 | -1=강한 감성/선동, +1=순수 사실 보도 |

LLM 제약: 0.0 금지, 최소 ±0.1 이상. 후처리: 0.0 → `None` (DB NULL).

#### `labeling/gemini/omission_labeler.py`

| 레벨 | DB 값 | 의미 |
|---|---|---|
| `low` | -1 | 핵심 사실·행위자 대부분 포함 |
| `mid` | 0 | 일부 중요 관점 누락 |
| `high` | 1 | 핵심 사실 또는 주요 이해관계자 관점 대부분 누락 |

클러스터 내 전체 기사의 NER에서 ≥30% 출현 엔티티를 클러스터 기준으로 사용.

---

### 4.3 clustering — 기사 군집화

#### 파이프라인

```
기사 텍스트 (title + body)
    ↓
embedding.py: sentence-transformers (SBERT)
    ↓
reducer.py: UMAP (n_components=2, metric="cosine")
    ↓
clustering.py: HDBSCAN
    min_cluster_size = max(3, min(10, n // 5))
    min_samples = min_cluster_size - 1
    ↓
llm_client.call_llm_json: 클러스터별 제목(15자 이내) + 요약(2-3문장)
    ↓
article_clusters 테이블 삽입 + articles.cluster_label 업데이트
```

HDBSCAN label=-1 = 노이즈 (어느 클러스터에도 속하지 않음).

---

### 4.4 timeline — 타임라인 분석

#### 파이프라인

```
기사 목록 (published_at 필수)
    ↓
날짜(YYYY-MM-DD)별 그룹화
    ↓
timepoint_rank.rank_timepoints(): TF-IDF 기반 중요도 점수
    ↓
상위 top_timepoints개 날짜 선정
    ↓
날짜별 대표 기사 = body 가장 긴 기사
    ↓
clear_timeline_for_query() → 기존 항목 삭제
save_timeline_entry() × N → queries_timeline 삽입
```

---

### 4.5 source — 데이터 로더

| 파일 | Supabase 인스턴스 | 대상 테이블 |
|---|---|---|
| `config/supabase_client.py` | #1 (SUPABASE_*) | ai_test |
| `config/supabase2_client.py` | #2 (SUPABASE2_*) | articles, comments, queries, article_features, article_clusters, queries_timeline |

`run_labeling.py`에서 `--source ai_test` 시 supabase_client, `--source supabase2` 시 supabase2_client 사용.

---

### 4.6 llm — LLM 유틸리티 (레거시)

`auto_labeler.py` (레거시 CSV 파이프라인) 전용.  
Gemini Flash-lite 기반. `labeling/` 파이프라인과 무관.  
JSON 파싱은 `labeling/api_client_base.robust_json_parse` 공유.

---

## 5. DB 스키마 & 업로드 플로우

### 테이블 구조 (Supabase2)

```
articles
  id          PK
  query_id    FK → queries.id
  title       text
  body_text   text
  published_at timestamp
  url         text
  source_id   FK → sources.id
  cluster_label FK → article_clusters.label  ← clustering 단계에서 채움

comments
  id          PK
  article_id  FK → articles.id
  cmt_content text
  cmt_emotion text        ← labeling 단계에서 채움
  cmt_words   text[]      ← labeling 단계에서 채움

article_features
  article_id  PK (= articles.id)
  frame_id    int (1-7)
  logic_id    int (1-6)
  stance_score float
  bias_x      float
  bias_y      float
  art_words   text[]
  body_depth  float
  omission_risk int (-1/0/1)
  model_version text
  created_at  timestamp

article_clusters
  label       PK (자동증가)
  query_id    FK → queries.id
  cluster_title text
  cluster_summary text
  rep_article_id FK → articles.id
  frame_id    FK (nullable)
  created_at  timestamp

queries_timeline
  id          PK (자동증가)
  query_id    FK → queries.id
  timeline_date timestamp
  top_art     FK → articles.id
  created_at  timestamp

queries
  id          PK
  query_text  text
```

### 업로드 플로우 (`upload_to_db.py`)

```
label_results/{keyword}_{id}_labeled.json
    ↓
labeled_to_feature_row(d: dict)
    frame (str) → FRAME_MAP → frame_id (1-7)
    logic (str) → LOGIC_MAP → logic_id (1-6)
    omission_risk (str) → OMISSION_MAP → int (-1/0/1)
    stance_score, bias_x, bias_y, art_words, body_depth → 그대로
    model_version = "v1.0+{frame_model}+{bias_model}+{omission_model}"
    ↓
_upsert_batch(rows, dry_run=False)
    → 기존 article_id 조회 → UPDATE / INSERT 분기
    → Supabase2.article_features.upsert()

label_results/{keyword}_{id}_comments.json
    ↓
    cmt_emotion.label → comments.cmt_emotion
    cmt_words → comments.cmt_words
    → Supabase2.comments.upsert(on_conflict="id")
```

---

## 6. API 클라이언트 구조

### 공통 베이스 (`labeling/api_client_base.py`)

```
load_api_keys(env_multi, env_single)  → list[str]
load_models(env_multi, env_single, default)  → list[str]

class RpdCounter:
    remaining(key_idx)    → int      # 1회 파일 읽기
    increment(key_idx)               # 읽기 + 쓰기
    max_remaining(n_keys) → int      # 1회 파일 읽기 (N키 통합)
    status(n_keys)        → dict     # 1회 파일 읽기 (N키 통합)

fix_json_string(text)   → str   # <think>, markdown, 전각따옴표, trailing comma
robust_json_parse(raw)  → any   # 4전략: 직접 / 배열추출 / 객체추출 / NDJSON

retry_json_call(call_fn, prompt, *, _retry_json=2, label="LLM")
    # call_fn(prompt) 호출 → JSON 파싱 → 실패 시 suffix 붙여 재시도
```

### 로테이션 상태 머신

두 클라이언트(Groq, Gemini) 동일한 구조:

```
모듈 로드 시:
  _API_KEYS = load_api_keys(...)   # 1회만 파싱
  _MODELS   = load_models(...)     # 1회만 파싱
  _counter  = RpdCounter(...)      # 파일 기반 일일 카운터

세션 상태 (절대 후퇴 없음):
  _cur_model: int = 0
  _cur_key:   int = 0

로테이션 우선순위:
  RPD 소진    → _cur_key += 1  (같은 모델)
  RPM 재시도 MAX_RETRIES 소진 → _cur_key += 1
  기타 에러   → _cur_key += 1
  모든 키 소진 → _cur_model += 1, _cur_key = 0
  모델 에러   → _cur_model += 1, _cur_key = 0  (즉시)
  모든 조합 소진 → RuntimeError

로테이션 이력:
  rotation_log: deque(maxlen=500)
```

### Groq vs Gemini 비교

| 항목 | Groq | Gemini Pro |
|---|---|---|
| 모델 | Llama 3.3 70B | gemini-2.5-pro |
| 용도 | frame / logic / bias | omission_risk |
| RPM 간격 | 2초 (30 RPM) | 13초 (5 RPM) |
| RPD 한도 | 14,400/키 | 100/키 |
| Thread-safe | `call_groq_serial()` | `call_gemini_pro_serial()` |
| `<think>` 처리 | ✅ (공통 base) | ✅ (공통 base) |

---

## 7. 피처별 모델 정리

| 피처 | 출력 형식 | 모델/방식 | 저장 위치 |
|---|---|---|---|
| `stance_score` | float [-1, +1] | KR-FinBert-SC | article_features |
| `stance_label` | str | 임계값 분류 | (JSON만, DB 미저장) |
| `body_depth` | float [0, 1] | 수식 (5지표) | article_features |
| `body_depth_label` | str | 임계값 분류 | (JSON만) |
| `art_words` | str[] | BIAS_WORDS 사전 | article_features |
| `loaded_word_density` | float | tier-1 비율 | (JSON만) |
| `frame` | str → int (1-7) | Groq Llama 70B | article_features.frame_id |
| `logic` | str → int (1-6) | Groq Llama 70B | article_features.logic_id |
| `bias_x` | float [-1, +1] | Groq Llama 70B | article_features |
| `bias_y` | float [-1, +1] | Groq Llama 70B | article_features |
| `omission_risk` | str → int (-1/0/1) | Gemini Pro | article_features |
| `cmt_emotion` | str | klue-bert-sentiment | comments |
| `cmt_words` | str[] | BIAS_WORDS 사전 | comments |

---

## 8. 환경변수 레퍼런스

```ini
# Supabase
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE2_URL=
SUPABASE2_ANON_KEY=

# Groq (frame / logic / bias)
GROQ_API_KEYS=key1,key2,key3          # 콤마 구분, 따옴표 없이
GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant
GROQ_CALL_INTERVAL_SEC=2              # 30 RPM → 2초
GROQ_MAX_RETRIES=4
GROQ_RPD_LIMIT=14400
GROQ_COUNTER_FILE=.groq_usage.json

# Gemini Pro (omission)
GEMINI_API_KEYS=key1,key2
GEMINI_PRO_MODELS=gemini-2.5-pro,gemini-2.0-flash
GEMINI_PRO_RPD_LIMIT=100
GEMINI_PRO_CALL_INTERVAL_SEC=13       # 5 RPM → 13초
GEMINI_PRO_COUNTER_FILE=.gemini_pro_usage.json

# Gemini Flash (llm/ 레거시, auto_labeler.py 전용)
GEMINI_API_KEY=
LLM_MODEL_1=gemini-2.5-flash-lite
LLM_MODEL_2=gemini-2.0-flash-lite
LLM_MODEL_3=gemini-1.5-flash
LLM_CALL_INTERVAL_SEC=7
LLM_MAX_RETRIES=4

# 로컬 모델 (선택)
EMOTION_MODEL=hun3359/klue-bert-base-sentiment
```

---

## 9. CLI 진입점 정리

### 전체 파이프라인

```bash
python -m AI.pipeline.run_pipeline \
  --query_id 15 \
  --steps clustering feature_map timeline \
  --features all \
  --top_timepoints 8
```

### 라벨링만 (supabase2)

```bash
python AI/labeling/run_labeling.py \
  --source supabase2 --query_id 15 \
  --features frame logic bias omission \
  --batch_size 100 --resume
```

### 라벨링만 (ai_test 레거시)

```bash
python AI/labeling/run_labeling.py \
  --source ai_test --keyword 종소세 \
  --features all --limit 50
```

### 에러 재시도

```bash
python AI/labeling/run_labeling.py \
  --source supabase2 --query_id 15 \
  --retry_errors --features frame bias
```

### DB 업로드 (수동)

```bash
# 기사 피처만
python AI/labeling/upload_to_db.py --mode articles --keyword 15 --label_dir ./label_results

# 기사 + 댓글
python AI/labeling/upload_to_db.py --mode all --keyword 15

# 미리보기 (실제 업로드 없음)
python AI/labeling/upload_to_db.py --mode all --keyword 15 --dry_run
```

### 통계 확인

```bash
python AI/labeling/label_stats.py
```
