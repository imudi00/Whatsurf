# Clustering & Timeline 모듈 가이드

> 최초 작성: 2026-03-23 | 테스트 환경: Python 3.13, Windows 11

---

## 목차

1. [디렉토리 구조](#1-디렉토리-구조)
2. [Clustering 모듈](#2-clustering-모듈)
   - [실행 방법](#21-실행-방법)
   - [모듈 구조](#22-모듈-구조)
   - [데이터 흐름](#23-데이터-흐름)
   - [실험 결과](#24-실험-결과)
3. [Timeline 모듈](#3-timeline-모듈)
   - [실행 방법](#31-실행-방법)
   - [모듈 구조](#32-모듈-구조)
   - [데이터 흐름](#33-데이터-흐름)
   - [입출력 예시](#34-입출력-예시)
4. [공통 설치](#4-공통-설치)
5. [알려진 이슈 및 해결](#5-알려진-이슈-및-해결)

---

## 1. 디렉토리 구조

```
AI/
├── clustering/
│   ├── src/
│   │   └── clustering/
│   │       ├── clustering.py        # KMeans / HDBSCAN
│   │       ├── data_loader.py       # Naver 뉴스 데이터셋 로드
│   │       ├── embedding.py         # SBERT 임베딩
│   │       ├── tfidf_embedding.py   # TF-IDF 임베딩
│   │       ├── reducer.py           # UMAP 차원 축소
│   │       ├── evaluation.py        # Silhouette Score
│   │       ├── incremental.py       # 증분 클러스터링 (개발 중)
│   │       └── service_pipeline.py  # SBERT→UMAP→HDBSCAN 통합
│   └── experiments/
│       ├── run_test.py              # SBERT vs TF-IDF 비교 실험
│       └── run_service_test.py      # 서비스 파이프라인 테스트
│
├── timeline/
│   ├── src/
│   │   └── timeline/
│   │       ├── io_utils.py              # CSV 로드 / JSON 저장
│   │       ├── text_utils.py            # 한글 문장 분할
│   │       ├── burst.py                 # Z-score 버스트 감지
│   │       ├── timepoint_rank.py        # TTP/ETP 중요도 계산
│   │       ├── summarize_extractive.py  # 추출형 요약
│   │       ├── export_timeline.py       # 최종 타임라인 구성
│   │       └── plot_counts.py           # 일일 기사수 시각화
│   ├── experiments/
│   │   └── run_timeline_test.py     # 통합 실행 진입점
│   ├── data/sample/
│   │   └── sample_news.csv          # 테스트용 샘플 데이터
│   └── artifacts/                   # 실행 결과 저장 위치
│
└── docs/
    └── clustering_timeline.md       # 이 문서
```

---

## 2. Clustering 모듈

### 2.1 실행 방법

**프로젝트 루트(`AI/`)에서 실행.**

```bash
# ① SBERT vs TF-IDF 비교 실험 (K=5,10,15,20 각각 Silhouette Score 측정)
cd AI
python -m clustering.experiments.run_test

# ② 서비스 파이프라인 테스트 (SBERT→UMAP→HDBSCAN)
python -m clustering.experiments.run_service_test
```

> **주의**: 첫 실행 시 HuggingFace에서 SBERT 모델 (~400MB) 과 Naver 뉴스 데이터셋을 다운로드합니다.
> SBERT 임베딩은 CPU 기준 약 **90~100초** 소요됩니다.

### 2.2 모듈 구조

| 파일 | 함수 | IN | OUT |
|------|------|----|-----|
| `data_loader.py` | `load_naver_news(sample_size=None)` | HuggingFace 데이터셋 | `List[str]` (최대 1000개 뉴스 본문) |
| `embedding.py` | `sbert_embedding(texts)` | `List[str]` | `ndarray (N, 768)` |
| `tfidf_embedding.py` | `tfidf_embedding(texts)` | `List[str]` | `sparse_matrix (N, 10000)` |
| `reducer.py` | `reduce_dimension(embeddings)` | `ndarray (N, 768)` | `ndarray (N, 5)` |
| `clustering.py` | `kmeans_cluster(embeddings, n_clusters=10)` | `ndarray` | `ndarray` (클러스터 ID) |
| `clustering.py` | `density_cluster(reduced_embeddings)` | `ndarray (N, 5)` | `ndarray` (-1=노이즈) |
| `evaluation.py` | `evaluate_cluster(embeddings, labels)` | `ndarray`, `ndarray` | `float` (Silhouette Score) |
| `service_pipeline.py` | `run_service_clustering(texts)` | `List[str]` | `ndarray` (HDBSCAN 레이블) |
| `incremental.py` | `assign_new_article(new_emb, centroids)` | `ndarray`, `ndarray` | `str` (클러스터 ID 또는 `"NEW_CLUSTER"`) |

#### 모델 설정값

| 파라미터 | 값 | 위치 |
|----------|-----|------|
| SBERT 모델 | `snunlp/KR-SBERT-V40K-klueNLI-augSTS` | `embedding.py` |
| 임베딩 배치 크기 | 32 | `embedding.py` |
| TF-IDF 최대 피처 | 10,000 | `tfidf_embedding.py` |
| UMAP n_neighbors | 15 | `reducer.py` |
| UMAP n_components | 5 | `reducer.py` |
| UMAP metric | cosine | `reducer.py` |
| HDBSCAN min_cluster_size | 10 | `clustering.py` |
| 증분 유사도 임계값 | 0.6 | `incremental.py` |

### 2.3 데이터 흐름

```
HuggingFace: daekeun-ml/naver-news-summarization-ko
    │
    ▼
load_naver_news()
    │ List[str]  (1,000개 뉴스 본문)
    ▼
┌─────────────────┬──────────────────┐
│ sbert_embedding │ tfidf_embedding  │
│ (N, 768)        │ (N, 10000)       │
└────────┬────────┴──────────────────┘
         │ SBERT only
         ▼
reduce_dimension()   ← UMAP
    │ (N, 5)
    ▼
┌──────────────────┬─────────────────────┐
│ kmeans_cluster() │ density_cluster()   │
│ K=5,10,15,20     │ HDBSCAN (-1=노이즈)  │
└──────────────────┴─────────────────────┘
    │
    ▼
evaluate_cluster()   ← Silhouette Score
    │
    ▼
experiment_log.csv   ← 결과 누적 저장
```

### 2.4 실험 결과

실제 실행 결과 (`run_test.py`, 2026-03-23):

| 모델 | K | Silhouette Score | 임베딩 시간 | 클러스터링 시간 |
|------|---|-----------------|------------|--------------|
| SBERT | 5 | 0.0387 | ~94초 | 0.51초 |
| SBERT | 10 | 0.0311 | (재사용) | 0.55초 |
| SBERT | 15 | 0.0351 | (재사용) | 0.78초 |
| SBERT | 20 | 0.0360 | (재사용) | 0.87초 |
| TF-IDF | 5 | 0.0110 | 0.51초 | 0.25초 |
| TF-IDF | 10 | 0.0121 | (재사용) | 0.36초 |
| TF-IDF | 15 | 0.0148 | (재사용) | 0.62초 |
| TF-IDF | 20 | 0.0157 | (재사용) | 0.76초 |

> Silhouette Score가 전반적으로 낮음 (0.01~0.04). 한국어 뉴스 특성상 주제가 광범위하게 분포해 클러스터 경계가 명확하지 않기 때문으로 보임. SBERT가 TF-IDF 대비 약 3배 높은 점수.

---

## 3. Timeline 모듈

### 3.1 실행 방법

```bash
# 기본 실행 (샘플 데이터 사용)
cd AI
python -m timeline.experiments.run_timeline_test \
  --input timeline/data/sample/sample_news.csv \
  --query "탄핵" \
  --out_dir timeline/experiments/artifacts

# 전체 파라미터
python -m timeline.experiments.run_timeline_test \
  --input <CSV_파일_경로> \       # 필수: date, title, body 컬럼 포함 CSV
  --query <검색_키워드> \          # 필수: 요약 시 쿼리 보너스에 사용
  --out_dir <출력_디렉토리> \      # 기본: experiments/artifacts
  --top_timepoints 8 \            # 기본: 상위 8개 시점
  --max_eojel 17 \                # 기본: 요약 문장 최대 어절 수
  --alpha 0.5 \                   # 기본: 쿼리 보너스 가중치
  --top_sentences 3               # 기본: 시점당 요약 문장 수
```

**입력 CSV 형식:**

```csv
date,title,body,url
2024-01-15,기사 제목1,기사 본문 내용...,https://...
2024-01-16,기사 제목2,기사 본문 내용...,
```

> - `date`: 날짜 (YYYY-MM-DD 권장, 자동 파싱)
> - `title`: 기사 제목 (필수)
> - `body`: 기사 본문 (필수, 비어있으면 자동 제외)
> - `url`: 기사 URL (선택)
> - 인코딩: UTF-8 → CP949 → EUC-KR 순으로 자동 감지

### 3.2 모듈 구조

| 파일 | 함수 | IN | OUT |
|------|------|----|-----|
| `io_utils.py` | `load_news_csv(path)` | CSV 파일 경로 | `DataFrame(date, title, body, url)` |
| `io_utils.py` | `save_json(obj, path)` | dict/list, 경로 | JSON 파일 저장 |
| `text_utils.py` | `split_sentences_kor(text)` | 본문 str | `List[str]` (한글 문장 리스트) |
| `burst.py` | `compute_daily_counts(df)` | DataFrame | `Series` (날짜별 기사 수) |
| `burst.py` | `detect_burst_points(counts, z_threshold=2.0)` | Series | `List[str]` (버스트 날짜) |
| `timepoint_rank.py` | `rank_timepoints(df, top_keywords_per_day=20)` | DataFrame | `DataFrame(date, TTP, ETP, importance)` |
| `summarize_extractive.py` | `summarize_timepoint(day_articles, query, ...)` | DataFrame, str | `str` (요약 문장) |
| `export_timeline.py` | `build_timeline(df, query, ranked, ...)` | DataFrame, str, DataFrame | `List[dict]` |
| `plot_counts.py` | `plot_daily_counts(counts, burst_dates, out_path)` | Series, List, 경로 | PNG 파일 저장 |

#### 알고리즘 설명

**버스트 감지 (Z-score)**
```
Z = (일일 기사 수 - μ) / σ
버스트 = Z ≥ z_threshold(2.0) AND 기사 수 ≥ min_count(2)
```

**시점 중요도 (TTP/ETP)**
```
TTP (Timepoint Popularity) = 날짜별 기사 수
ETP (Event Term Popularity) = 날짜별 상위 20개 키워드 출현 빈도 합산
Importance = (ETP / max_ETP) × (TTP / max_TTP)
```

**추출형 요약 (TF-IDF + 쿼리 보너스)**
```
score = base_tfidf + α × query_bonus   (α = 0.5)
```
- 어절 1~17개 범위 문장만 사용
- 상위 3개 문장 선택 후 중복 제거

### 3.3 데이터 흐름

```
뉴스 CSV (date, title, body, url)
    │
    ▼
load_news_csv()
    │ DataFrame
    ▼
compute_daily_counts()
    │ Series (날짜별 기사 수)
    ├──────────────────────────────┐
    ▼                              ▼
detect_burst_points()       plot_daily_counts()
    │ List[str] (버스트 날짜)       │ daily_counts.png
    ▼
rank_timepoints()
    │ DataFrame (date, TTP, ETP, importance)
    │ ranked_timepoints.json
    ▼
build_timeline()
    ├─ summarize_timepoint()  ← TF-IDF 추출 요약
    └─ choose_representative_article()  ← 가장 긴 본문 기사
    │ List[dict]
    ▼
save_json()
    └─ timeline.json
```

### 3.4 입출력 예시

**`ranked_timepoints.json`**
```json
[
  {"date": "2024-01-15", "TTP": 12, "ETP": 847, "importance": 1.0},
  {"date": "2024-01-08", "TTP": 9,  "ETP": 623, "importance": 0.72},
  ...
]
```

**`timeline.json`**
```json
{
  "query": "탄핵",
  "burst_dates": ["2024-01-15", "2024-01-22"],
  "timeline": [
    {
      "date": "2024-01-08",
      "importance": 0.72,
      "summary": "헌법재판소가 탄핵 심판 일정을 발표했다. 여야는 각각 입장을 밝혔다. 시민단체들은 신속한 결론을 촉구했다.",
      "representative": {
        "title": "헌재, 탄핵 심판 일정 공개",
        "url": "https://..."
      },
      "articles": [
        {"title": "헌재, 탄핵 심판 일정 공개", "url": "https://..."},
        {"title": "여야 반응 엇갈려", "url": "https://..."}
      ],
      "count": 9
    },
    ...
  ]
}
```

---

## 4. 공통 설치

```bash
# 가상환경 활성화
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Mac/Linux

# 필수 패키지 설치
pip install -r requirements/ai.txt

# clustering 추가 의존성 (ai.txt에 없을 경우)
pip install sentence-transformers hdbscan umap-learn datasets
```

**패키지별 용도:**

| 패키지 | 용도 |
|--------|------|
| `sentence-transformers` | SBERT 한국어 임베딩 |
| `hdbscan` | 밀도 기반 클러스터링 |
| `umap-learn` | 차원 축소 (UMAP) |
| `datasets` | HuggingFace 데이터셋 로드 |
| `scikit-learn` | KMeans, TF-IDF, Silhouette Score |
| `pandas`, `numpy` | 공통 데이터 처리 |
| `matplotlib` | 타임라인 시각화 |

---

## 5. 알려진 이슈 및 해결

### ① Windows cp949 UnicodeEncodeError

**증상**: `clustering/experiments/run_test.py` 실행 시 클러스터 샘플 출력 단계에서 아래 오류 발생
```
UnicodeEncodeError: 'cp949' codec can't encode character '\xa9' in position 48
```

**원인**: Naver 뉴스 데이터셋에 `©` 등 cp949 미지원 특수문자 포함. Windows 기본 콘솔 인코딩(cp949)과 충돌.

**해결**: `run_test.py`, `run_service_test.py` 상단에 stdout 재설정 추가 (이미 적용됨)
```python
import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
```

---

### ② joblib WinError 2 (wmic 경고)

**증상**: 실행 중 아래 경고 출력 (실행에는 영향 없음)
```
UserWarning: Could not find the number of physical cores...
[WinError 2] 지정된 파일을 찾을 수 없습니다.
```

**원인**: joblib이 `wmic`으로 CPU 코어 수를 조회하는데, 일부 Windows 환경에서 wmic 명령 미지원.

**해결**: 환경변수로 코어 수를 직접 지정 (이미 적용됨)
```bash
set LOKY_MAX_CPU_COUNT=4
# 또는 코드에서
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
```

---

### ③ SBERT 임베딩 속도

**증상**: 1,000개 문서 임베딩에 CPU 기준 약 90~100초 소요.

**해결 방안**:
- GPU 환경이면 자동으로 CUDA 사용 (10배 이상 빠름)
- 소량 테스트 시 `load_naver_news(sample_size=100)` 으로 축소 가능

---

### ④ `incremental.py` 미완성

`compute_centroids()` 함수는 시그니처만 정의되어 있고 본문 구현이 없음. 증분 클러스터링 기능은 사용 불가.

```python
# incremental.py — 미구현 상태
def compute_centroids(embeddings, labels):
    pass  # TODO
```

---

### ⑤ HuggingFace 비인증 경고

**증상**: 실행 시 아래 경고 출력
```
Warning: You are sending unauthenticated requests to the HF Hub.
```

**해결**: `.env`에 HuggingFace 토큰 추가 (선택사항, 없어도 동작함)
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```
