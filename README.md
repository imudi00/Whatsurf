# WhatSurf — 뉴스 분석 AI 파이프라인

## 디렉토리 구조
```
AI/
├── labeling/           # 자동 라벨링 (메인 파이프라인)
│   ├── run_labeling.py     ← 통합 실행 진입점
│   ├── upload_to_db.py     ← Supabase2 article_features 업로드
│   ├── groq/               ← frame / logic / bias (Groq Llama)
│   ├── gemini/             ← omission_risk (Gemini Pro)
│   ├── local/              ← stance / emotion / loaded_words / body_depth (로컬)
│   ├── docs/               ← 아키텍처 · 이슈 문서
│   └── research_report.py  ← 실행 리포트
├── feature_map/        # 피처 추출 모듈
│   ├── pipeline.py         ← 전체 파이프라인 (LLM 포함)
│   ├── context/            ← body_depth, omission_risk, 키워드 추출
│   ├── emotion/            ← 편향 벡터, 감정 모델, 편향 단어
│   └── stance/             ← 스탠스 분석, LLM 클라이언트
├── llm/                # LLM 유틸
│   ├── llm_client.py       ← Gemini API 래퍼
│   └── llm_batch.py        ← 배치 처리
└── source/             # 데이터 소스
    ├── data_loader.py      ← Supabase / Supabase2 로더
    └── config/
        ├── supabase_client.py   ← 기존 DB (ai_test)
        └── supabase2_client.py  ← 신규 DB (articles, article_features)
```

---

## .env 필수 키
```env
# 기존 Supabase (ai_test 소스)
SUPABASE_URL=...
SUPABASE_ANON_KEY=...

# 신규 Supabase2 (articles / article_features)
SUPABASE2_URL=...
SUPABASE2_ANON_KEY=...

# Groq (frame / logic / bias)
GROQ_API_KEYS=key1,key2
GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant
GROQ_RPD_LIMIT=14400
GROQ_CALL_INTERVAL_SEC=2
GROQ_MAX_RETRIES=4

# Gemini Pro (omission_risk)
GEMINI_API_KEYS=key1,key2
GEMINI_PRO_MODELS=gemini-2.5-pro,gemini-2.5-flash
GEMINI_PRO_RPD_LIMIT=100
GEMINI_PRO_CALL_INTERVAL_SEC=13
```

---

## 실행 방법

### 1. 라벨링 — 기존 DB (ai_test)

```bash
# 전체 피처 (stance / loaded_words / body_depth / frame / logic / bias / omission)
python AI/labeling/run_labeling.py --keyword 탄핵

# 특정 피처만
python AI/labeling/run_labeling.py --keyword 탄핵 --features frame logic bias
python AI/labeling/run_labeling.py --keyword 탄핵 --features omission   # Gemini 100건 차감

# 수식 기반 피처만 (API 미사용, 빠름)
python AI/labeling/run_labeling.py --keyword 탄핵 --features stance loaded_words body_depth --skip_comments

# 중단 후 재개 (기존 json 유지 + omission만 추가)
python AI/labeling/run_labeling.py --keyword 탄핵 --features omission --resume

# 건수 제한 / 오프셋
python AI/labeling/run_labeling.py --keyword 탄핵 --limit 50 --offset 100

# 병렬 비활성화 (Groq · Gemini 순차 실행)
python AI/labeling/run_labeling.py --keyword 탄핵 --no_parallel

# 댓글 제외
python AI/labeling/run_labeling.py --keyword 탄핵 --skip_comments
```

### 2. 라벨링 — 신규 DB (Supabase2 articles)

```bash
# query_id 기준으로 Supabase2 articles 테이블에서 데이터 로드 후 라벨링
python AI/labeling/run_labeling.py --source supabase2 --query_id 42

# 특정 피처만
python AI/labeling/run_labeling.py --source supabase2 --query_id 42 --features frame logic bias body_depth

# 재개
python AI/labeling/run_labeling.py --source supabase2 --query_id 42 --features omission --resume
```

### 3. Supabase2 업로드

```bash
# 기사 피처 + 댓글 피처 한번에 (기본)
python AI/labeling/upload_to_db.py --mode all --keyword 탄핵

# 기사 피처만 (article_features 테이블)
python AI/labeling/upload_to_db.py --mode articles --keyword 탄핵

# 댓글 피처만 (comments 테이블 — cmt_emotion, cmt_words 업데이트)
python AI/labeling/upload_to_db.py --mode comments --keyword 탄핵

# label_results 전체 업로드
python AI/labeling/upload_to_db.py --mode all --all

# 미리보기 (실제 업로드 없음)
python AI/labeling/upload_to_db.py --mode all --keyword 탄핵 --dry_run
```

> **주의:** comments 업로드는 `--source supabase2` 로 라벨링한 경우에만 가능.
> ai_test 소스는 댓글에 DB의 `id` 가 없어 건너뜀.

### 4. feature_map 파이프라인 (LLM 포함)

```bash
python AI/feature_map/pipeline.py --keyword 탄핵 --limit 10 --batch_size 3
```

---

## 피처 설명

| 피처 | 담당 | 설명 |
|------|------|------|
| `stance_score` | 로컬 HuggingFace | 기사 입장 (-1 반대 ~ +1 지지) |
| `art_words` | 로컬 규칙 | 편향 표현 단어 목록 |
| `body_depth` | 로컬 수식 | 본문 정보 깊이 (0~1, 길이·어휘·인용·수치·구조) |
| `frame_id` | Groq Llama | 기사 프레임 (1~7, DB 테이블 참조) |
| `logic_id` | Groq Llama | 논리 유형 (1~6, DB 테이블 참조) |
| `bias_x / bias_y` | Groq Llama | 편향 벡터 (-1~1) |
| `omission_risk` | Gemini Pro | 누락 위험도 (1=high / 0=mid / -1=low) |

### body_depth 튜닝
`AI/labeling/local/body_depth.py` 상단 상수 수정:
```python
WEIGHTS = {
    "length": 0.20, "diversity": 0.25,
    "quotes": 0.20, "numerics":  0.20, "structure": 0.15,
}
LENGTH_SOFT_MAX     = 2000   # 이 자수에서 length_score ≈ 1.0
DIVERSITY_TTR_REF   = 0.55   # 어절 TTR 기준값
QUOTE_RATIO_REF     = 0.12   # 인용 비율 기준
NUMERIC_DENSITY_REF = 0.06   # 수치 밀도 기준
STRUCTURE_PARA_MAX  = 8      # 기준 문단 수
```

---

## 출력 파일
```
label_results/
├── {keyword}_{id}_labeled.json    ← 기사 라벨 (피처 전체)
├── {keyword}_{id}_comments.json  ← 댓글 감정 + 편향 단어
└── reports/
    ├── run_report_{ts}.json       ← 실행 전체 리포트
    ├── label_dist_{ts}.json       ← 라벨 분포 요약
    ├── feature_timing_{ts}.json   ← 피처별 처리 속도
    ├── sample_labels_{ts}.json    ← 피처별 샘플 10건
    └── comments_stats_{ts}.json   ← 댓글 통계
```
