# WhatSurf AI 라벨링 파이프라인 — 전체 개요

## 프로젝트 목적

Supabase에 저장된 한국어 뉴스 기사와 댓글을 자동 라벨링하여 연구에 활용하기 위한 파이프라인.

## 처리 대상 데이터

| 테이블 | 컬럼 | 설명 |
|---|---|---|
| `ai_test` | `id, title, body, comments, keyword` | 뉴스 기사 + 댓글(JSONB) |
| `article_features` | `art_words` | 기사 편향 단어 (JSONB) |
| `comments` | `cmt_emotion`, `cmt_words` | 댓글별 감정/편향 단어 (JSONB) |

## 피처 목록

| 피처 | 처리 방식 | 모델/도구 |
|---|---|---|
| `stance_score` | 로컬 HuggingFace | klue-bert |
| `art_words` | 로컬 규칙 기반 | BIAS_WORDS + OPINION_WORDS 사전 |
| `frame`, `logic` | Groq API | Llama 3.3 70B |
| `bias_x`, `bias_y` | Groq API | Llama 3.3 70B |
| `omission_risk` | Gemini API | Gemini 2.5 Pro |
| `cmt_emotion` | 로컬 HuggingFace | klue-bert-sentiment |
| `cmt_words` | 로컬 규칙 기반 | BIAS_WORDS + OPINION_WORDS 사전 |

## 디렉토리 구조

```
AI/
├── labeling/
│   ├── run_labeling.py          # 진입점
│   ├── research_report.py       # 연구 리포트 생성
│   ├── groq/
│   │   ├── groq_client.py       # Groq API (키·모델 로테이션)
│   │   ├── frame_labeler.py     # frame/logic 라벨링
│   │   └── bias_labeler.py      # bias_x/bias_y 라벨링
│   ├── gemini/
│   │   ├── gemini_client.py     # Gemini API (키·모델 로테이션)
│   │   └── omission_labeler.py  # omission_risk 라벨링
│   └── local/
│       ├── stance_labeler.py
│       ├── emotion_labeler.py
│       └── loaded_words_labeler.py
├── feature_map/
│   └── emotion/src/feature_map/
│       └── loaded_words.py      # BIAS_WORDS / OPINION_WORDS 사전
└── source/
    └── config/
        └── supabase_client.py
```

## 실행 흐름

```
[1] Supabase 로드 (페이지네이션, 전체 데이터)
[2] 기사 라벨링
    ├─ 로컬: stance → loaded_words (순차)
    └─ API: [Groq(frame+bias) ‖ Gemini(omission)] (병렬)
[3] 댓글 라벨링 (로컬, 기사 단위 병렬)
[4] 연구 리포트 저장 (5개 파일)
```

## 실행 예시

```bash
# 전체 데이터, 기본 설정
python AI/labeling/run_labeling.py --keyword 의대정원

# 대용량 재개 (앞의 5000개 건너뜀)
python AI/labeling/run_labeling.py --keyword 의대정원 --offset 5000

# Gemini 하루 한도 제어 (기본 200개)
python AI/labeling/run_labeling.py --keyword 의대정원 --max_gemini 100

# 디버깅 (병렬 끄기)
python AI/labeling/run_labeling.py --keyword 의대정원 --no_parallel --limit 10
```

## .env 설정 예시

```env
# Groq — 따옴표 없이, 콤마 구분
GROQ_API_KEYS=gsk_key1,gsk_key2
GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant,mixtral-8x7b-32768

# Gemini — 따옴표 없이, 콤마 구분
GEMINI_API_KEYS=AIza_key1,AIza_key2
GEMINI_PRO_MODELS=gemini-2.5-pro,gemini-2.0-flash

# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
```

> **중요**: `.env` 값에 개별 따옴표 금지. `KEY='a','b'` → ❌ / `KEY=a,b` → ✅
