# WhatSurf AI 라벨링 파이프라인 — 전체 개요

## 프로젝트 목적

Supabase에 저장된 한국어 뉴스 기사와 댓글을 자동 라벨링하여 연구에 활용하기 위한 파이프라인.

## 처리 대상 데이터

| 소스 | 테이블 | 컬럼 | 설명 |
|---|---|---|---|
| ai_test | `ai_test` | `id, title, body, comments, keyword` | 뉴스 기사 + 댓글(JSONB) |
| supabase2 | `articles` | `id, query_id, title, body_text` | 뉴스 기사 |
| supabase2 | `comments` | `id, article_id, cmt_content` | 댓글 |
| supabase2 | `article_features` | `article_id, frame_id, logic_id, ...` | 라벨링 결과 저장 |

## 피처 목록

| 피처 | 처리 방식 | 모델/도구 |
|---|---|---|
| `stance_score` | 로컬 HuggingFace | KR-FinBert-SC |
| `body_depth` | 수식 기반 (LLM 미사용) | 없음 |
| `art_words` | 로컬 규칙 기반 | BIAS_WORDS + OPINION_WORDS 사전 |
| `frame`, `logic` | Groq API | Llama 3.3 70B |
| `bias_x`, `bias_y` | Groq API | Llama 3.3 70B |
| `omission_risk` | Gemini API | Gemini 2.5 Pro |
| `cmt_emotion` | 로컬 HuggingFace | klue-bert-sentiment |
| `cmt_words` | 로컬 규칙 기반 | BIAS_WORDS + OPINION_WORDS 사전 |

## 디렉토리 구조

```
AI/
├── labeling/                          # ★ 메인 패키지
│   ├── run_labeling.py                # 진입점 — 라벨링 실행
│   ├── upload_to_db.py                # 진입점 — Supabase2 업로드
│   ├── research_report.py             # 실행 리포트 생성
│   │
│   ├── features/                      # 로컬 텍스트 피처 (feature_map에서 통합)
│   │   ├── preprocessor.py            # build_article_struct
│   │   ├── text_utils.py              # 텍스트 전처리 유틸
│   │   ├── loaded_words.py            # BIAS_WORDS / OPINION_WORDS 사전
│   │   ├── keyword_extractor.py       # KPF-BERT NER → extract_features
│   │   ├── bias_vector.py             # 규칙 기반 편향 벡터
│   │   └── omission_risk.py           # 규칙 기반 누락 위험도
│   │
│   ├── local/                         # 로컬 HuggingFace 모델 래퍼
│   │   ├── body_depth.py              # 본문 깊이 (수식 기반)
│   │   ├── emotion_labeler.py         # 댓글 감정 분류
│   │   ├── loaded_words_labeler.py    # 편향 단어 추출
│   │   └── stance_labeler.py          # 기사 논조 점수
│   │
│   ├── groq/                          # Groq LLM (frame/logic/bias)
│   │   ├── groq_client.py
│   │   ├── frame_labeler.py
│   │   └── bias_labeler.py
│   │
│   ├── gemini/                        # Gemini LLM (omission)
│   │   ├── gemini_client.py
│   │   └── omission_labeler.py
│   │
│   └── docs/                          # 이슈 로그·아키텍처 문서
│
├── clustering/                        # 기사 클러스터링 (독립 모듈)
├── timeline/                          # 시간 버스트 분석 (독립 모듈)
├── source/config/                     # Supabase 클라이언트
│   ├── supabase_client.py             # ai_test 소스
│   └── supabase2_client.py            # articles/comments 소스
└── feature_map/                       # ⚠️ DEPRECATED → labeling/features/ 로 통합
```

## 실행 흐름

```
[1] Supabase 로드 (ai_test 또는 supabase2, 페이지네이션)
[2] 기사 라벨링
    ├─ 로컬 순차: body_depth → stance → loaded_words
    └─ API 병렬: Groq (frame+logic+bias) ‖ Gemini (omission)
[3] 댓글 라벨링 (로컬, 기사 단위 병렬)
[4] labeled.json 저장 (배치 완료 즉시 저장)
[5] 연구 리포트 저장
[6] upload_to_db.py → Supabase2 업로드
```

## 실행 예시

```bash
# ai_test 소스 (keyword 기반)
python AI/labeling/run_labeling.py --keyword 의대정원

# supabase2 소스 (query_id 기반)
python AI/labeling/run_labeling.py --source supabase2 --query_id Q_001

# 중단 재개
python AI/labeling/run_labeling.py --keyword 의대정원 --resume

# DB 업로드
python AI/labeling/upload_to_db.py --labeled_dir AI/labeling/output --mode all
```

자세한 옵션은 **AI/README.md** 참고.

## .env 필수 키

```env
GROQ_API_KEYS=gsk_...
GROQ_MODELS=meta-llama/llama-3.3-70b-versatile,...
GEMINI_API_KEYS=AIza...
GEMINI_PRO_MODELS=gemini-2.5-pro,...
SUPABASE_URL=https://...
SUPABASE_KEY=eyJ...
SUPABASE2_URL=https://...
SUPABASE2_ANON_KEY=eyJ...
```

> **중요**: `.env` 값에 개별 따옴표 금지. `KEY='a','b'` → ❌ / `KEY=a,b` → ✅
