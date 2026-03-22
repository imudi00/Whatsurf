# WhatSurf AI — 뉴스 라벨링 파이프라인

한국어 뉴스 기사와 댓글을 자동 라벨링하여 Supabase DB에 저장하는 파이프라인.

---

## 디렉토리 구조

```
AI/
├── labeling/                          # ★ 메인 패키지 (여기만 실행하면 됨)
│   ├── run_labeling.py                # 진입점 — 라벨링 실행
│   ├── upload_to_db.py                # 진입점 — labeled.json → Supabase2 업로드
│   ├── research_report.py             # 실행 리포트 생성
│   │
│   ├── features/                      # 로컬 텍스트 피처 모듈 (LLM 미사용)
│   │   ├── preprocessor.py            # build_article_struct (Groq 입력용)
│   │   ├── text_utils.py              # extract_headline / lead / quotes 등
│   │   ├── loaded_words.py            # BIAS_WORDS / OPINION_WORDS 사전 + 감지 함수
│   │   ├── keyword_extractor.py       # KPF-BERT NER → extract_features
│   │   ├── bias_vector.py             # 좌-우 편향 벡터 (규칙 기반)
│   │   └── omission_risk.py           # 클러스터 대비 누락도 (규칙 기반 경량 버전)
│   │
│   ├── local/                         # 로컬 HuggingFace 모델 래퍼
│   │   ├── body_depth.py              # 본문 깊이 점수 (수식 기반, 모델 미사용)
│   │   ├── emotion_labeler.py         # 댓글 감정 분류 (klue-bert-sentiment)
│   │   ├── loaded_words_labeler.py    # 편향 단어 추출 (features/loaded_words 사용)
│   │   └── stance_labeler.py          # 기사 논조 점수 (KR-FinBert-SC)
│   │
│   ├── groq/                          # Groq LLM API
│   │   ├── groq_client.py             # API 키·모델 로테이션, JSON 파싱
│   │   ├── frame_labeler.py           # frame / logic 라벨링
│   │   └── bias_labeler.py            # bias_x / bias_y 라벨링
│   │
│   ├── gemini/                        # Google Gemini API
│   │   ├── gemini_client.py           # API 키·모델 로테이션, RPD 추적
│   │   └── omission_labeler.py        # omission_risk 라벨링 (고난도, RPD 제한)
│   │
│   └── docs/                          # 이슈 로그 및 아키텍처 문서
│       ├── 00_overview.md
│       ├── 01~07_issues_*.md
│       └── 05_architecture_decisions.md
│
├── clustering/                        # 기사 클러스터링 모듈 (독립)
├── timeline/                          # 시간 기반 버스트 분석 (독립)
├── source/                            # Supabase 클라이언트 설정
│   └── config/
│       ├── supabase_client.py         # ai_test 소스 (SUPABASE_URL / SUPABASE_KEY)
│       └── supabase2_client.py        # articles/comments 소스 (SUPABASE2_URL / SUPABASE2_ANON_KEY)
├── data/                              # 샘플 CSV 데이터
└── feature_map/                       # ⚠️ DEPRECATED — labeling/features/ 로 통합됨
```

---

## 환경 설정

### 1. .venv 설치

```bash
cd C:\Users\user\Documents\40.WhatSurf
python -m venv .venv
.venv\Scripts\activate
pip install supabase python-dotenv transformers torch groq google-generativeai
```

### 2. .env 파일 (`AI/` 또는 프로젝트 루트)

```env
# Groq — 따옴표 없이, 콤마 구분 (공백 없음)
GROQ_API_KEYS=gsk_key1,gsk_key2
GROQ_MODELS=meta-llama/llama-3.3-70b-versatile,qwen/qwen3-32b

# Gemini — 따옴표 없이, 콤마 구분
GEMINI_API_KEYS=AIza_key1,AIza_key2
GEMINI_PRO_MODELS=gemini-2.5-pro,gemini-2.0-flash

# Supabase (ai_test 소스)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...

# Supabase2 (articles / article_features / comments 소스)
SUPABASE2_URL=https://yyy.supabase.co
SUPABASE2_ANON_KEY=eyJ...

# 로컬 모델 (선택 — 기본값으로 자동 설정)
EMOTION_MODEL=hun3359/klue-bert-base-sentiment
```

> **주의**: `.env` 값에 따옴표 금지. `KEY='a','b'` → ❌ / `KEY=a,b` → ✅

---

## 실행 방법

### 반드시 `.venv` Python으로 실행

```bash
# Windows
C:\Users\user\Documents\40.WhatSurf\.venv\Scripts\python.exe AI/labeling/run_labeling.py [옵션]

# 또는 venv 활성화 후
.venv\Scripts\activate
python AI/labeling/run_labeling.py [옵션]
```

---

## 1. 라벨링 실행 (`run_labeling.py`)

### 소스: ai_test 테이블 (keyword 기반)

```bash
# 기본 실행 (keyword로 조회)
python AI/labeling/run_labeling.py --keyword 의대정원

# 피처 선택
python AI/labeling/run_labeling.py --keyword 의대정원 \
    --features stance loaded_words body_depth frame logic bias omission

# 개수 제한 + 배치 크기 조정
python AI/labeling/run_labeling.py --keyword 의대정원 --limit 50 --batch_size 5

# 중단된 라벨링 재개 (기존 파일 있으면 로드)
python AI/labeling/run_labeling.py --keyword 의대정원 --resume

# Gemini 하루 한도 제어 (기본 200)
python AI/labeling/run_labeling.py --keyword 의대정원 --max_gemini 100

# 댓글 라벨링 건너뜀
python AI/labeling/run_labeling.py --keyword 의대정원 --skip_comments

# 병렬 실행 끄기 (디버깅용)
python AI/labeling/run_labeling.py --keyword 의대정원 --no_parallel --limit 5
```

### 소스: Supabase2 articles 테이블 (query_id 기반)

```bash
# query_id로 조회
python AI/labeling/run_labeling.py --source supabase2 --query_id Q_001

# 개수 제한
python AI/labeling/run_labeling.py --source supabase2 --query_id Q_001 --limit 30
```

### 출력 파일

```
AI/labeling/output/
├── {keyword}_{article_id}_labeled.json   # 기사별 라벨 결과
└── reports/
    ├── run_report_{timestamp}.json        # 실행 요약 (시간/API 호출/품질)
    └── label_dist_{timestamp}.json        # 레이블 분포 통계
```

---

## 2. DB 업로드 (`upload_to_db.py`)

`labeled.json` 결과를 Supabase2의 `article_features` / `comments` 테이블에 업로드.

```bash
# 기사 피처만 업로드
python AI/labeling/upload_to_db.py \
    --labeled_dir AI/labeling/output \
    --mode articles

# 댓글 감정/단어만 업로드
python AI/labeling/upload_to_db.py \
    --labeled_dir AI/labeling/output \
    --mode comments

# 전체 (articles + comments)
python AI/labeling/upload_to_db.py \
    --labeled_dir AI/labeling/output \
    --mode all
```

### DB 매핑 (article_features 테이블)

| labeled.json 필드 | DB 컬럼 | 타입 |
|---|---|---|
| `article_id` | `article_id` | integer |
| `frame` | `frame_id` | integer (1~7) |
| `logic` | `logic_id` | integer (1~6) |
| `bias_x` | `bias_x` | float |
| `bias_y` | `bias_y` | float |
| `omission_risk` | `omission_risk` | integer (-1/0/1) |
| `body_depth` | `body_depth` | float |
| `stance_score` | `stance_score` | float |
| `art_words` | `art_words` | jsonb |

### DB 매핑 (comments 테이블)

| labeled.json 필드 | DB 컬럼 | 타입 |
|---|---|---|
| `cmt_emotion` | `cmt_emotion` | jsonb (top 10 감정) |
| `cmt_words` | `cmt_words` | jsonb |

---

## 3. 피처 설명

| 피처 | 처리 방식 | 모델/도구 | 설명 |
|---|---|---|---|
| `stance_score` | 로컬 HuggingFace | KR-FinBert-SC | -1.0(비판) ~ +1.0(우호적) |
| `art_words` | 로컬 규칙 기반 | BIAS_WORDS 사전 | 편향 단어 목록 |
| `body_depth` | 수식 기반 | 없음 | 본문 정보 깊이 0~1 (길이·다양성·인용·수치·구조) |
| `frame` | Groq API | Llama 3.3 70B | 7가지 프레임 분류 (int 1~7) |
| `logic` | Groq API | Llama 3.3 70B | 6가지 논리 유형 (int 1~6) |
| `bias_x` / `bias_y` | Groq API | Llama 3.3 70B | 2D 편향 좌표 |
| `omission_risk` | Gemini API | Gemini 2.5 Pro | 클러스터 대비 누락도 (-1/0/1) |
| `cmt_emotion` | 로컬 HuggingFace | klue-bert-sentiment | 댓글 top 10 감정 JSON |
| `cmt_words` | 로컬 규칙 기반 | BIAS_WORDS 사전 | 댓글 편향 단어 목록 |

### frame 레이블 (1~7)
1. 사건 원인 집중 / 2. 갈등·대립 강조 / 3. 개인 사례 중심
4. 경제적 영향 강조 / 5. 윤리·도덕 판단 / 6. 안전·안보 위협 / 7. 권리·인권 강조

### logic 레이블 (1~6)
1. 정책적 비난 / 2. 전문가 견해 / 3. 피해자 서사
4. 파급효과 / 5. 해결책 제시 / 6. 사실·정보 전달

### omission_risk 값
`high → 1` / `mid → 0` / `low → -1`

---

## 4. 실행 흐름

```
[1] Supabase 로드 (ai_test 또는 supabase2 선택, 페이지네이션)
[2] 기사 라벨링
    ├─ 로컬 순차: body_depth → stance → loaded_words
    └─ API 병렬:  Groq (frame+logic+bias) ‖ Gemini (omission)
[3] 댓글 라벨링 (로컬 병렬: emotion + loaded_words)
[4] labeled.json 저장 (배치 단위 즉시 저장, 중단 복구 가능)
[5] 연구 리포트 저장 (run_report / label_dist)
[6] upload_to_db.py 로 Supabase2 업로드 (별도 실행)
```

---

## 5. 자주 쓰는 옵션 요약

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--keyword` | (필수, ai_test) | ai_test.keyword 컬럼 값 |
| `--source` | `ai_test` | `ai_test` 또는 `supabase2` |
| `--query_id` | (필수, supabase2) | articles.query_id 값 |
| `--features` | 전체 | 실행할 피처 (공백 구분) |
| `--limit` | `0` (전체) | 가져올 기사 수 |
| `--batch_size` | `5` | LLM 배치 크기 |
| `--max_gemini` | `200` | Gemini 하루 최대 처리 수 |
| `--out_dir` | `./output` | 결과 저장 폴더 |
| `--resume` | `False` | 기존 labeled.json 이어서 작업 |
| `--skip_comments` | `False` | 댓글 라벨링 건너뜀 |
| `--no_parallel` | `False` | Groq+Gemini 순차 실행 (디버깅용) |

---

## 6. 트러블슈팅

### ImportError: cannot import name 'genai'
→ `.venv` Python으로 실행하지 않았을 때 발생.
```bash
C:\Users\user\Documents\40.WhatSurf\.venv\Scripts\python.exe AI/labeling/run_labeling.py ...
```

### frame/logic 결과가 모두 None
→ LLM 모델이 숫자가 아닌 한글 텍스트를 출력했거나 JSON 파싱 실패.
→ `.env`의 `GROQ_MODELS` 첫 번째 모델이 한국어 지원 모델인지 확인.
→ 배치 실패 시 자동으로 1개씩 재시도함.

### Gemini RPD 한도 초과
→ `--max_gemini 50` 으로 하루 소비량 제한.
→ `--features stance loaded_words body_depth frame logic bias` (omission 제외) 로 실행.

### DB 업로드 PK 충돌
→ `upload_to_db.py`는 SELECT → upsert 방식으로 자동 처리됨 (중복 실행 안전).
