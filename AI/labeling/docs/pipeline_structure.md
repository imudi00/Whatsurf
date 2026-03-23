# 통합 파이프라인 구조 문서

---

## 1. 전체 실행 흐름

```
[Source]           [run_labeling.py]              [Output]             [upload_to_db.py]
Supabase           ┌─────────────────────────┐    labeled.json    →   Supabase2
  ai_test      ──▶ │  1. 데이터 로드           │
  articles     ──▶ │  2. 기사 피처 라벨링       │──▶ *_labeled.json
  comments     ──▶ │  3. 댓글 피처 라벨링       │──▶ reports/
                   │  4. 리포트 저장            │
                   └─────────────────────────┘
```

---

## 2. 데이터 소스

### 소스 1: `ai_test` (Supabase)
```python
# 로드 함수: load_by_keyword(keyword, limit)
# 테이블: ai_test
# 조건: WHERE keyword = '{keyword}'
# 컬럼: id, title, body, comments (JSONB)

Row = {
    "id":       123,
    "title":    "의대 증원 강행",
    "body":     "정부가 2000명 증원을 확정...",
    "comments": [
        {"id": "c1", "content": "이게 말이 되냐"},
        {"id": "c2", "content": "잘 됐다"}
    ]
}
```

### 소스 2: `articles` + `comments` (Supabase2)
```python
# 로드 함수: load_by_query_id(query_id, limit)
# 테이블: articles + comments (article_id 공유)

Article = {
    "id":        456,
    "query_id":  "Q_001",
    "title":     "...",
    "body_text": "..."  → body로 정규화
}
Comment = {
    "id":          789,
    "article_id":  456,
    "cmt_content": "댓글 내용"  → content로 정규화
}
```

---

## 3. 기사 라벨링 파이프라인

### 3.1 텍스트 준비

```python
text = f"{row['title']}\n\n{row['body']}"
```

### 3.2 로컬 순차 블록

```
texts (List[str])
    │
    ├─▶ [body_depth]
    │       local/body_depth.compute_body_depth(text)
    │       → body_depth: float, body_depth_label: str
    │
    ├─▶ [stance]
    │       local/stance_labeler.label_stance_batch(texts)
    │       → stance_score: float, stance_label: str
    │
    └─▶ [loaded_words]
            local/loaded_words_labeler.label_loaded_words_batch(texts)
            → art_words: list, loaded_word_density: float, is_biased: bool
```

### 3.3 Groq 블록 (병렬)

```
features/preprocessor.build_article_struct(text) → struct
    │
    ├─▶ [frame / logic]  — batch_size 단위
    │       groq/frame_labeler.label_frame_logic_batch(structs)
    │       → frame: str, logic: str, frame_reason: str, logic_reason: str
    │       ⚠️ 배치 실패 시 1개씩 재시도
    │
    └─▶ [bias]  — batch_size 단위
            groq/bias_labeler.label_bias_batch(structs)
            → bias_x: float, bias_y: float, bias_reason: str
            ⚠️ 배치 실패 시 1개씩 재시도
```

### 3.4 Gemini 블록 (병렬)

```
features/keyword_extractor.extract_features(text) → NER entities
    │
    └─▶ [omission] — batch_size 단위
            gemini/omission_labeler.label_omission_batch(articles, cluster_entities)
            → omission_risk: str, omission_reason: str
            ⚠️ 배치 실패 시 1개씩 재시도
```

### 3.5 Groq + Gemini 병렬 실행

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    future_groq   = executor.submit(_run_groq_block,   ...)
    future_gemini = executor.submit(_run_gemini_block, ...)
    groq_result   = future_groq.result()
    gemini_result = future_gemini.result()
```

---

## 4. 댓글 라벨링 파이프라인

```
row["comments"] (List[{id, content}])
    │
    ├─▶ [emotion] — 댓글 텍스트 배치
    │       local/emotion_labeler.label_emotion_batch(texts)
    │       → primary_emotion: str, emotion_intensity: float, emotion_probs: dict
    │
    └─▶ [loaded_words] — 댓글 텍스트 배치
            local/loaded_words_labeler.label_loaded_words_batch(texts)
            → loaded_words: list, loaded_word_density: float
```

---

## 5. 출력 파일 구조

### 5.1 기사 라벨 결과 (`*_labeled.json`)

```json
{
  "article_id":    "123",
  "title":         "의대 증원 강행",
  "body_depth":    0.6123,
  "body_depth_label": "medium",
  "stance_score":  -0.4821,
  "stance_label":  "비판적",
  "art_words":     ["독재", "재앙", "선동"],
  "loaded_word_density": 0.1250,
  "is_biased":     true,
  "frame":         "갈등/대립 강조",
  "frame_reason":  "정부-의협 간 대립 구도 전면 배치",
  "logic":         "정책적 비난",
  "logic_reason":  "정부 정책의 부당성을 논거로 구성",
  "frame_model_used": "meta-llama/llama-3.3-70b-versatile",
  "bias_x":        -0.35,
  "bias_y":        0.20,
  "bias_rationale":"진보 성향 신호 강함, 일부 사실적 수치 인용",
  "bias_model_used": "meta-llama/llama-3.3-70b-versatile",
  "omission_risk": "mid",
  "omission_reason": "의협 측 입장 누락",
  "omission_model_used": "gemini-2.5-pro",
  "comments": [
    {
      "id":                "c1",
      "primary_emotion":   "anger",
      "emotion_intensity": 0.8312,
      "cmt_words":         ["미쳤다", "최악"],
      "cmt_word_density":  0.0833
    }
  ]
}
```

### 5.2 리포트 (`reports/run_report_{ts}.json`)

```json
{
  "keyword":         "의대정원",
  "run_id":          "의대정원_20260322_143022",
  "n_articles":      50,
  "total_comments":  312,
  "api_calls":       {"groq": 20, "gemini_pro": 50},
  "key_rotations":   [{"from_key": 0, "to_key": 1, "reason": "rate_limit"}],
  "feature_times":   {"body_depth": 0.8, "stance": 45.2, "frame_logic": 120.3, ...},
  "label_dist":      {"frame": {"갈등/대립 강조": 18, "사건 원인 집중": 14, ...}, ...},
  "fallback_count":  {"frame": 2, "bias": 0, "omission": 1}
}
```

---

## 6. DB 업로드 파이프라인 (`upload_to_db.py`)

### 입력

```
labeled.json 파일들
    │
    ├─▶ labeled_to_feature_row()   → article_features 테이블 row
    └─▶ labeled_to_comment_rows()  → comments 테이블 rows
```

### 필드 매핑

#### article_features

| labeled.json | DB 컬럼 | 변환 |
|---|---|---|
| `article_id` | `article_id` | str → int (`_to_int()`) |
| `frame` | `frame_id` | str → int (FRAME_MAP 1~7) |
| `logic` | `logic_id` | str → int (LOGIC_MAP 1~6) |
| `bias_x` | `bias_x` | float |
| `bias_y` | `bias_y` | float |
| `omission_risk` | `omission_risk` | str → int (high=1, mid=0, low=-1) |
| `body_depth` | `body_depth` | float |
| `stance_score` | `stance_score` | float |
| `art_words` | `art_words` | list → jsonb |
| — | `created_at` | datetime.now() |

#### comments

| labeled.json | DB 컬럼 | 변환 |
|---|---|---|
| `emotion_probs` top 10 | `cmt_emotion` | jsonb |
| `loaded_words` | `cmt_words` | jsonb |

### Upsert 전략 (article_features)

```
SELECT id FROM article_features WHERE article_id = {aid}
    │
    ├─ 존재 → row에 id 주입 → upsert (UPDATE 분기)
    └─ 없음 → id 없이 upsert  (INSERT 분기)
```

---

## 7. 소스별 실행 커맨드 요약

```bash
# ai_test 소스
python AI/labeling/run_labeling.py --keyword 의대정원

# supabase2 소스
python AI/labeling/run_labeling.py --source supabase2 --query_id Q_001

# 특정 피처만
python AI/labeling/run_labeling.py --keyword 의대정원 \
    --features body_depth stance loaded_words

# DB 업로드
python AI/labeling/upload_to_db.py \
    --labeled_dir AI/labeling/output --mode all

# 실험 (정답 없이 분포 분석)
python AI/labeling/experiments/rule_based/body_depth_tuner.py \
    --labeled_dir AI/labeling/output --mode distribution
```

---

## 8. 모듈 의존 관계

```
run_labeling.py
 ├── source/config/supabase_client.py
 ├── source/config/supabase2_client.py
 ├── labeling/research_report.py
 ├── labeling/features/preprocessor.py
 │       └── labeling/features/text_utils.py
 ├── labeling/features/keyword_extractor.py
 ├── labeling/local/body_depth.py              (독립)
 ├── labeling/local/stance_labeler.py          (torch + transformers)
 ├── labeling/local/loaded_words_labeler.py
 │       └── labeling/features/loaded_words.py
 ├── labeling/local/emotion_labeler.py         (torch + transformers)
 ├── labeling/groq/frame_labeler.py
 │       └── labeling/groq/groq_client.py
 ├── labeling/groq/bias_labeler.py
 │       └── labeling/groq/groq_client.py
 └── labeling/gemini/omission_labeler.py
         └── labeling/gemini/gemini_client.py

upload_to_db.py
 └── source/config/supabase2_client.py
```
