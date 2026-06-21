# 파이프라인 Diagram

> 전체 라벨링 파이프라인의 데이터 흐름을 단계별 IN/OUT 예시와 함께 정리합니다.
> 모든 다이어그램은 Mermaid 문법으로 작성되어 복사·붙여넣기 후 즉시 렌더링됩니다.

---

## 1. 전체 흐름 개요

```mermaid
flowchart TD
    SRC1[(Supabase\nai_test)]
    SRC2[(Supabase2\narticles + comments)]

    subgraph LABELING["run_labeling.py"]
        LOAD[1. 데이터 로드]
        ART[2. 기사 라벨링]
        CMT[3. 댓글 라벨링]
        RPT[4. 리포트 저장]
    end

    OUT1[/"*_labeled.json"/]
    OUT2[/"reports/run_report_{ts}.json"/]

    subgraph UPLOAD["upload_to_db.py"]
        MAP1[labeled_to_feature_row]
        MAP2[labeled_to_comment_rows]
    end

    DB[(Supabase2\narticle_features\ncomments)]

    SRC1 -->|keyword| LOAD
    SRC2 -->|query_id| LOAD
    LOAD --> ART
    LOAD --> CMT
    ART --> OUT1
    CMT --> OUT1
    ART --> RPT
    CMT --> RPT
    RPT --> OUT2
    OUT1 --> MAP1
    OUT1 --> MAP2
    MAP1 --> DB
    MAP2 --> DB
```

---

## 2. 데이터 로드

```mermaid
flowchart LR
    subgraph SRC1["소스 1: ai_test"]
        Q1["load_by_keyword(keyword, limit)"]
    end

    subgraph SRC2["소스 2: articles + comments"]
        Q2["load_by_query_id(query_id, limit)"]
    end

    NORM["정규화\nbody_text → body\ncmt_content → content"]
    ROW["Row List\nList[{id, title, body, comments}]"]

    Q1 --> ROW
    Q2 --> NORM --> ROW
```

### IN

```
# 소스 1 (ai_test)
keyword = "의대정원"
limit   = 50

# 소스 2 (articles + comments)
query_id = "Q_001"
limit    = 50
```

### OUT

```python
[
    {
        "id":       123,
        "title":    "의대 증원 강행",
        "body":     "정부가 2000명 증원을 확정...",
        "comments": [
            {"id": "c1", "content": "이게 말이 되냐"},
            {"id": "c2", "content": "잘 됐다"}
        ]
    },
    ...
]
```

---

## 3. 기사 라벨링 파이프라인

### 3-A. 전체 흐름

```mermaid
flowchart TD
    ROW["Row\n{id, title, body, comments}"]
    TEXT["text = title + '\n\n' + body"]

    subgraph LOCAL["로컬 순차 블록"]
        BD["body_depth\n.compute_body_depth(text)"]
        ST["stance\n.label_stance_batch(texts)"]
        LW["loaded_words\n.label_loaded_words_batch(texts)"]
    end

    subgraph PREP["전처리"]
        STRUCT["build_article_struct(text)\n→ struct"]
        NER["extract_features(text)\n→ cluster_entities"]
    end

    subgraph GROQ["Groq 블록 (ThreadPoolExecutor)"]
        FL["label_frame_logic_batch(structs)\n→ frame, logic"]
        BI["label_bias_batch(structs)\n→ bias_x, bias_y"]
    end

    subgraph GEMINI["Gemini 블록 (ThreadPoolExecutor)"]
        OM["label_omission_batch(articles, cluster_entities)\n→ omission_risk"]
    end

    MERGE["결과 병합\n→ labeled dict"]

    ROW --> TEXT
    TEXT --> LOCAL
    TEXT --> PREP
    PREP --> GROQ
    PREP --> GEMINI
    LOCAL --> MERGE
    GROQ --> MERGE
    GEMINI --> MERGE
```

### 3-B. 로컬 블록 상세

```mermaid
flowchart LR
    TEXT["text: str\n제목+본문"]

    subgraph BD["body_depth.py"]
        BD1["compute_body_depth(text)"]
        BD2["describe_body_depth(score)"]
    end

    subgraph ST["stance_labeler.py\nsnunlp/KR-FinBert-SC"]
        ST1["label_stance_batch(texts)"]
    end

    subgraph LW["loaded_words_labeler.py"]
        LW1["label_loaded_words_batch(texts)"]
    end

    BD_OUT["body_depth: 0.6123\nbody_depth_label: 'medium'"]
    ST_OUT["stance_score: -0.4821\nstance_label: '비판적'"]
    LW_OUT["art_words: ['독재','재앙']\nloaded_word_density: 0.1250\nis_biased: true"]

    TEXT --> BD1 --> BD2 --> BD_OUT
    TEXT --> ST1 --> ST_OUT
    TEXT --> LW1 --> LW_OUT
```

**body_depth IN/OUT:**

```python
# IN
text = "의대 정원 2000명 증원 확정\n\n정부는 2025학년도부터 의대 정원을 2000명 늘리기로 했다..."

# OUT
{
    "body_depth":       0.6123,
    "body_depth_label": "medium"
}
```

**stance IN/OUT:**

```python
# IN
texts = ["정부의 의대 증원 결정은 매우 잘못되었다.", "새 정책이 발표되었다."]

# OUT
[
    {"stance_score": -0.6821, "stance_label": "비판적"},
    {"stance_score":  0.0312, "stance_label": "중립"}
]
```

**loaded_words IN/OUT:**

```python
# IN
texts = ["정부의 독재적 선동이 재앙을 부르고 있다.", "오늘 국회에서 예산안을 논의했다."]

# OUT
[
    {
        "loaded_words":        ["독재", "선동", "재앙"],
        "loaded_word_density": 0.1667,
        "is_biased":           True,
        "tier":                1
    },
    {
        "loaded_words":        ["오늘", "국회"],   # Tier-3 폴백
        "loaded_word_density": 0.0,
        "is_biased":           False,
        "tier":                3
    }
]
```

---

### 3-C. Groq 블록 상세

```mermaid
flowchart LR
    TEXT["text: str"]

    STRUCT["build_article_struct(text)"]

    STRUCT_OUT["{headline, lead,\nquotes, numbers,\njudgment_words,\nsources,\nsampled_sentences}"]

    subgraph PARALLEL["ThreadPoolExecutor(max_workers=2)"]
        FL["label_frame_logic_batch(structs)\ngroq/frame_labeler.py"]
        BI["label_bias_batch(structs)\ngroq/bias_labeler.py"]
    end

    FL_OUT["frame: '갈등/대립 강조'\nframe_reason: '...'\nlogic: '정책적 비난'\nlogic_reason: '...'\nframe_model_used: 'llama-3.3-70b'"]
    BI_OUT["bias_x: -0.35\nbias_y: 0.20\nbias_reason: '...'\nbias_model_used: 'llama-3.3-70b'"]

    TEXT --> STRUCT --> STRUCT_OUT
    STRUCT_OUT --> FL --> FL_OUT
    STRUCT_OUT --> BI --> BI_OUT
```

**build_article_struct IN/OUT:**

```python
# IN
text = "의대 증원 강행…교육부, 대학에 최후통첩\n\n정부가 2025학년도 의대 정원을..."

# OUT
{
    "headline":          "의대 증원 강행…교육부, 대학에 최후통첩",
    "lead":              "정부가 2025학년도 의대 정원을 2000명 늘리기로 확정하면서...",
    "quotes":            ["전례 없는 일이다", "협의 없이 일방 통보했다"],
    "numbers":           ["2000명", "30%", "40억원"],
    "judgment_words":    ["강행비판", "우려촉구", "반발"],
    "sources":           ["교육부", "의협", "대학 관계자"],
    "sampled_sentences": ["정부가 결정했다.", "학교 측은 반대했다.", "사태는 장기화될 전망이다."],
    "total_sentences":   24
}
```

**label_frame_logic_batch IN/OUT:**

```python
# IN
structs = [
    {
        "headline":       "의대 증원 강행…교육부, 대학에 최후통첩",
        "lead":           "정부가 2025학년도 의대 정원을 2000명 늘리기로...",
        "quotes":         ["전례 없는 일이다", "협의 없이 일방 통보했다"],
        "numbers":        ["2000명", "30%"],
        "judgment_words": ["강행비판", "우려촉구"],
        ...
    }
]

# OUT
[
    {
        "frame":        "갈등/대립 강조",
        "frame_reason": "정부-의협 간 갈등 구도를 전면에 배치",
        "logic":        "정책적 비난",
        "logic_reason": "정부 정책의 부당성을 중심으로 논거 구성",
        "frame_model_used": "meta-llama/llama-3.3-70b-versatile"
    }
]
```

**label_bias_batch IN/OUT:**

```python
# IN
structs = [{"headline": "...", "lead": "...", ...}]

# OUT
[
    {
        "bias_x":      -0.35,
        "bias_y":       0.20,
        "bias_reason": "정부 정책에 비판적, 피해자 감성 중심이나 일부 수치 인용",
        "bias_model_used": "meta-llama/llama-3.3-70b-versatile"
    }
]
```

---

### 3-D. Gemini 블록 상세

```mermaid
flowchart LR
    TEXT["text: str"]

    NER["extract_features(text)\nkeyword_extractor.py\nKPF-BERT NER"]

    NER_OUT["cluster_entities:\n['교육부','의협','대학','학생','의사']"]

    ART_LIST["articles:\n[{id, title, body_snippet}]"]

    OM["label_omission_batch(\n  articles,\n  cluster_entities\n)\ngemini/omission_labeler.py"]

    OM_OUT["omission_risk: 'mid'\nomission_reason: '의협 측 입장 누락'\nomission_model_used: 'gemini-2.5-pro'"]

    TEXT --> NER --> NER_OUT
    TEXT --> ART_LIST
    NER_OUT --> OM
    ART_LIST --> OM
    OM --> OM_OUT
```

**extract_features IN/OUT:**

```python
# IN
text = "교육부가 의협의 반발에도 불구하고 2000명 증원을 강행했다..."

# OUT
{
    "sentences":       ["교육부가 의협의 반발에도...", "대학들은 일방 통보에 항의했다.", ...],
    "entities":        [
        {"word": "교육부", "entity_group": "ORG", "score": 0.99},
        {"word": "의협",   "entity_group": "ORG", "score": 0.98},
        {"word": "2000명", "entity_group": "QTY", "score": 0.95}
    ],
    "entity_count":    8,
    "entity_density":  0.33,
    "num_density":     0.12,
    "quote_count":     3,
    "source_count":    4,
    "paragraph_count": 6,
    "sentence_count":  24
}
# cluster_entities 추출 → ["교육부", "의협", "대학", "학생", "의사"]
```

**label_omission_batch IN/OUT:**

```python
# IN
articles = [
    {
        "id":           "123",
        "title":        "의대 증원 강행",
        "body_snippet": "정부가 2000명 증원을 강행하기로 했다..."
    }
]
cluster_entities = ["교육부", "의협", "대학", "학생", "의사"]

# OUT
[
    {
        "omission_risk":   "mid",
        "omission_reason": "의협의 반발 입장이 언급되지 않았고, 학생 피해 관련 수치 누락",
        "omission_model_used": "gemini-2.5-pro"
    }
]
```

---

### 3-E. 배치 실패 폴백

```mermaid
flowchart TD
    BATCH["label_xxx_batch(batch)"]
    OK{"성공?"}
    RESULT["results 반환"]
    SINGLE["1개씩 재시도\nfor struct in batch"]
    OK2{"단건 성공?"}
    APPEND_OK["results.append(res[0])"]
    APPEND_FAIL["results.append(\n  {frame: None, logic: None,\n   frame_reason: 'api_error', ...}\n)"]

    BATCH --> OK
    OK -- "✅" --> RESULT
    OK -- "❌ Exception" --> SINGLE
    SINGLE --> OK2
    OK2 -- "✅" --> APPEND_OK
    OK2 -- "❌" --> APPEND_FAIL
    APPEND_OK --> SINGLE
    APPEND_FAIL --> SINGLE
```

---

## 4. 댓글 라벨링 파이프라인

```mermaid
flowchart LR
    CMT_LIST["row['comments']\nList[{id, content}]"]

    TEXTS["texts = [c['content'] for c in comments]"]

    subgraph PARALLEL["병렬 또는 순차"]
        EM["label_emotion_batch(texts)\nlocal/emotion_labeler.py\nhun3359/klue-bert-base-sentiment"]
        LW["label_loaded_words_batch(texts)\nlocal/loaded_words_labeler.py"]
    end

    EM_OUT["primary_emotion: 'anger'\nemotion_intensity: 0.8312\nemotion_probs: {anger:0.83, ...}"]
    LW_OUT["loaded_words: ['미쳤다','최악']\nloaded_word_density: 0.0833"]

    MERGE_CMT["댓글 결과 병합\n→ comments list"]

    CMT_LIST --> TEXTS
    TEXTS --> EM --> EM_OUT --> MERGE_CMT
    TEXTS --> LW --> LW_OUT --> MERGE_CMT
```

**label_emotion_batch IN/OUT:**

```python
# IN
texts = ["이게 말이 되냐고 진짜 화난다", "좋은 결과가 나왔으면 좋겠다"]

# OUT
[
    {
        "primary_emotion":   "anger",
        "raw_label":         "분노",
        "emotion_intensity": 0.8312,
        "emotion_probs": {
            "anger":   0.8312,
            "disgust": 0.0921,
            "fear":    0.0412,
            "joy":     0.0180,
            "neutral": 0.0175
        }
    },
    {
        "primary_emotion":   "joy",
        "raw_label":         "기쁨",
        "emotion_intensity": 0.6240,
        "emotion_probs": {"joy": 0.6240, "neutral": 0.2100, ...}
    }
]
```

---

## 5. 출력 파일 생성

```mermaid
flowchart LR
    ART_RESULT["기사 라벨 결과\n{body_depth, stance_score,\n art_words, frame, logic,\n bias_x, bias_y, omission_risk,\n comments:[...]}"]

    JSON["*_labeled.json\n(기사 1개당 1 JSON 파일)"]

    RPT_DATA["리포트 데이터\n{n_articles, api_calls,\n key_rotations, feature_times,\n label_dist, fallback_count}"]

    REPORT["reports/run_report_{ts}.json"]

    ART_RESULT --> JSON
    ART_RESULT --> RPT_DATA
    RPT_DATA --> REPORT
```

**`*_labeled.json` 전체 예시:**

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

**`run_report_{ts}.json` 전체 예시:**

```json
{
  "keyword":        "의대정원",
  "run_id":         "의대정원_20260322_143022",
  "n_articles":     50,
  "total_comments": 312,
  "api_calls":      {"groq": 20, "gemini_pro": 50},
  "key_rotations":  [{"from_key": 0, "to_key": 1, "reason": "rate_limit"}],
  "feature_times":  {
    "body_depth": 0.8,
    "stance":     45.2,
    "frame_logic": 120.3,
    "bias":        118.7,
    "omission":    95.1,
    "emotion":     38.4,
    "loaded_words": 2.1
  },
  "label_dist": {
    "frame": {"갈등/대립 강조": 18, "사건 원인 집중": 14, "경제적 영향 강조": 8, ...},
    "logic": {"정책적 비난": 20, "전문가 견해": 12, "사실/정보 전달": 10, ...}
  },
  "fallback_count": {"frame": 2, "bias": 0, "omission": 1}
}
```

---

## 6. DB 업로드 파이프라인

```mermaid
flowchart TD
    JSON["*_labeled.json\n(output/ 디렉토리)"]

    subgraph UPLOAD["upload_to_db.py"]
        READ["labeled.json 로드\nload_labeled_dir()"]

        subgraph MAP["필드 매핑"]
            FEA["labeled_to_feature_row()\n→ article_features row"]
            CMT["labeled_to_comment_rows()\n→ comments rows"]
        end

        subgraph UPSERT["Upsert (article_features)"]
            SEL["SELECT id FROM article_features\nWHERE article_id = {aid}"]
            EXISTS{"존재?"}
            UPD["id 주입 → UPDATE"]
            INS["id 없이 → INSERT"]
        end
    end

    DB_ART[(article_features)]
    DB_CMT[(comments)]

    JSON --> READ --> MAP
    FEA --> SEL
    SEL --> EXISTS
    EXISTS -- "✅ 존재" --> UPD --> DB_ART
    EXISTS -- "❌ 없음" --> INS --> DB_ART
    CMT --> DB_CMT
```

**필드 매핑 IN/OUT:**

```python
# IN (labeled.json)
{
    "article_id":    "456",
    "frame":         "갈등/대립 강조",    # → FRAME_MAP[str] → 2
    "logic":         "정책적 비난",       # → LOGIC_MAP[str] → 1
    "bias_x":        -0.35,
    "bias_y":        0.20,
    "omission_risk": "mid",              # → 0
    "body_depth":    0.6123,
    "stance_score":  -0.4821,
    "art_words":     ["독재", "재앙"],
    "comments": [
        {
            "id":            "c1",
            "emotion_probs": {"anger": 0.8312, "disgust": 0.0921, ...},
            "cmt_words":     ["미쳤다", "최악"]
        }
    ]
}

# OUT → article_features row
{
    "article_id":    456,          # str → int
    "frame_id":      2,            # "갈등/대립 강조" → 2
    "logic_id":      1,            # "정책적 비난" → 1
    "bias_x":        -0.35,
    "bias_y":        0.20,
    "omission_risk": 0,            # "mid" → 0
    "body_depth":    0.6123,
    "stance_score":  -0.4821,
    "art_words":     ["독재", "재앙"],   # jsonb
    "created_at":    "2026-03-22T14:30:22"
}

# OUT → comments rows (id=c1)
{
    "cmt_emotion": {               # top 10, jsonb
        "anger":   0.8312,
        "disgust": 0.0921,
        "fear":    0.0412,
        "joy":     0.0180,
        "neutral": 0.0175
    },
    "cmt_words": ["미쳤다", "최악"]    # jsonb
}
```

---

## 7. 실험 프레임워크 흐름

```mermaid
flowchart TD
    OUTPUT["labeling/output/\n*_labeled.json"]
    GT["experiments/data/\nground_truth.jsonl"]

    subgraph RULE["rule_based/ 튜닝"]
        DIST["distribution 모드\n정답 불필요\n분포·분산 분석"]
        GRID["grid 모드\n정답 필요\nPearson/Spearman 기준\n최적 config 선택"]
        BEST["best config\n→ body_depth.py\n   WEIGHTS 교체"]
    end

    subgraph EVAL["model_eval/ 평가"]
        FRAME["FrameEvaluator\nLogicEvaluator\nAccuracy/F1/Kappa"]
        BIAS["BiasEvaluator\nPearson/사분면 정확도"]
        STANCE["StanceEvaluator\nOmissionEvaluator\n논조/누락도 F1"]
    end

    RESULTS["experiments/results/\n{name, feature, metrics, timestamp}.json"]

    OUTPUT --> DIST
    GT --> GRID
    OUTPUT --> GRID
    GRID --> BEST
    GT --> EVAL
    OUTPUT --> EVAL
    EVAL --> RESULTS
    DIST --> RESULTS
```

**Ground Truth JSONL 형식:**

```jsonl
{"id": "123", "text": "의대 증원 강행\n\n정부는 2025학년도부터...", "labels": {"body_depth": 0.75, "frame": 2, "logic": 1, "bias_x": -0.35, "bias_y": 0.20, "omission_risk": 0, "stance_score": -0.45, "loaded_words": ["독재", "재앙"]}}
{"id": "124", "text": "의협 총파업 예고\n\n의사협회는 오늘 총파업을...", "labels": {"body_depth": 0.52, "frame": 1, "logic": 3, "bias_x": -0.10, "bias_y": -0.30, "omission_risk": 1, "stance_score": -0.20, "loaded_words": ["파업", "압박"]}}
```

**ExperimentResult 저장 형식:**

```json
{
  "name":      "diversity_quote_heavy",
  "feature":   "body_depth",
  "config":    {
    "w_diversity": 0.35, "w_quotes": 0.30,
    "w_length": 0.15, "w_numerics": 0.10, "w_structure": 0.10,
    "length_soft_max": 2000
  },
  "metrics":   {"pearson": 0.812, "spearman": 0.803, "mae": 0.091},
  "n_samples": 50,
  "timestamp": "20260322_143022"
}
```

---

## 8. 모듈 의존 관계

```mermaid
graph LR
    RUN["run_labeling.py"]
    UP["upload_to_db.py"]

    SB1["source/config/\nsupabase_client.py"]
    SB2["source/config/\nsupabase2_client.py"]

    RR["labeling/\nresearch_report.py"]

    subgraph FEAT["labeling/features/"]
        PRE["preprocessor.py"]
        TU["text_utils.py"]
        KW["keyword_extractor.py"]
        LW_F["loaded_words.py"]
        BV["bias_vector.py"]
        OR["omission_risk.py"]
    end

    subgraph LOCAL["labeling/local/"]
        BD["body_depth.py"]
        SL["stance_labeler.py"]
        LL["loaded_words_labeler.py"]
        EL["emotion_labeler.py"]
    end

    subgraph GROQ_DIR["labeling/groq/"]
        GC["groq_client.py"]
        FL2["frame_labeler.py"]
        BL["bias_labeler.py"]
    end

    subgraph GEM["labeling/gemini/"]
        GEMC["gemini_client.py"]
        OL["omission_labeler.py"]
    end

    RUN --> SB1
    RUN --> SB2
    RUN --> RR
    RUN --> PRE
    RUN --> KW
    RUN --> BD
    RUN --> SL
    RUN --> LL
    RUN --> EL
    RUN --> FL2
    RUN --> BL
    RUN --> OL

    PRE --> TU
    LL --> LW_F
    FL2 --> GC
    BL --> GC
    OL --> GEMC

    UP --> SB2
```
