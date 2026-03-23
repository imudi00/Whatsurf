# Issue Log 06 — Supabase2 연동 및 DB 업로드 오류

## 배경

기존 `ai_test` 소스와 별도로 Supabase2 DB(`articles` / `article_features` / `comments` 테이블)를
라벨링 파이프라인에 연동하면서 발생한 이슈들을 기록.

---

## 이슈 1: frame / logic 매핑 전체 None

### 증상
```
labeled.json 결과 확인 시 frame, logic 값이 100% None
upload_to_db.py 실행 후 DB에 frame_id, logic_id 모두 null
```

### 원인
`upload_to_db.py`에서 `frame_codes`, `logic_codes` 테이블을 Supabase2에 조회해 매핑 딕셔너리를 구성하는 로직 사용.
해당 테이블이 존재하지 않거나 쿼리 실패 시 `_FRAME_MAP = {}`, `_LOGIC_MAP = {}`이 되어 모든 값이 None으로 매핑됨.

### 해결
DB 조회 로직을 완전히 제거하고, DB 이미지 기준 하드코딩 매핑으로 전환.

```python
FRAME_MAP = {
    "사건 원인 집중": 1, "갈등/대립 강조": 2, "개인 사례 중심": 3,
    "경제적 영향 강조": 4, "윤리/도덕 판단": 5, "안전/안보 위협": 6, "권리/인권 강조": 7,
}
LOGIC_MAP = {
    "정책적 비난": 1, "전문가 견해": 2, "피해자 서사": 3,
    "파급효과": 4, "해결책 제시": 5, "사실/정보 전달": 6,
}
OMISSION_MAP = {"low": -1, "mid": 0, "med": 0, "medium": 0, "high": 1}
```

---

## 이슈 2: omission_risk "medium" → None 매핑 실패

### 증상
```json
"omission_risk": null
```
LLM은 `"medium"`을 출력했지만 OMISSION_MAP 에는 `"mid"` 키만 존재.

### 원인
`omission_labeler.py` 프롬프트에 `low/mid/high`로 지정했으나,
Gemini 모델이 자연스럽게 `"medium"`을 출력하는 경우가 빈번함.

### 해결
1. OMISSION_MAP에 `"medium"`, `"med"` 키 추가 (방어적 처리)
2. `omission_labeler.py` 프롬프트 내 예시를 `mid`로 통일 (근본 해결)

```python
OMISSION_MAP = {"low": -1, "mid": 0, "med": 0, "medium": 0, "high": 1}
```

```python
# omission_labeler.py 프롬프트
"omission_risk: low(핵심 대부분 포함) / mid(일부 누락) / high(핵심 다수 누락)\n\n"
```

---

## 이슈 3: article_features upsert PK 충돌 (3단계)

### 3-1) `42P10: no unique constraint`

#### 증상
```
postgrest.exceptions.APIError: {'code': '42P10',
  'message': 'there is no unique or exclusion constraint matching the ON CONFLICT specification'}
```

#### 원인
`supabase.table("article_features").upsert(rows, on_conflict="article_id")` 호출 시
`article_id` 컬럼에 UNIQUE constraint가 없어 Postgres가 ON CONFLICT 처리 불가.

#### 시도 1: delete + insert
```python
supabase2.table("article_features").delete().in_("article_id", ids).execute()
supabase2.table("article_features").insert(rows).execute()
```
→ 타입 불일치로 DELETE 필터 미동작, 이후 INSERT에서 PK 중복 발생.

---

### 3-2) `23505: duplicate key value violates unique constraint "article_features_pkey"`

#### 증상
```
postgrest.exceptions.APIError: {'code': '23505',
  'message': 'duplicate key value violates unique constraint "article_features_pkey"',
  'details': 'Key (id)=(2) already exists.'}
```

#### 원인
`article_id`가 Python에서 문자열(`"2"`)로 전달되어 Postgres의 integer 컬럼과 타입 불일치.
DELETE 조건 `article_id IN (["2","3"])` 가 정수 컬럼과 매칭되지 않아 기존 행이 삭제 안 됨.
이후 INSERT 시 PK 자동증가값이 기존 `id=2`와 충돌.

#### 시도 2: `_to_int()` 추가 후 delete+insert
```python
def _to_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return val
```
→ DELETE는 성공했으나 INSERT 시 다시 `id` 충돌 발생 (시퀀스 리셋 미동작).

---

### 3-3) 최종 해결: SELECT → PK 주입 → upsert

#### 해결
INSERT/DELETE를 포기하고, 먼저 기존 행의 PK(`id`)를 SELECT로 조회한 뒤
새 행에 `id`를 명시 주입하여 upsert (PK 기준 UPDATE/INSERT 분기).

```python
def _upsert_batch(rows: list[dict]):
    article_ids = [row["article_id"] for row in rows]
    existing_resp = (
        supabase2.table("article_features")
        .select("id, article_id")
        .in_("article_id", article_ids)
        .execute()
    )
    existing_pk = {row["article_id"]: row["id"] for row in existing_resp.data}

    upsert_rows = []
    for row in rows:
        r = dict(row)
        pk = existing_pk.get(r["article_id"])
        if pk is not None:
            r["id"] = pk   # 기존 행: PK 명시 → UPDATE
        upsert_rows.append(r)

    supabase2.table("article_features").upsert(upsert_rows).execute()
```

---

## 이슈 4: comments 테이블 article_id 타입 불일치

### 증상
`upload_comments()` 실행 시 comments 조회 결과 빈 배열 또는 타입 오류.

### 원인
`labeled.json`의 `article_id`가 문자열로 저장되어 있고,
Supabase2 `comments` 테이블의 `article_id` 컬럼은 integer.

### 해결
`_to_int()` 함수로 업로드 전 article_id 정수 변환 일관 적용.

```python
"article_id": _to_int(d.get("article_id")),
```

---

## 이슈 5: cmt_emotion 값 형식 통일

### 1차 구현 (0.3 이상 필터)
```python
cmt_emotion = {k: round(v, 4) for k, v in probs.items() if v >= 0.3}
```
→ 문턱값이 높아 감정 모델 결과에 따라 cmt_emotion이 빈 dict `{}`가 되는 경우 발생.

### 최종 구현 (top 10 감정)
```python
top10 = sorted(
    ((k, v) for k, v in probs.items() if isinstance(v, (int, float))),
    key=lambda x: x[1], reverse=True
)[:10]
cmt_emotion = {k: round(v, 4) for k, v in top10}
```
→ 항상 최대 10개 감정이 저장됨. 비어있는 결과 방지.
