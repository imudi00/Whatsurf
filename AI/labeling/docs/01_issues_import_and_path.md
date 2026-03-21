# Issue Log 01 — Import 오류 및 sys.path 문제

## 배경

`AI/` 하위에 `feature_map/`, `labeling/`, `llm/` 세 패키지가 분리되어 있고,
각 패키지가 서로의 모듈을 참조해야 하는 구조. 설치 없이 `sys.path` 조작만으로 실행.

---

## 이슈 1: `ImportError: cannot import name 'genai' from 'google'`

### 증상
```
ImportError: cannot import name 'genai' from 'google'
```

### 원인
`google-genai` 패키지가 시스템 Python에 없고, 프로젝트는 `.venv`를 사용 중.
시스템 Python으로 실행하면 `.venv` 패키지를 인식 못 함.

### 해결
```bash
# 반드시 .venv Python으로 실행
C:\Users\user\Documents\40.WhatSurf\.venv\Scripts\python.exe AI/labeling/run_labeling.py
```

---

## 이슈 2: `ImportError: attempted relative import with no known parent package`

### 증상
```
ImportError: attempted relative import with no known parent package
  File "feature_map/stance/src/feature_map/preprocessor.py"
    from .text_utils import ...
```

### 원인
`preprocessor.py`는 내부적으로 `from .text_utils import ...` (상대 임포트)를 사용.
파일을 직접 `sys.path`에 추가한 뒤 `import preprocessor`로 불러오면
상대 임포트의 기준 패키지(parent package)가 없어서 실패.

### 해결
`feature_map/stance/`, `feature_map/context/`, `feature_map/emotion/` 각 디렉토리에
`__init__.py` 생성 → 패키지로 인식시킨 뒤 **풀 패키지 경로**로 임포트.

```python
# ❌ 잘못된 방식
sys.path.insert(0, "AI/feature_map/stance/src/feature_map/")
from preprocessor import build_article_struct

# ✅ 올바른 방식
sys.path.insert(0, "AI/")   # AI/ 한 곳만 추가
from feature_map.stance.src.feature_map.preprocessor import build_article_struct
```

---

## 이슈 3: Groq SDK vs 로컬 `groq/` 폴더 충돌

### 증상
```
pip install groq  →  "Requirement already satisfied"
실행 시 →  ImportError: cannot import name 'Groq' from 'groq'
```

### 원인
Python은 스크립트 실행 시 **스크립트 디렉토리를 `sys.path[0]`에 자동 추가**.
`run_labeling.py`가 `AI/labeling/`에 있으므로, `AI/labeling/`이 `sys.path[0]`가 됨.
`from groq import Groq` 시 SDK가 아닌 로컬 `AI/labeling/groq/` 폴더를 참조.

### 해결
`run_labeling.py` 시작 부분에서 `AI/labeling/` 경로를 명시적으로 제거.

```python
_HERE = Path(__file__).resolve().parent   # AI/labeling/
_AI   = _HERE.parent                      # AI/

# 스크립트 디렉토리가 sys.path에 자동으로 들어가므로 명시적 제거
_here_str = str(_HERE)
while _here_str in sys.path:
    sys.path.remove(_here_str)

sys.path.insert(0, str(_AI))  # AI/ 만 추가
```

### 원칙
> **진입점(run_*.py) 은 `AI/`만 `sys.path`에 추가.**
> 서브패키지 내부 파일들은 `from .module import x` 상대 임포트 사용.

---

## 이슈 4: 함수명 불일치 (caller vs callee)

### 증상
```
ImportError: cannot import name 'label_emotion_batch' from 'emotion_labeler'
ImportError: cannot import name 'label_omission_risk_batch'
```

### 원인
`run_labeling.py`가 호출하는 함수명과 실제 labeler에 정의된 함수명이 달랐음.

| caller 호출 | labeler 실제 함수 |
|---|---|
| `label_emotion_batch` | 존재 안 함 |
| `label_omission_risk_batch` | `label_omission_batch` |
| `res["bias_rationale"]` | `res["bias_reason"]` |

### 해결
각 `local/` 모듈에 `run_labeling.py` 인터페이스 규격에 맞는 래퍼 함수 추가.

```python
# local/emotion_labeler.py
def label_emotion_batch(texts: list[str]) -> list[dict]:
    labeler = _get_labeler()
    return labeler.label_batch(texts)

# local/loaded_words_labeler.py
def label_loaded_words_batch(texts: list[str]) -> list[dict]:
    ...
```

---

## 이슈 5: `KeyError: 'id'` in omission_labeler

### 증상
```
KeyError: 'id'
  File "gemini/omission_labeler.py", line 29
    f"--- 기사 {i} (id={a['id']}) ---\n"
```

### 원인
`build_article_struct()` 출력 구조체에는 `headline`, `lead` 등이 있지만 `id` 키가 없음.
`omission_labeler`는 `{"id", "title", "body_snippet"}` 형태를 기대하는데,
`run_labeling.py`에서 struct를 그대로 넘겼음.

### 해결
`run_labeling.py`에서 `omission_labeler` 전용 dict를 별도 구성.

```python
articles = [
    {
        "id":           ids[j],
        "title":        rows[j]["title"],
        "body_snippet": rows[j]["body"][:400],
    }
    for j in range(len(rows))
]
```

`omission_labeler.py` 내부도 `.get()` fallback 추가.

```python
f"id={a.get('id', i)}"
f"제목: {a.get('title', a.get('headline', ''))}"
```
