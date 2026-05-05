# labeling/qwen/qwen_client.py
"""
Qwen2.5-7B-Instruct 싱글톤 로더 + LoRA adapter swap.

어댑터 경로: AI/labeling/adapters/{name}/
  name: "call1" | "call2" | "call3"
"""

from pathlib import Path
from threading import Lock

_ADAPTERS_DIR = Path(__file__).resolve().parent.parent / "adapters"
_BASE_MODEL   = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
_MAX_SEQ_LEN  = 2048

_model                  = None
_tokenizer              = None
_current_adapter: str | None = None
_lock                   = Lock()

last_call_info: dict = {}


def _ensure_base_loaded() -> None:
    global _model, _tokenizer
    if _model is not None:
        return
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "unsloth 미설치. GPU 서버에서 `pip install unsloth` 후 사용 가능."
        ) from exc

    print(f"[qwen_client] 베이스 모델 로딩: {_BASE_MODEL}")
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=_BASE_MODEL,
        max_seq_length=_MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(_model)
    print(f"[qwen_client] 베이스 모델 로드 완료")


def get_model(adapter_name: str):
    """
    adapter_name 의 LoRA 어댑터를 로드하고 (model, tokenizer) 반환.
    이미 로드된 어댑터면 reload 생략 (swap 비용 절약).

    Args:
        adapter_name: "call1" | "call2" | "call3"

    Returns:
        (model, tokenizer)
    """
    global _current_adapter

    with _lock:
        _ensure_base_loaded()

        if _current_adapter != adapter_name:
            adapter_path = _ADAPTERS_DIR / adapter_name
            if not adapter_path.exists():
                raise FileNotFoundError(
                    f"어댑터 없음: {adapter_path}\n"
                    f"먼저 train/train_{adapter_name}.py 로 학습 후 저장하세요."
                )
            _model.load_adapter(str(adapter_path))
            _current_adapter = adapter_name
            print(f"[qwen_client] adapter swap → {adapter_name}")

        last_call_info["model"] = f"qwen2.5-7b/{adapter_name}"
        return _model, _tokenizer
