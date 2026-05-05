# labeling/qwen/call1_labeler.py
"""
adapter_call1: frame(1~7) + logic(1~6) 로컬 추론.

groq/frame_labeler.label_frame_logic_batch 와 동일한 출력 인터페이스.
입력은 raw article dict (title + body) 또는 build_article_struct() 결과 모두 허용.
"""
import json
from difflib import SequenceMatcher

import torch

from labeling.api_client_base import robust_json_parse
from .qwen_client import get_model, last_call_info

FRAME_LABELS = [
    "사건 원인 집중",
    "갈등/대립 강조",
    "개인 사례 중심",
    "경제적 영향 강조",
    "윤리/도덕 판단",
    "안전/안보 위협",
    "권리/인권 강조",
]

LOGIC_LABELS = [
    "정책적 비난",
    "전문가 견해",
    "피해자 서사",
    "파급효과",
    "해결책 제시",
    "사실/정보 전달",
]

_SYSTEM = """당신은 한국어 뉴스 기사를 분석하는 AI입니다.
기사를 읽고 아래 정의에 따라 JSON만 출력하세요.
다른 설명, 마크다운 블록, 줄바꿈 없이 JSON 한 줄만 출력합니다.

[frame 정의 - 1~7 정수]
1: 사건 원인 집중  2: 갈등/대립 강조  3: 개인 사례 중심
4: 경제적 영향 강조  5: 윤리/도덕 판단  6: 안전/안보 위협  7: 권리/인권 강조

[logic 정의 - 1~6 정수]
1: 정책적 비난  2: 전문가 견해  3: 피해자 서사
4: 파급효과  5: 해결책 제시  6: 사실/정보 전달"""

_FUZZY_THRESHOLD = 0.40
_MAX_NEW_TOKENS  = 64
_MAX_BODY_CHARS  = 1800   # max_seq_len(2048) 내에서 title + body 수용


def _match_label(val, labels: list[str], field: str = "") -> str | None:
    """groq/frame_labeler._match_label 과 동일한 매핑 로직."""
    if not val:
        return None
    val_str = str(val).strip()

    if val_str in labels:
        return val_str
    try:
        idx = int(val_str) - 1
        if 0 <= idx < len(labels):
            return labels[idx]
    except ValueError:
        pass
    for label in labels:
        if val_str in label or label in val_str:
            return label

    best  = max(labels, key=lambda l: SequenceMatcher(None, val_str, l).ratio())
    ratio = SequenceMatcher(None, val_str, best).ratio()
    if ratio >= _FUZZY_THRESHOLD:
        print(f"  [call1_labeler] {field} 유사도 매핑 ({ratio:.2f}): {val_str!r} → {best!r}")
        return best

    print(f"  [call1_labeler 경고] {field} 매핑 실패 (유사도 {ratio:.2f}): {val_str!r}")
    return None


def _extract_text(art: dict) -> tuple[str, str]:
    """
    article dict 에서 (title, body) 추출.
    build_article_struct() 결과(headline/lead)와 raw dict(title/body) 모두 허용.
    """
    title = art.get("title") or art.get("headline") or ""
    body  = (
        art.get("body")
        or art.get("lead")
        or art.get("body_snippet")
        or ""
    )
    return title, body[:_MAX_BODY_CHARS]


def label_frame_logic_batch(articles: list[dict]) -> list[dict]:
    """
    기사 N개를 순차 추론 (adapter_call1 기준 1건씩 호출).

    Args:
        articles: [{"title": ..., "body": ...}, ...]
                  또는 build_article_struct() 결과 (headline, lead 키 허용)

    Returns:
        [{"idx": i, "frame": str|None, "frame_reason": str,
          "logic": str|None, "logic_reason": str}, ...]
    """
    model, tokenizer = get_model("call1")
    out = []

    for i, art in enumerate(articles):
        title, body = _extract_text(art)

        messages = [
            {"role": "system",    "content": _SYSTEM},
            {"role": "user",      "content": f"기사: {title}\n{body}"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(
            output_ids[0][inputs.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        parsed = robust_json_parse(raw)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            parsed = {}

        frame = _match_label(parsed.get("frame"), FRAME_LABELS, "frame")
        logic = _match_label(parsed.get("logic"), LOGIC_LABELS, "logic")

        out.append({
            "idx":          i,
            "frame":        frame,
            "frame_reason": parsed.get("frame_reason", ""),
            "logic":        logic,
            "logic_reason": parsed.get("logic_reason", ""),
        })

    return out


def label_frame_logic(article: dict) -> dict:
    return label_frame_logic_batch([article])[0]
