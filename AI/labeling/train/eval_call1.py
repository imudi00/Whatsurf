# labeling/train/eval_call1.py
"""
adapter_call1 검증 스크립트.

측정 지표:
  - frame Accuracy + Macro F1  (목표: F1 ≥ 0.75)
  - logic Accuracy + Macro F1  (목표: F1 ≥ 0.75)
  - JSON 파싱 성공률           (목표: ≥ 95%)

실행:
  python eval_call1.py
  python eval_call1.py --adapter ../adapters/call1 --data data/call1_val.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_LABELING_DIR = _HERE.parent
sys.path.insert(0, str(_LABELING_DIR.parent))  # AI/ → labeling.* import 가능

from labeling.api_client_base import robust_json_parse

FRAME_LABELS = [
    "사건 원인 집중", "갈등/대립 강조", "개인 사례 중심",
    "경제적 영향 강조", "윤리/도덕 판단", "안전/안보 위협", "권리/인권 강조",
]
LOGIC_LABELS = [
    "정책적 비난", "전문가 견해", "피해자 서사",
    "파급효과", "해결책 제시", "사실/정보 전달",
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

_MAX_BODY_CHARS = 1800
_MAX_NEW_TOKENS = 64


def _load_examples(path: str) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _infer_one(model, tokenizer, title: str, body: str) -> tuple[str, bool]:
    """
    단일 기사 추론 → (raw_output, json_parse_ok)
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": f"기사: {title}\n{body[:_MAX_BODY_CHARS]}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
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
        output_ids[0][inputs.shape[-1]:], skip_special_tokens=True
    ).strip()

    parsed = robust_json_parse(raw)
    ok = isinstance(parsed, dict) and ("frame" in parsed or "logic" in parsed)
    return raw, ok


def _int_label(val, labels: list[str]) -> int | None:
    """레이블 문자열 또는 정수 → 0-based index (None이면 None)."""
    if val is None:
        return None
    try:
        idx = int(str(val).strip()) - 1
        if 0 <= idx < len(labels):
            return idx
    except ValueError:
        pass
    val_str = str(val).strip()
    if val_str in labels:
        return labels.index(val_str)
    return None


def evaluate(adapter_dir: str, data_path: str, limit: int = 0) -> dict:
    from unsloth import FastLanguageModel
    from sklearn.metrics import accuracy_score, f1_score

    print(f"[eval_call1] 어댑터 로딩: {adapter_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    examples = _load_examples(data_path)
    if limit > 0:
        examples = examples[:limit]
    print(f"[eval_call1] 평가 샘플: {len(examples)}개")

    y_true_frame, y_pred_frame = [], []
    y_true_logic, y_pred_logic = [], []
    parse_ok = 0

    for i, ex in enumerate(examples):
        raw, ok = _infer_one(model, tokenizer, ex["title"], ex.get("body", ""))
        if ok:
            parse_ok += 1

        parsed = robust_json_parse(raw)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            parsed = {}

        pred_frame = _int_label(parsed.get("frame"), FRAME_LABELS)
        pred_logic = _int_label(parsed.get("logic"), LOGIC_LABELS)
        true_frame = _int_label(ex.get("frame"), FRAME_LABELS)
        true_logic = _int_label(ex.get("logic"), LOGIC_LABELS)

        if true_frame is not None and pred_frame is not None:
            y_true_frame.append(true_frame)
            y_pred_frame.append(pred_frame)
        if true_logic is not None and pred_logic is not None:
            y_true_logic.append(true_logic)
            y_pred_logic.append(pred_logic)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(examples)} 완료")

    parse_rate = parse_ok / len(examples) if examples else 0.0

    frame_acc = accuracy_score(y_true_frame, y_pred_frame) if y_true_frame else 0.0
    frame_f1  = f1_score(y_true_frame, y_pred_frame, average="macro", zero_division=0) if y_true_frame else 0.0
    logic_acc = accuracy_score(y_true_logic, y_pred_logic) if y_true_logic else 0.0
    logic_f1  = f1_score(y_true_logic, y_pred_logic, average="macro", zero_division=0) if y_true_logic else 0.0

    result = {
        "samples":           len(examples),
        "json_parse_rate":   round(parse_rate, 4),
        "frame_accuracy":    round(frame_acc, 4),
        "frame_macro_f1":    round(frame_f1, 4),
        "logic_accuracy":    round(logic_acc, 4),
        "logic_macro_f1":    round(logic_f1, 4),
        "frame_target_met":  frame_f1 >= 0.75,
        "logic_target_met":  logic_f1 >= 0.75,
        "parse_target_met":  parse_rate >= 0.95,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="adapter_call1 검증")
    ap.add_argument("--adapter", default=str(_LABELING_DIR / "adapters" / "call1"))
    ap.add_argument("--data",    default=str(_HERE / "data" / "call1_val.jsonl"))
    ap.add_argument("--limit",   type=int, default=0, help="평가 샘플 수 제한 (0=전체)")
    args = ap.parse_args()

    result = evaluate(args.adapter, args.data, args.limit)

    print("\n" + "=" * 50)
    print("  adapter_call1 평가 결과")
    print("=" * 50)
    print(f"  샘플 수         : {result['samples']}")
    print(f"  JSON 파싱률     : {result['json_parse_rate']:.1%}  "
          f"{'✅' if result['parse_target_met'] else '❌'} (목표 ≥95%)")
    print(f"  frame Accuracy  : {result['frame_accuracy']:.1%}")
    print(f"  frame Macro F1  : {result['frame_macro_f1']:.3f}  "
          f"{'✅' if result['frame_target_met'] else '❌'} (목표 ≥0.75)")
    print(f"  logic Accuracy  : {result['logic_accuracy']:.1%}")
    print(f"  logic Macro F1  : {result['logic_macro_f1']:.3f}  "
          f"{'✅' if result['logic_target_met'] else '❌'} (목표 ≥0.75)")
    print("=" * 50)

    all_pass = all([result["frame_target_met"], result["logic_target_met"],
                    result["parse_target_met"]])
    print(f"  종합: {'✅ 모든 목표 달성' if all_pass else '❌ 미달 항목 있음 — 재학습 필요'}")
    print("=" * 50 + "\n")

    # JSON 저장
    out_path = Path(args.adapter) / "eval_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장: {out_path}")


if __name__ == "__main__":
    main()
