# labeling/train/train_call1.py
"""
adapter_call1 학습 스크립트 — frame(1~7) + logic(1~6) 분류.

환경: GPU 서버 (RTX 4080 16GB 이상), unsloth + trl + peft 설치 필수.

실행:
  python train_call1.py
  python train_call1.py --data data/call1_train.jsonl --eval data/call1_val.json
  python train_call1.py --epochs 2 --lr 5e-5

하이퍼파라미터 근거 (PDF 모델 연구 & 파인튜닝 설계 문서 §6.2):
  lora_r=32, lora_alpha=64 (r×2), dropout=0.05 (분류형)
  lr=1e-4, batch=4, grad_accum=8 (effective batch=32), epochs=3
  target_modules: Attention + MLP 전체 (QLoRA-All)
"""
import argparse
import json
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_LABELING_DIR = _HERE.parent


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="adapter_call1 QLoRA 학습")
    ap.add_argument("--model",      default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    ap.add_argument("--data",       default=str(_HERE / "data" / "call1_train.json"))
    ap.add_argument("--eval",       default=str(_HERE / "data" / "call1_val.json"))
    ap.add_argument("--output",     default=str(_LABELING_DIR / "adapters" / "call1"))
    ap.add_argument("--max_seq",    type=int,   default=2048)
    ap.add_argument("--lora_r",     type=int,   default=32)
    ap.add_argument("--lora_alpha", type=int,   default=64)
    ap.add_argument("--dropout",    type=float, default=0.05)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--batch",      type=int,   default=4)
    ap.add_argument("--grad_accum", type=int,   default=8)
    ap.add_argument("--epochs",     type=int,   default=3)
    return ap.parse_args()


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


def _to_chatml(example: dict, tokenizer) -> dict:
    title  = example.get("title", "")
    body   = example.get("body", "")[:_MAX_BODY_CHARS]
    target = json.dumps(
        {"frame": int(example["frame"]), "logic": int(example["logic"])},
        ensure_ascii=False,
    )
    messages = [
        {"role": "system",    "content": _SYSTEM},
        {"role": "user",      "content": f"기사: {title}\n{body}"},
        {"role": "assistant", "content": target},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )}


def main() -> None:
    args = build_args()

    # ── 학습 데이터 존재 확인 ────────────────────────────────
    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"학습 데이터 없음: {args.data}\n"
            "data/call1_train.jsonl 경로에 골든 데이터셋을 준비하세요.\n"
            "포맷: {\"title\": \"...\", \"body\": \"...\", \"frame\": 4, \"logic\": 3}"
        )

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    # ── 모델 + LoRA 설정 ─────────────────────────────────────
    print(f"[train_call1] 베이스 모델 로딩: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── 데이터셋 로딩 + ChatML 변환 ──────────────────────────
    print(f"[train_call1] 학습 데이터 로딩: {args.data}")
    train_ds = load_dataset("json", data_files=args.data, split="train").map(
        lambda ex: _to_chatml(ex, tokenizer)
    )

    eval_ds = None
    if os.path.exists(args.eval):
        print(f"[train_call1] 검증 데이터 로딩: {args.eval}")
        eval_ds = load_dataset("json", data_files=args.eval, split="train").map(
            lambda ex: _to_chatml(ex, tokenizer)
        )
    else:
        print(f"[train_call1] 검증 데이터 없음 ({args.eval}), eval 생략")

    print(f"[train_call1] 학습 샘플: {len(train_ds)}개"+ (f" / 검증 샘플: {len(eval_ds)}개" if eval_ds else ""))

    # ── SFTTrainer ────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq,
        args=SFTConfig(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,  # effective batch = 32
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="epoch" if eval_ds else "no",
            save_strategy="epoch",
            load_best_model_at_end=bool(eval_ds),
            metric_for_best_model="eval_loss" if eval_ds else None,
            report_to="none",
        ),
    )

    print(
        f"\n[train_call1] 학습 시작\n"
        f"  lora_r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.dropout}\n"
        f"  lr={args.lr}, batch={args.batch}, grad_accum={args.grad_accum} "
        f"(effective={args.batch * args.grad_accum}), epochs={args.epochs}\n"
        f"  저장 경로: {args.output}\n"
    )
    trainer.train()

    # ── 어댑터 저장 ──────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n[train_call1] 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
