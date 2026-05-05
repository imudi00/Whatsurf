# labeling/train/data_prep.py
"""
call1 학습 데이터 준비 유틸.

JSONL 포맷 (한 줄 = 한 기사):
  {"id": "...", "title": "...", "body": "...", "frame": 4, "logic": 3}

사용:
  python data_prep.py --input call1_raw.jsonl --output call1_train.jsonl --check
"""
import argparse
import json
from pathlib import Path

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

_MAX_BODY_CHARS = 1800


def validate_call1_jsonl(path: str) -> tuple[int, list[str]]:
    """
    JSONL 검증. 필수 필드: title, body, frame(1~7), logic(1~6).

    Returns:
        (valid_count, errors)  — errors 가 빈 리스트면 모두 정상
    """
    errors: list[str] = []
    valid = 0

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"L{lineno}: JSON 파싱 실패 — {e}")
                continue

            missing = [k for k in ("title", "body", "frame", "logic") if k not in ex]
            if missing:
                errors.append(f"L{lineno}: 필수 필드 누락 {missing}")
                continue

            if not (1 <= int(ex["frame"]) <= 7):
                errors.append(f"L{lineno}: frame 범위 오류 ({ex['frame']}), 1~7 필요")
                continue

            if not (1 <= int(ex["logic"]) <= 6):
                errors.append(f"L{lineno}: logic 범위 오류 ({ex['logic']}), 1~6 필요")
                continue

            valid += 1

    return valid, errors


def to_chatml_text(example: dict, tokenizer) -> str:
    """
    단일 example dict → ChatML 문자열 (tokenizer.apply_chat_template 사용).

    example 필드: title, body, frame(int), logic(int)
    """
    title = example["title"]
    body  = example["body"][:_MAX_BODY_CHARS]
    target = json.dumps(
        {"frame": int(example["frame"]), "logic": int(example["logic"])},
        ensure_ascii=False,
    )
    messages = [
        {"role": "system",    "content": _SYSTEM},
        {"role": "user",      "content": f"기사: {title}\n{body}"},
        {"role": "assistant", "content": target},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def convert_jsonl(src: str, dst: str, tokenizer) -> int:
    """
    src JSONL → dst JSONL ({"text": "<chatml string>"} 형식).

    Returns: 변환된 행 수
    """
    count = 0
    with open(src, encoding="utf-8") as fin, \
         open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            text = to_chatml_text(ex, tokenizer)
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="call1 학습 데이터 준비")
    ap.add_argument("--input",  required=True, help="원본 JSONL 경로")
    ap.add_argument("--output", help="출력 ChatML JSONL 경로 (--check 없을 때 필수)")
    ap.add_argument("--check",  action="store_true", help="검증만 수행 (변환 안 함)")
    args = ap.parse_args()

    valid, errors = validate_call1_jsonl(args.input)
    print(f"검증 결과: {valid}개 정상")
    if errors:
        print(f"오류 {len(errors)}건:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... 외 {len(errors) - 20}건")

    if not args.check and args.output:
        from unsloth import FastLanguageModel
        _, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        n = convert_jsonl(args.input, args.output, tokenizer)
        print(f"변환 완료: {n}개 → {args.output}")
