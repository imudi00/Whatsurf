'''
요청 간 5초 간격 (분당 12회로 제한)
실패 시 RETRY_LIMIT=3 재시도 + 10초 대기
BATCH_SIZE=50마다 중간 저장 → 중간에 죽어도 --start 옵션으로 이어하기 가능

# 처음 실행
python auto_labeler.py --input crawled.csv --output labeled_output.csv

# 500번에서 끊겼을 때 이어하기
python auto_labeler.py --input crawled.csv --output labeled_output.csv --start 500

auto_labeler.py
크롤링 CSV → Gemini 자동 라벨링 → labeled_output.csv 저장

'''

import os
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai

# 로컬 모듈
from preprocessor import preprocess

# ── 환경 설정 ──────────────────────────────────────────────
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

FRAME_LABELS = ["인과", "대립", "개인", "경제적가치", "도덕", "안보", "권리"]
LOGIC_LABELS = ["정책적 비난", "전문가 견해", "피해자 서사", "파급효과", "해결책 제시", "사실/정보 전달"]

# rate limit 대응: 분당 최대 요청 수 (free tier 기준)
REQUESTS_PER_MINUTE = 12       # 여유있게 15에서 12로 설정
BATCH_SIZE = 50                # 50건마다 중간 저장 (체크포인트)
RETRY_LIMIT = 3                # 실패 시 재시도 횟수
RETRY_DELAY = 10               # 재시도 전 대기 (초)


# ── 프롬프트 ───────────────────────────────────────────────
FEW_SHOT_FRAME = """
예시1) 헤드라인: 의대 정원 확대, 의사협회 강력 반발 / 판단어: 반발, 규탄 → frame: 대립
예시2) 헤드라인: 전기요금 8% 인상, 중소기업 원가 부담 / 수치: 8%, 3조원 → frame: 경제적가치
예시3) 헤드라인: 성범죄 피해자 2차 가해 근절 촉구 / 판단어: 촉구, 요구 → frame: 권리
"""

FEW_SHOT_LOGIC = """
예시1) 출처: 야당, 시민단체 / 인용: "명백한 실정이다" → logic: 정책적 비난
예시2) 출처: 경제연구원, 교수 / 수치: GDP 0.3% 하락 → logic: 전문가 견해
예시3) 출처: 피해자 가족 / 인용: "삶이 무너졌다" → logic: 피해자 서사
예시4) 출처: 정부 부처 / 판단어: 없음 / 공식 발표 위주 → logic: 사실/정보 전달
"""


def build_prompt(struct: dict) -> str:
    return f"""한국어 뉴스 기사를 분석해 JSON만 출력하세요. 다른 텍스트 없이 JSON만 출력.

## frame 정의 (하나 선택)
인과/대립/개인/경제적가치/도덕/안보/권리
{FEW_SHOT_FRAME}

## logic 정의 (하나 선택)
정책적 비난/전문가 견해/피해자 서사/파급효과/해결책 제시/사실·정보 전달
{FEW_SHOT_LOGIC}

## stance_score
-1.0(강한 비판) ~ 0.0(중립) ~ +1.0(강한 옹호)

## 분석 대상
헤드라인: {struct['headline']}
리드: {struct['lead']}
인용구: {struct['quotes']}
판단어: {struct['judgment_words']}
출처: {struct['sources']}
수치: {struct['numbers']}

출력 형식:
{{"frame":"...","logic":"...","stance_score":0.0,"frame_reason":"한줄","logic_reason":"한줄"}}"""


def label_one(struct: dict) -> dict | None:
    """단건 라벨링. 실패 시 RETRY_LIMIT만큼 재시도."""
    prompt = build_prompt(struct)

    for attempt in range(RETRY_LIMIT):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            # 유효성 검증
            if result.get("frame") not in FRAME_LABELS:
                result["frame"] = "인과"
            if result.get("logic") not in LOGIC_LABELS:
                result["logic"] = "사실/정보 전달"
            result["stance_score"] = max(-1.0, min(1.0, float(result.get("stance_score", 0.0))))
            return result

        except Exception as e:
            print(f"\n  ⚠ 시도 {attempt+1}/{RETRY_LIMIT} 실패: {e}")
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY)

    return None  # 최종 실패


def save_checkpoint(records: list, output_path: str):
    """중간 저장"""
    pd.DataFrame(records).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  💾 체크포인트 저장: {len(records)}건 → {output_path}")


# ── 메인 ───────────────────────────────────────────────────
def main(input_path: str, output_path: str, start_idx: int = 0):
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"총 {len(df)}건 로드. {start_idx}번부터 시작.")

    # 이어하기: 이미 저장된 결과 불러오기
    if start_idx > 0 and os.path.exists(output_path):
        done = pd.read_csv(output_path, encoding="utf-8-sig").to_dict("records")
        print(f"  기존 결과 {len(done)}건 불러옴.")
    else:
        done = []

    interval = 60 / REQUESTS_PER_MINUTE  # 요청 간 최소 간격(초)
    failed_indices = []

    for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx):
        text = str(row.get("body", ""))
        title = str(row.get("title", ""))

        if len(text.strip()) < 50:
            # 본문이 너무 짧으면 스킵
            failed_indices.append(i)
            continue

        # 전처리
        struct = preprocess(text)
        if not struct["headline"]:
            struct["headline"] = title  # 헤드라인 없으면 title로 보완

        # 라벨링
        result = label_one(struct)

        if result:
            done.append({
                "idx":          i,
                "title":        title,
                "date":         row.get("date", ""),
                "press":        row.get("press", ""),
                "body":         text[:300],   # 원문 300자만 저장 (용량 절약)
                "frame":        result["frame"],
                "logic":        result["logic"],
                "stance_score": result["stance_score"],
                "frame_reason": result.get("frame_reason", ""),
                "logic_reason": result.get("logic_reason", ""),
            })
        else:
            failed_indices.append(i)

        # 체크포인트: BATCH_SIZE마다 저장
        if len(done) % BATCH_SIZE == 0:
            save_checkpoint(done, output_path)

        # rate limit 대응: 요청 간 간격 유지
        time.sleep(interval)

    # 최종 저장
    save_checkpoint(done, output_path)

    print(f"\n✅ 완료: 성공 {len(done)}건 / 실패 {len(failed_indices)}건")
    if failed_indices:
        print(f"  실패 인덱스: {failed_indices}")
        pd.DataFrame({"failed_idx": failed_indices}).to_csv(
            output_path.replace(".csv", "_failed.csv"), index=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,  help="크롤링 CSV 경로")
    parser.add_argument("--output", required=True,  help="라벨링 결과 CSV 경로")
    parser.add_argument("--start",  type=int, default=0, help="이어하기 시작 인덱스")
    args = parser.parse_args()

    main(args.input, args.output, args.start)