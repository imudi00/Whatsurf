import os
import sys
import json
import time
import re
from pathlib import Path
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PACKAGE_PARENT not in sys.path:
    sys.path.append(PACKAGE_PARENT)

from google import genai
from google.genai import types
from AI.source.data_loader import load_ai_test_body_map_by_ids

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY가 .env 또는 현재 터미널 환경변수에 없습니다.")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash-lite"

CURRENT_DIR = Path(__file__).resolve().parent

INPUT_PATH = CURRENT_DIR / "artifacts" / "cluster_samsung_1000" / "cluster_summary.json"
OUTPUT_PATH = CURRENT_DIR / "llm_results" / "llm_cluster_summary_gemini_samsung.json"

MAX_CLUSTERS = 10
MAX_TITLES_PER_CLUSTER = 5
SLEEP_SEC = 1.0


def build_prompt(cluster_input: dict) -> str:
    representative_title = cluster_input.get("representative_title", "")
    titles = cluster_input.get("titles", [])
    key_sentences = cluster_input.get("key_sentences", [])

    key_sentences_text = ""
    if key_sentences:
        key_sentences_text = "\n핵심 문장:\n" + "\n".join([f"- {s}" for s in key_sentences])

    return f"""
너는 뉴스 타임라인 서비스용 클러스터 요약기다.

입력은 같은 사건으로 묶인 기사 클러스터의 압축 정보다.
이 정보를 바탕으로 아래 두 값을 생성하라.

반드시 JSON만 출력하라.
설명문, 마크다운, 코드블록 없이 JSON 객체만 출력하라.

출력 형식:
{{
  "headline": "클러스터 전체를 대표하는 한 줄 제목",
  "summary": "2~3문장 요약"
}}

작성 규칙:
- headline은 기사 묶음 전체의 공통 사건을 대표해야 한다.
- headline은 너무 자극적이거나 과장되면 안 된다.
- summary는 핵심 내용만 간결하게 정리한다.
- 추측, 해석, 과장 표현 금지
- 중복 표현 금지
- 한국어로 작성
- 입력에 핵심 문장이 있으면 summary에 우선 반영하고, 없으면 제목들을 기반으로 요약하라.

입력 데이터:
대표 제목:
{representative_title}

기사 제목 목록:
{json.dumps(titles, ensure_ascii=False, indent=2)}{key_sentences_text}

메타데이터:
{json.dumps(cluster_input.get("meta", {}), ensure_ascii=False, indent=2)}
""".strip()


def parse_model_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "headline": "",
            "summary": "",
            "parse_error": True,
            "raw_text": text
        }

def split_sentences(text: str) -> list[str]:
    text = str(text).strip()
    if not text:
        return []

    # 아주 단순한 문장 분리
    sentences = re.split(r'(?<=[.!?다요])\s+|\n+', text)
    cleaned = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) < 15:
            continue
        cleaned.append(s)

    return cleaned


def is_too_similar(a: str, b: str) -> bool:
    a_words = set(a.split())
    b_words = set(b.split())

    if not a_words or not b_words:
        return False

    overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
    return overlap >= 0.7

def extract_key_sentences_from_articles(
    article_rows: list[dict],
    representative_title: str,
    cluster_titles: list[str],
    max_sentences: int = 3
) -> list[str]:
    """
    제목 관련도가 높은 기사만 우선 사용하고,
    문장도 대표 제목/클러스터 제목과 겹치는 키워드가 많은 문장을 우선 선택
    """
    cluster_context = " ".join([representative_title] + cluster_titles)

    scored_candidates = []

    for article in article_rows:
        article_title = str(article.get("title", "")).strip()
        body = str(article.get("body", "")).strip()

        if not body:
            continue

        # 1) 기사 제목이 대표 제목/클러스터 제목과 너무 안 맞으면 제외
        article_relevance = token_overlap_score(article_title, cluster_context)
        if article_relevance < 0.2:
            continue

        sentences = split_sentences(body)

        for idx, sent in enumerate(sentences[:6]):
            sent_norm = normalize_text(sent)

            # 너무 짧거나 이상한 문장 제외
            if len(sent_norm) < 15:
                continue

            # 기자명/괄호 시작 문장/숫자표만 있는 문장 등 약하게 패널티
            penalty = 0.0
            if sent.startswith("(") or "기자" in sent:
                penalty += 0.2
            if len(sent) > 120:
                penalty += 0.1

            # 2) 문장 관련도 점수
            sentence_relevance = token_overlap_score(sent, cluster_context)

            # 3) 앞문장 약간 가산점
            position_bonus = max(0, 0.15 - idx * 0.03)

            score = sentence_relevance + article_relevance + position_bonus - penalty

            scored_candidates.append((score, sent, article_title))

    # 점수 높은 순 정렬
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    for score, sent, article_title in scored_candidates:
        if any(is_too_similar(sent, prev) for prev in selected):
            continue
        selected.append(sent)
        if len(selected) >= max_sentences:
            break

    return selected

def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r'[^0-9a-zA-Z가-힣\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def token_overlap_score(a: str, b: str) -> float:
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / max(1, len(a_tokens))

def convert_cluster_to_llm_input(data: dict, cluster: dict) -> dict:
    representative_title = cluster.get("representative_article", {}).get("title", "")

    titles = []
    article_ids = []

    raw_articles = cluster.get("articles", [])

    for article in raw_articles:
        title = str(article.get("title", "")).strip()
        article_id = article.get("id")

        if not title:
            continue

    # 대표 제목이 있으면 관련도 낮은 제목은 제외
        if representative_title:
            if token_overlap_score(title, representative_title) < 0.15:
                continue

        titles.append(title)

        if article_id is not None:
            article_ids.append(str(article_id))

        if len(titles) >= MAX_TITLES_PER_CLUSTER:
            break

        if not representative_title and titles:
            representative_title = titles[0]

    # Supabase에서 body 다시 조회
    body_map = load_ai_test_body_map_by_ids(article_ids)
    article_rows = []

    for article_id in article_ids:
        if article_id in body_map:
            article_rows.append(body_map[article_id])

    key_sentences = extract_key_sentences_from_articles(
        article_rows=article_rows,
        representative_title=representative_title,
        cluster_titles=titles,
        max_sentences=3
    )
    
    llm_input = {
        "representative_title": representative_title,
        "titles": titles,
        "key_sentences": key_sentences,
        "meta": {
            "query": data.get("query", ""),
            "cluster_id": cluster.get("cluster_id"),
            "article_count": cluster.get("article_count"),
        }
    }

    return llm_input

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters = data.get("clusters", [])[:MAX_CLUSTERS]

    results = []
    total_start = time.perf_counter()

    print(f"[INFO] source file = {INPUT_PATH}")
    print(f"[INFO] loaded clusters = {len(clusters)}")

    for idx, cluster in enumerate(clusters, start=1):
        cluster_id = cluster.get("cluster_id")
        llm_input = convert_cluster_to_llm_input(data, cluster)

        result_item = {
            "cluster_id": cluster_id,
            "input": llm_input,
            "model": MODEL_NAME,
            "success": False,
            "generated_headline": "",
            "generated_summary": "",
            "original_title": cluster.get("title", ""),
            "original_summary": cluster.get("summary", ""),
            "raw_output": "",
            "error": "",
            "elapsed_sec": 0.0,
        }

        if not llm_input["titles"]:
            result_item["error"] = "클러스터에서 사용할 제목을 찾지 못함"
            results.append(result_item)
            continue

        start = time.perf_counter()

        try:
            prompt = build_prompt(llm_input)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    max_output_tokens=300,
                )
            )

            content = response.text or ""
            parsed = parse_model_json(content)

            result_item["raw_output"] = content
            result_item["generated_headline"] = parsed.get("headline", "")
            result_item["generated_summary"] = parsed.get("summary", "")
            result_item["success"] = not parsed.get("parse_error", False)

            if parsed.get("parse_error", False):
                result_item["error"] = "Gemini 응답 JSON 파싱 실패"

        except Exception as e:
            result_item["error"] = str(e)

        end = time.perf_counter()
        result_item["elapsed_sec"] = round(end - start, 4)

        print(
            f"[{idx}/{len(clusters)}] "
            f"cluster_id={cluster_id} "
            f"success={result_item['success']} "
            f"time={result_item['elapsed_sec']}s"
        )

        results.append(result_item)
        time.sleep(SLEEP_SEC)

    total_end = time.perf_counter()

    output_data = {
        "query": data.get("query", ""),
        "model": MODEL_NAME,
        "cluster_count": len(results),
        "total_elapsed_sec": round(total_end - total_start, 4),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("saved:", OUTPUT_PATH)
    print("total elapsed:", round(total_end - total_start, 4), "sec")


if __name__ == "__main__":
    main()