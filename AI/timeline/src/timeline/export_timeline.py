import json
import os
from typing import Any

from google import genai
from google.genai import types

from .summarize_extractive import summarize_timepoint


_GEMINI_CLIENT = None


def _get_gemini_client():
    global _GEMINI_CLIENT

    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def choose_representative_article(day_articles):
    """
    개발 테스트용 휴리스틱:
    가장 긴 본문을 대표 기사로 선택
    """
    idx = day_articles["body"].astype(str).str.len().idxmax()
    row = day_articles.loc[idx]
    return {
        "title": str(row.get("title", "")),
        "url": str(row.get("url", "")),
        "published": str(row.get("published", "")),
        "id": str(row.get("id", "")) if "id" in day_articles.columns else "",
    }


def _build_article_key(row):
    """
    기사 중복/차집합 비교용 키 생성
    id 컬럼이 있으면 id 우선 사용
    없으면 title + url 조합 사용
    """
    if "id" in row and row["id"] is not None:
        return f"id::{row['id']}"
    title = str(row.get("title", "")).strip()
    url = str(row.get("url", "")).strip()
    return f"title_url::{title}::{url}"


def _filter_new_articles(day_df, prev_article_keys):
    """
    이전 시점에 없던 새 기사만 추출
    """
    if len(day_df) == 0:
        return day_df.copy()

    keys = day_df.apply(_build_article_key, axis=1)
    mask = ~keys.isin(prev_article_keys)
    return day_df[mask].copy()


def _make_update_summary(new_articles, query, max_eojel=17, top_sentences=2, alpha=0.5):
    """
    이전 시점 대비 새로 들어온 기사들만 요약
    """
    if len(new_articles) == 0:
        return "이전 시점 대비 핵심 업데이트 없음"

    return summarize_timepoint(
        new_articles,
        query=query,
        max_eojel=max_eojel,
        top_sentences=top_sentences,
        alpha=alpha,
    )


def _safe_json_loads(text: str) -> dict:
    """
    LLM 응답이 코드블록이나 여분 텍스트를 포함해도 최대한 JSON만 추출
    """
    text = str(text).strip()

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

    return json.loads(text)


def _build_timeline_llm_input(
    *,
    query: str,
    date: str,
    importance: float,
    day_df,
    new_articles,
    summary: str,
    update_summary: str,
    representative: dict[str, Any],
) -> dict:
    """
    LLM에 넣을 타임라인 시점 압축 정보 생성
    """
    top_article_titles = [
        str(row.get("title", "")).strip()
        for _, row in day_df.head(5).iterrows()
        if str(row.get("title", "")).strip()
    ]

    new_article_titles = [
        str(row.get("title", "")).strip()
        for _, row in new_articles.head(5).iterrows()
        if str(row.get("title", "")).strip()
    ]

    key_sentences = []
    if summary and summary.strip():
        key_sentences.append(summary.strip())
    if update_summary and update_summary.strip() and update_summary.strip() != "이전 시점 대비 핵심 업데이트 없음":
        key_sentences.append(update_summary.strip())

    return {
        "query": query,
        "date": date,
        "importance": round(float(importance), 4),
        "article_count": int(len(day_df)),
        "new_article_count": int(len(new_articles)),
        "representative_title": representative.get("title", ""),
        "top_article_titles": top_article_titles,
        "new_article_titles": new_article_titles,
        "key_sentences": key_sentences,
    }


def generate_timeline_llm_summary(point: dict) -> dict:
    """
    특정 타임라인 시점에 대해 headline + summary 생성
    """
    client = _get_gemini_client()
    if client is None:
        return {
            "phase_headline": "",
            "phase_summary": "",
            "llm_error": "GEMINI_API_KEY not found",
        }

    prompt = f"""
너는 뉴스 타임라인 요약기다.
입력은 특정 시점에 해당하는 기사 묶음의 압축 정보다.

반드시 JSON만 출력하라.
설명문, 마크다운, 코드블록 없이 JSON 객체만 출력하라.

형식:
{{
  "phase_headline": "해당 시점을 대표하는 한 줄 제목",
  "phase_summary": "2문장 요약"
}}

규칙:
- phase_headline은 해당 날짜/시점의 핵심 변화가 드러나야 한다.
- phase_summary는 해당 시점에서 새롭게 드러난 내용과 핵심 쟁점을 중심으로 작성한다.
- 입력에 update 정보가 있으면 우선 반영한다.
- 과장 금지
- 추측 금지
- 한국어로 작성
- 중복 표현 금지

입력:
{json.dumps(point, ensure_ascii=False, indent=2)}
""".strip()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                max_output_tokens=250,
            ),
        )

        parsed = _safe_json_loads(response.text)

        return {
            "phase_headline": parsed.get("phase_headline", "").strip(),
            "phase_summary": parsed.get("phase_summary", "").strip(),
            "llm_error": "",
        }

    except Exception as e:
        return {
            "phase_headline": "",
            "phase_summary": "",
            "llm_error": str(e),
        }


def build_timeline(
    df,
    query,
    ranked_timepoints,
    top_timepoints=8,
    max_eojel=17,
    top_sentences=3,
    alpha=0.5,
    use_llm_summary=True,
):
    top = ranked_timepoints.head(top_timepoints)
    dates = top["date"].tolist()

    timeline = []
    prev_article_keys = set()

    for d in sorted(dates):
        day = df[df["date"].astype(str) == d].copy()
        if len(day) == 0:
            continue

        summary = summarize_timepoint(
            day,
            query=query,
            max_eojel=max_eojel,
            top_sentences=top_sentences,
            alpha=alpha,
        )

        rep = choose_representative_article(day)

        # 이전 시점 대비 새 기사만 추출
        new_articles = _filter_new_articles(day, prev_article_keys)

        # 업데이트 요약 생성
        update_summary = _make_update_summary(
            new_articles,
            query=query,
            max_eojel=max_eojel,
            top_sentences=max(1, top_sentences - 1),
            alpha=alpha,
        )

        importance = float(top[top["date"] == d]["importance"].iloc[0])

        timeline_item = {
            "date": d,
            "importance": importance,
            "summary": summary,
            "update_summary": update_summary,
            "representative": rep,
            "articles": [
                {
                    "id": str(row["id"]) if "id" in day.columns else "",
                    "title": str(row.get("title", "")),
                    "url": str(row.get("url", "")),
                    "published": str(row.get("published", "")),
                }
                for _, row in day.iterrows()
            ],
            "new_articles": [
                {
                    "id": str(row["id"]) if "id" in new_articles.columns else "",
                    "title": str(row.get("title", "")),
                    "url": str(row.get("url", "")),
                    "published": str(row.get("published", "")),
                }
                for _, row in new_articles.iterrows()
            ],
            "count": int(len(day)),
            "new_count": int(len(new_articles)),
        }

        if use_llm_summary:
            llm_input = _build_timeline_llm_input(
                query=query,
                date=d,
                importance=importance,
                day_df=day,
                new_articles=new_articles,
                summary=summary,
                update_summary=update_summary,
                representative=rep,
            )

            llm_result = generate_timeline_llm_summary(llm_input)

            timeline_item["llm_input"] = llm_input
            timeline_item["llm_error"] = llm_result.get("llm_error", "")

            phase_headline = llm_result.get("phase_headline", "").strip()
            phase_summary = llm_result.get("phase_summary", "").strip()

            # fallback: LLM 실패 시 비LLM 결과로 대체
            timeline_item["phase_headline"] = (
                phase_headline if phase_headline else rep.get("title", "")[:40]
            )
            timeline_item["phase_summary"] = (
                phase_summary
                if phase_summary
                else (
                    update_summary
                    if update_summary and update_summary != "이전 시점 대비 핵심 업데이트 없음"
                    else summary
                )
            )
        else:
            timeline_item["llm_input"] = {}
            timeline_item["llm_error"] = "LLM disabled"
            timeline_item["phase_headline"] = rep.get("title", "")[:40]
            timeline_item["phase_summary"] = (
                update_summary
                if update_summary and update_summary != "이전 시점 대비 핵심 업데이트 없음"
                else summary
            )

        timeline.append(timeline_item)

        # 현재 날짜 기사들을 다음 시점 비교 기준으로 누적
        current_keys = set(day.apply(_build_article_key, axis=1).tolist())
        prev_article_keys.update(current_keys)

    return timeline