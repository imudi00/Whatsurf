import os
import asyncio
import urllib.parse
from urllib.parse import urlparse
import requests
import trafilatura
import time
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# .env 로드
load_dotenv()

# ✅ 경로 설정
SAVE_FOLDER = r"C:\Users\Administrator\Desktop\2026-1\2026-1_CreativeProject\data_log"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CLIENT_ID = os.getenv("client_id")
CLIENT_SECRET = os.getenv("client_secret")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 💡 커밋 메시지 (실험 버전 기록용)
COMMIT_MESSAGE = "공백 포함 단어"

def parse_naver_date(date_str):
    """네이버 API의 pubDate 문자열을 YYYY-MM-DD 형식으로 변환합니다."""
    try:
        # 형식 예: "Tue, 24 Feb 2026 14:00:00 +0900"
        dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S +0900")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # 파싱 실패 시 오늘 날짜 반환
        return datetime.now().strftime("%Y-%m-%d")

def crawl_with_trafilatura(item):
    """모든 뉴스 링크에서 본문을 추출하고 실제 발행일을 정리합니다."""
    # originallink가 있으면 쓰고, 없으면 일반 link(네이버뉴스) 사용
    url = item.get("originallink") or item.get("link")
    title_clean = item.get("title").replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&")
    
    try:
        print(f"🔍 [시도] {title_clean[:30]}...")
        
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "fail_download"
        
        body = trafilatura.extract(downloaded, include_comments=False, output_format='txt')
        if not body:
            return "fail_extract"

        # 1. 실제 기사 발행일 파싱
        published_date = parse_naver_date(item.get("pubDate"))
        
        # 2. 데이터 생성 시각 (timestamp용)
        now = datetime.now()
        created_at = now.strftime("%Y-%m-%d %H:%M:%S")

        # 언론사 명칭 추출 (없으면 도메인 사용)
        metadata = trafilatura.extract_metadata(downloaded)
        media = metadata.sitename if metadata and metadata.sitename else urlparse(url).netloc

        return {
            "title": title_clean,
            "url": url,
            "body": body.strip(),
            "media": media,
            "published": published_date, # ✅ 진짜 기사 날짜
            "comments": None,            # ✅ 댓글은 NULL 처리
            "created": created_at,       # ✅ 수집 시각
            "cm": COMMIT_MESSAGE         # ✅ 실험 메모
        }
    except Exception as e:
        return "fail_exception"

def save_to_supabase(data):
    """Supabase에 데이터를 저장합니다 (중복 체크 포함)."""
    try:
        # URL 중복 체크
        existing = supabase.table("news").select("id").eq("url", data["url"]).execute()
        if existing.data:
            return "skipped"

        # DB 저장
        response = supabase.table("news").insert([data]).execute()
        return "saved" if response.data else "db_error"
    except Exception as e:
        print(f"   ❌ [DB 에러] {e}")
        return "db_error"

async def main_crawler(query):
    start_time = time.time()
    print(f"\n🚀 '{query}' 대량 수집 시작 (커밋: {COMMIT_MESSAGE})")
    
    encoded_query = urllib.parse.quote(query)
    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}

    stats = {
        "total_links": 0, "success_extract": 0, "saved_new": 0, "skipped_dupe": 0
    }

    display = 100 # 100개씩
    for start in range(201, 301, display): #1에서 101까지 (페이지)
        api_url = f"https://openapi.naver.com/v1/search/news.json?query={encoded_query}&display={display}&start={start}&sort=sim"
        
        response = requests.get(api_url, headers=headers)
        if response.status_code != 200: break

        items = response.json().get("items", [])
        if not items: break
        stats["total_links"] += len(items)

        # ⚡ 병렬 처리 (Thread 기반 비동기 실행)
        tasks = [asyncio.to_thread(crawl_with_trafilatura, item) for item in items]
        results = await asyncio.gather(*tasks)

        for res in results:
            if isinstance(res, dict):
                stats["success_extract"] += 1
                status = save_to_supabase(res)
                if status == "saved": stats["saved_new"] += 1
                elif status == "skipped": stats["skipped_dupe"] += 1

    # --- [리포트 생성 및 저장] ---
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = (
        f"{'='*50}\n📊 '{query}' 수집 최종 요약 리포트\n{'-'*50}\n"
        f"⏱️ 총 소요 시간: {minutes}분 {seconds}초\n"
        f"📅 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"📝 커밋 메시지: {COMMIT_MESSAGE}\n"
        f"✅ 신규 저장: {stats['saved_new']}개 (중복 스킵: {stats['skipped_dupe']}개)\n"
        f"🔗 분석된 총 링크: {stats['total_links']}개\n{'='*50}\n"
    )

    print("\n" + report)

    try:
        if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)
        file_path = os.path.join(SAVE_FOLDER, f"수집결과_{query}_{now_str}.txt")
        with open(file_path, "w", encoding="utf-8") as f: f.write(report)
    except: pass

if __name__ == "__main__":
    asyncio.run(main_crawler('"이재명 대통령"'))