import os
import asyncio
import urllib.parse
from urllib.parse import urlparse
import requests
import trafilatura
import time
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
#ai 테스트 데이터셋 만들기 위한 코드 - ai_test table 저장
#언론사명 제외한 뉴스 원문과 댓글 수집만 정상 작동.
# .env 로드
load_dotenv()

# 경로 및 환경 변수 설정
SAVE_FOLDER = r"C:\Users\Administrator\Desktop\2026-1\2026-1_CreativeProject\data_log"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CLIENT_ID = os.getenv("client_id")
CLIENT_SECRET = os.getenv("client_secret")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_news_id(url):
    """URL에서 네이버 댓글용 objectId 추출"""
    try:
        path_parts = urlparse(url).path.split('/')
        params = [p for p in path_parts if p][-2:] 
        return f"news{','.join(params)}"
    except:
        return ""

def get_naver_comments_http(news_url):
    """네이버 뉴스 댓글 수집"""
    try:
        object_id = get_news_id(news_url)
        if not object_id: return []

        api_url = "https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json"
        params = {
            "ticket": "news", "pool": "cbox5", "lang": "ko", "country": "KR",
            "objectId": object_id, "pageSize": 10, "indexSize": 10,
            "pageType": "more", "page": 1, "sort": "favorite", "callback": "_callback"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": news_url
        }

        response = requests.get(api_url, params=params, headers=headers, timeout=5)
        match = re.search(r'_callback\((.*)\);', response.text)
        if not match: return []
        
        data = json.loads(match.group(1))
        if data.get("success"):
            comment_list = data.get("result", {}).get("commentList", [])
            # 댓글 리스트를 JSONB 형태로 저장하기 위해 리스트 객체 반환
            return [c['contents'].replace("\n", " ").strip() for c in comment_list if 'contents' in c]
        return []
    except:
        return []

def crawl_task(item):
    """개별 기사를 수집하여 ai_test 테이블 형식으로 변환"""
    url = item.get("link")
    if "n.news.naver.com" not in url and "news.naver.com" not in url:
        return None

    title_clean = item.get("title").replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&")
    
    try:
        # 1. 발행일 파싱 (스키마의 'date' 타입에 맞게 YYYY-MM-DD 형식)
        raw_pub_date = item.get("pubDate")
        try:
            clean_date_obj = datetime.strptime(raw_pub_date, "%a, %d %b %Y %H:%M:%S +0900")
            published_date = clean_date_obj.strftime("%Y-%m-%d")
        except:
            published_date = datetime.now().strftime("%Y-%m-%d")

        # 2. 본문 수집
        downloaded = trafilatura.fetch_url(url)
        body = trafilatura.extract(downloaded, include_comments=False)
        if not body: return None

        # 3. 댓글 수집
        comments = get_naver_comments_http(url)

        # ai_test 테이블 스키마에 정확히 매칭되는 데이터 구조
        return {
            "title": title_clean,
            "body": body.strip(),
            "url": url,
            "published": published_date,
            "comments": comments  # jsonb 컬럼으로 들어감
        }
    except:
        return None

async def main_crawler(query):
    start_time = time.time()
    print(f"\n'{query}' 수집 및 ai_test 테이블 저장 시작")

    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
    
    # 검색 설정
    display_num = 100
    start_num = 901
    api_url = f"https://openapi.naver.com/v1/search/news.json?query={urllib.parse.quote(query)}&display={display_num}&start={start_num}&sort=sim"
    
    res = requests.get(api_url, headers=headers)
    items = res.json().get("items", [])
    if not items: 
        print("검색 결과가 없습니다.")
        return

    # 병렬 처리
    tasks = [asyncio.to_thread(crawl_task, item) for item in items]
    results = await asyncio.gather(*tasks)

    stats = {"saved": 0, "skipped": 0}

    for res_data in results:
        if res_data:
            try:
                # 1. URL 중복 체크 (ai_test 테이블 기준)
                existing = supabase.table("ai_test").select("id").eq("url", res_data["url"]).execute()
                if existing.data:
                    stats["skipped"] += 1
                    continue

                # 2. Supabase ai_test 테이블에 저장
                supabase.table("ai_test").insert([res_data]).execute()
                stats["saved"] += 1
                print(f" ✅ [{stats['saved']}] {res_data['title'][:20]}...")
                
            except Exception as e:
                print(f" ❌ DB 에러: {e}")
        else:
            stats["skipped"] += 1

    # 최종 리포트
    elapsed = time.time() - start_time
    print(f"\n 완료! {stats['saved']}개 데이터가 'ai_test'에 저장되었습니다. (소요시간: {int(elapsed)}초)")

if __name__ == "__main__":
    asyncio.run(main_crawler("휘발유"))