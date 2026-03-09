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

# .env 로드
load_dotenv()

# 경로 설정 (log 데이터 확인용)
SAVE_FOLDER = r"C:\Users\Administrator\Desktop\2026-1\2026-1_CreativeProject\data_log"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CLIENT_ID = os.getenv("client_id")
CLIENT_SECRET = os.getenv("client_secret")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 실험 메모 - 버전관리용
COMMIT_MESSAGE = "1차 버전"

def get_news_id(url):
    #URL에서 oid와 aid를 추출하여 네이버 댓글용 objectId를 생성
    try:
        path_parts = urlparse(url).path.split('/')
        params = [p for p in path_parts if p][-2:] 
        return f"news{','.join(params)}"
    except:
        return ""

def get_naver_comments_http(news_url):
    #자바스크립트 JSONP 방식을 이식한 초고속 댓글 수집 함수
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
            return [c['contents'].replace("\n", " ").strip() for c in comment_list if 'contents' in c]
        return []
    except:
        return []

def crawl_task(item):
    #개별 기사를 수집하는 단위 작업 (Thread 기반 병렬 실행)
    url = item.get("originallink") or item.get("link")

    title_clean = item.get("title").replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&")
    
    try:
        # 발행일 파싱
        raw_pub_date = item.get("pubDate")
        try:
            clean_date_obj = datetime.strptime(raw_pub_date, "%a, %d %b %Y %H:%M:%S +0900")
            published_date = clean_date_obj.strftime("%Y-%m-%d")
        except:
            published_date = datetime.now().strftime("%Y-%m-%d")

        # HTML 다운로드 및 메타데이터/본문 추출
        downloaded = trafilatura.fetch_url(url)
        if not downloaded: return None
        
        # 언론사 정보 추출
        metadata = trafilatura.extract_metadata(downloaded)
        media_name = metadata.sitename if metadata and metadata.sitename else urlparse(url).netloc
        
        # 본문 추출
        body = trafilatura.extract(downloaded, include_comments=False)
        if not body: return None

        # 댓글 수집 (HTTP 역추적 방식)
        comments = get_naver_comments_http(url)

        return {
            "title": title_clean,
            "url": url,
            "body": body.strip(),
            "media": media_name,
            "published": published_date,
            "comments": comments,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cm": COMMIT_MESSAGE
        }
    except Exception as e:
        print(f" 개별 태스크 에러 ({url}): {e}")
        return None

async def main_crawler(query):
    start_time = time.time()
    print(f"\n '{query}' 수집 시작")

    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
    
    display_num = 100
    start_num = 201
    api_url = f"https://openapi.naver.com/v1/search/news.json?query={urllib.parse.quote(query)}&display={display_num}&start={start_num}&sort=sim"
    
    res = requests.get(api_url, headers=headers)
    items = res.json().get("items", [])
    if not items: 
        print("검색 결과가 없습니다.")
        return

    # 병렬 처리
    tasks = [asyncio.to_thread(crawl_task, item) for item in items]
    results = await asyncio.gather(*tasks)

    stats = {"saved": 0, "comments": 0, "skipped": 0}

    for res_data in results:
        if res_data:
            try:
                # URL 중복 체크
                existing = supabase.table("news").select("id").eq("url", res_data["url"]).execute()
                if existing.data:
                    stats["skipped"] += 1
                    continue

                # Supabase 저장
                supabase.table("news").insert([res_data]).execute()
                stats["saved"] += 1
                stats["comments"] += len(res_data['comments'])
                print(f" ✅ [{stats['saved']}] {res_data['media']} | {res_data['title'][:20]}... (댓글: {len(res_data['comments'])}개)")
                
            except Exception as e:
                print(f" ❌ DB 에러: {e}")
        else:
            stats["skipped"] += 1

    # -------- 로그 생성 및 저장 -------- 
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = (
        f"{'='*50}\n '{query}' 수집 최종 요약 리포트\n{'-'*50}\n"
        f"총 소요 시간: {minutes}분 {seconds}초\n"
        f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"커밋 메시지: {COMMIT_MESSAGE}\n"
        f"신규 저장: {stats['saved']}개\n"
        f"스킵/실패: {stats['skipped']}개\n"
        f"총 댓글 수: {stats['comments']}개\n{'='*50}\n"
    )

    print("\n" + report)

    if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)
    file_path = os.path.join(SAVE_FOLDER, f"최종리포트_{query}_{now_str}.txt")
    with open(file_path, "w", encoding="utf-8") as f: f.write(report)
    print(f"리포트 저장 완료: {file_path}")

if __name__ == "__main__":
    asyncio.run(main_crawler("이재명"))