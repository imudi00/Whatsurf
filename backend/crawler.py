import os
import asyncio
import urllib.parse
from urllib.parse import urlparse, parse_qs
import requests
import trafilatura
import time
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
#네이버 뉴스 url 타겟팅 -> 댓글 제대로 됨.
#언론사까지 id로 추출 완료.. 1차

#.env 로드
load_dotenv()

#경로 설정
SAVE_FOLDER = r"C:\Users\Administrator\Desktop\2026-1\2026-1_CreativeProject\data_log"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CLIENT_ID = os.getenv("client_id")
CLIENT_SECRET = os.getenv("client_secret")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

#실험 메모
COMMIT_MESSAGE = "댓글 대량수집"

def get_naver_comments_http(news_url):
    """자바스크립트 JSONP 방식 댓글 수집 함수"""
    try:
        object_id = get_news_id(news_url)
        if not object_id: return []

        api_url = "https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json"
        params = {
            "ticket": "news", "pool": "cbox5", "lang": "ko", "country": "KR",
            "objectId": object_id, "pageSize": 100, "indexSize": 10,
            "pageType": "more", "page": 1, "sort": "favorite", "callback": "_callback"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": news_url
        }

        response = requests.get(api_url, params=params, headers=headers, timeout=5)
        
        # JSONP 응답에서 JSON 추출 (_callback(...); 제거)
        match = re.search(r'_callback\((.*)\);', response.text)
        if not match: return []
        
        data = json.loads(match.group(1))
        if data.get("success"):
            comment_list = data.get("result", {}).get("commentList", [])
            return [c['contents'].replace("\n", " ").strip() for c in comment_list if 'contents' in c]
        return []
    except:
        return []

#모든 형식의 URL에서 oid 추출
def extract_oid_from_url(url):
    try:
        parsed = urlparse(url)
        #쿼리 파라미터 확인 (?oid=...)
        qs = parse_qs(parsed.query)
        if 'oid' in qs: return int(qs['oid'][0])
        
        #경로 확인 (/001/...)
        parts = [p for p in parsed.path.split('/') if p]
        for i, p in enumerate(parts):
            if p == 'article' and i + 1 < len(parts):
                return int(parts[i+1])
    except:
        return None

def get_news_id(url):
    # 댓글용 objectId 생성
    oid = extract_oid_from_url(url)
    
    # aid는 보통 경로의 맨 마지막 숫자
    try:
        aid = urlparse(url).path.split('/')[-1]
        if oid and aid:
            # oid를 3자리 문자열로 맞추기 (예: 1 -> 001)
            return f"news{str(oid).zfill(3)},{aid}"
    except:
        pass
    return ""

def crawl_task(item):
    #개별 기사를 수집하는 단위 작업
    url = item.get("link")
    if "n.news.naver.com" not in url and "news.naver.com" not in url:
        return None

    #언론사 ID(oid) 추출
    oid = extract_oid_from_url(url)
    if oid is None:
        return None # 언론사 ID를 알 수 없는 기사는 스킵

    title_clean = item.get("title").replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&")
    
    try:
        # 발행일 파싱
        raw_pub_date = item.get("pubDate")
        try:
            clean_date_obj = datetime.strptime(raw_pub_date, "%a, %d %b %Y %H:%M:%S +0900")
            published_date = clean_date_obj.strftime("%Y-%m-%d")
        except:
            published_date = datetime.now().strftime("%Y-%m-%d")

        # 본문 수집 (Trafilatura)
        downloaded = trafilatura.fetch_url(url)
        body = trafilatura.extract(downloaded, include_comments=False)
        if not body: return None

        # 댓글 수집
        comments = get_naver_comments_http(url)

        return {
            "title": title_clean,
            "url": url,
            "body": body.strip(),
            "media": oid,
            "published": published_date,
            "comments": comments,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cm": COMMIT_MESSAGE
        }
    except:
        return None

async def main_crawler(query_text):
    start_time = time.time()
    print(f"\n '{query_text}' 수집 및 DB 저장 시작")

    # 1. queries 테이블에 검색어 저장
    try:
        query_data = supabase.table("queries").insert({
            "query_text": query_text,
            "requested_at": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }).execute()
        query_id = query_data.data[0]['id']
    except Exception as e:
        print(f"쿼리 저장 에러: {e}")
        return

    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
    api_url = f"https://openapi.naver.com/v1/search/news.json?query={urllib.parse.quote(query_text)}&display=100&sort=sim"
    
    res = requests.get(api_url, headers=headers)
    items = res.json().get("items", [])
    
    tasks = [asyncio.to_thread(crawl_task, item) for item in items]
    results = await asyncio.gather(*tasks)

    stats = {"articles": 0, "comments": 0}

    for res_data in results:
        if not res_data: continue
            
        try:
            # 2. articles 테이블 저장
            article_payload = {
                "query_id": query_id,
                "source_id": res_data["media"],
                "title": res_data["title"],
                "body_text": res_data["body"],
                "url": res_data["url"],
                "published_at": res_data["published"],
                "created_at": datetime.now().isoformat(),
                "cluster_label": None # 명시적 NULL
            }

            # URL 중복 체크
            existing = supabase.table("articles").select("id").eq("url", res_data["url"]).execute()
            if existing.data:
                continue

            article_res = supabase.table("articles").insert(article_payload).execute()
            article_id = article_res.data[0]['id']
            stats["articles"] += 1

            # 3. comments 테이블에 댓글 개별 저장
            if res_data["comments"]:
                comment_payloads = []
                # enumerate를 사용하여 순서대로 순위 부여 (1위부터 시작)
                for idx, content in enumerate(res_data["comments"], start=1):
                    comment_payloads.append({
                        "article_id": article_id,
                        "cmt_content": content,
                        "cmt_rank": idx,
                        "cmt_emotion": None, # NULL 허용
                        "cmt_words": None    # NULL 허용
                    })
                
                # 댓글 대량 삽입 (Bulk Insert)
                if comment_payloads:
                    supabase.table("comments").insert(comment_payloads).execute()
                    stats["comments"] += len(comment_payloads)

            print(f" ✅ [{stats['articles']}] 기사 저장 완료 및 댓글 {len(res_data['comments'])}개 처리")

        except Exception as e:
            print(f" 저장 중 에러: {e}")

    print(f"\n🚀 작업 완료! 기사: {stats['articles']}개 / 댓글: {stats['comments']}개 저장됨")


    # --- 리포트 생성 및 저장 ---
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = (
        f"{'='*50}\n '{query_text}' 수집 최종 요약 리포트\n{'-'*50}\n"
        f"총 소요 시간: {minutes}분 {seconds}초\n"
        f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"커밋 메시지: {COMMIT_MESSAGE}\n"
        f"기사 저장: {stats['articles']}개\n"  # 변수명도 articles로 맞춤
        f"총 댓글 수: {stats['comments']}개\n"
        f"{'='*50}\n"
    )

    print("\n" + report)

    if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)
    # 파일명에도 query_text 적용
    file_path = os.path.join(SAVE_FOLDER, f"최종리포트_{query_text}_{now_str}.txt")
    with open(file_path, "w", encoding="utf-8") as f: f.write(report)
    print(f"리포트 저장 완료: {file_path}")

if __name__ == "__main__":
    asyncio.run(main_crawler("의대 증원"))