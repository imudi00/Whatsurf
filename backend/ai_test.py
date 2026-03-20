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

def clean_text(text):
    """JSON 에러를 방지하기 위해 특수 유니코드 및 이모지 제거"""
    if not text: return ""
    # 한글, 영문, 숫자, 기본 문장부호만 남기고 제거
    return re.sub(r'[^\x00-\x7F가-힣ㄱ-ㅎㅏ-ㅣ\s.,!?\'\"%()-]', '', text)

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
            "objectId": object_id, "pageSize": 100, "indexSize": 10,
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
    """개별 기사를 수집하여 ai_test 테이블 형식으로 변환 (타임스탬프 및 이상치 처리 적용)"""
    url = item.get("link")
    if "n.news.naver.com" not in url and "news.naver.com" not in url:
        return None

    title_clean = item.get("title").replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&")
    
    try:
        # 1. 발행일 파싱 (시각 포함 타임스탬프 형식)
        raw_pub_date = item.get("pubDate")
        try:
            # 시, 분, 초까지 포함된 datetime 객체 생성
            clean_date_obj = datetime.strptime(raw_pub_date, "%a, %d %b %Y %H:%M:%S +0900")
            # YYYY-MM-DD HH:MM:SS 형식으로 변환
            published_date = clean_date_obj.strftime("%Y-%m-%d %H:%M:%S")
        except:
            # 파싱 실패 시 현재 시각 대신 None을 넣어 이상치로 분리 (DB에는 NULL로 기록됨)
            published_date = None 
            print(f"⚠️ 날짜 파싱 실패 (NULL 처리): {title_clean[:15]}...")

        # 2. 본문 수집
        downloaded = trafilatura.fetch_url(url)
        raw_body = trafilatura.extract(downloaded, include_comments=False)
        if not raw_body: return None
        body = clean_text(raw_body.strip()) 

        # 3. 댓글 수집
        raw_comments = get_naver_comments_http(url)
        comments = [clean_text(c) for c in raw_comments]

        return {
            "title": title_clean,
            "body": body.strip(),
            "url": url,
            "published": published_date, # NULL 혹은 '2026-02-24 14:00:00' 형태
            "comments": comments  
        }
    except:
        return None

async def main_crawler(query):
    start_time = time.time()
    print(f"\n '{query}' 1000개 대량 수집 및 ai_test 저장 시작")

    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
    stats = {"saved": 0, "comments": 0, "skipped": 0}

    # 1. 100개씩 10번 반복하여 총 1000개 수집 (start 파라미터 활용)
    for start_num in range(1, 1001, 100):
        print(f"\n--- 현재 수집 구간: {start_num} ~ {start_num + 99} ---")
        
        api_url = f"https://openapi.naver.com/v1/search/news.json?query={urllib.parse.quote(query)}&display=100&start={start_num}&sort=sim"
        
        try:
            res = requests.get(api_url, headers=headers)
            items = res.json().get("items", [])
            if not items:
                print(f" 구간 {start_num}: 결과가 더 이상 없습니다. 수집을 종료합니다.")
                break

            # 2. 병렬 처리 실행 (현재 구간 100개 기사)
            tasks = [asyncio.to_thread(crawl_task, item) for item in items]
            results = await asyncio.gather(*tasks)

            for res_data in results:
                if not res_data:
                    stats["skipped"] += 1
                    continue

                try:
                    # (1) URL 중복 체크
                    existing_url = supabase.table("ai_test").select("id").eq("url", res_data["url"]).execute()
                    if existing_url.data:
                        stats["skipped"] += 1
                        continue

                    # (2) 제목+본문 중복 체크
                    duplicate_content = supabase.table("ai_test") \
                        .select("id") \
                        .eq("title", res_data["title"]) \
                        .eq("body", res_data["body"]) \
                        .execute()
                    
                    if duplicate_content.data:
                        stats["skipped"] += 1
                        continue

                    # (3) 키워드 정보 주입
                    res_data["keyword"] = query

                    # (4) DB 저장 및 예외 처리
                    try:
                        supabase.table("ai_test").insert([res_data]).execute()
                        stats["saved"] += 1
                        stats["comments"] += len(res_data['comments'])
                        
                        if stats["saved"] % 10 == 0:
                            print(f" ✅ {stats['saved']}개 완료...")
                            
                    except Exception as db_err:
                        print(f"\n ❌ [저장 실패] 기사: {res_data['title'][:15]} | 에러: {db_err}")
                        stats["skipped"] += 1

                except Exception as e:
                    print(f" ⚠️ 기사 처리 중 오류: {e}")
                    stats["skipped"] += 1
            
            # --- 구간 사이 시간 지연 (중요!) ---
            # 100개 처리가 끝날 때마다 1.5초간 쉬어줍니다. (API & DB 과부하 방지)
            print(f"--- {start_num}구간 완료, 안정적인 처리를 위해 잠시 대기합니다. ---")
            await asyncio.sleep(1.5) 

        except Exception as api_err:
            print(f" 🚨 API 호출 구간 에러 ({start_num}): {api_err}")
            continue

    # --- 리포트 생성 및 저장 ---
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = (
        f"{'='*50}\n '{query}' 1000개 수집 최종 리포트\n{'-'*50}\n"
        f"총 소요 시간: {minutes}분 {seconds}초\n"
        f"신규 저장: {stats['saved']}개\n"
        f"스킵/실패: {stats['skipped']}개\n"
        f"총 댓글 수: {stats['comments']}개\n{'='*50}\n"
    )

    print("\n" + report)
    if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)
    file_path = os.path.join(SAVE_FOLDER, f"1000개_리포트_{query}_{now_str}.txt")
    with open(file_path, "w", encoding="utf-8") as f: f.write(report)
    print(f"리포트 저장 완료: {file_path}")

if __name__ == "__main__":
    asyncio.run(main_crawler("어도어 민희진 갈등"))