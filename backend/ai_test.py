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
        raw_body = trafilatura.extract(downloaded, include_comments=False)
        if not raw_body: return None
        body = clean_text(raw_body.strip()) # 본문 정제 추가

        # 3. 댓글 수집
        raw_comments = get_naver_comments_http(url)
        # 리스트 안의 각 댓글을 하나씩 깨끗하게 만듭니다.
        comments = [clean_text(c) for c in raw_comments]

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
    print(f"\n '{query}' 수집 및 ai_test 테이블 저장 시작 (100개 단위)")

    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
    
    # 100개 수집 설정 (start_num은 필요에 따라 변경 가능)
    display_num = 100
    start_num = 901
    api_url = f"https://openapi.naver.com/v1/search/news.json?query={urllib.parse.quote(query)}&display={display_num}&start={start_num}&sort=sim"
    
    try:
        res = requests.get(api_url, headers=headers)
        items = res.json().get("items", [])
    except Exception as e:
        print(f" API 호출 실패: {e}")
        return

    if not items: 
        print("검색 결과가 없습니다.")
        return

    # 병렬 처리 실행
    tasks = [asyncio.to_thread(crawl_task, item) for item in items]
    results = await asyncio.gather(*tasks)

    stats = {"saved": 0, "comments": 0, "skipped": 0}

    for res_data in results:
        if not res_data:
            stats["skipped"] += 1
            continue

        try:
            # 1. URL 중복 체크 (ai_test 테이블 기준)
            # 쿼리 전송 자체에서 에러가 날 수 있으므로 세부 try-except 적용
            try:
                existing_url = supabase.table("ai_test").select("id").eq("url", res_data["url"]).execute()
                if existing_url.data:
                    stats["skipped"] += 1
                    continue
            except Exception as e:
                print(f" [중복체크 에러] URL 확인 중 오류: {e}")
                continue

            # 2. 제목과 본문이 모두 일치하는 경우 체크 (ai_test 테이블 기준)
            try:
                duplicate_content = supabase.table("ai_test") \
                    .select("id") \
                    .eq("title", res_data["title"]) \
                    .eq("body", res_data["body"]) \
                    .execute()
                
                if duplicate_content.data:
                    print(f" 중복 기사 스킵(제목/본문 일치): {res_data['title'][:20]}...")
                    stats["skipped"] += 1
                    continue
            except Exception as e:
                print(f" [중복체크 에러] 내용 확인 중 오류: {e}")
                continue

            # 3. 키워드 정보 추가
            res_data["keyword"] = query

            # 4. ai_test 테이블 저장
            try:
                supabase.table("ai_test").insert([res_data]).execute()
                stats["saved"] += 1
                stats["comments"] += len(res_data['comments'])
                print(f" ✅ [{stats['saved']}] {res_data['title'][:20]}... (댓글: {len(res_data['comments'])}개)")
                
            except Exception as db_err:
                #여기서 'JSON could not be generated'
                print(f"\n [DB 저장 실패] 기사: {res_data['title'][:20]}")
                print(f"    - 이유: {db_err}")
                print(f"    - URL: {res_data['url']}")
                stats["skipped"] += 1

        except Exception as unknown_e:
            print(f"[알 수 없는 로직 에러]: {unknown_e}")
            stats["skipped"] += 1

    # --- 리포트 생성 및 저장 ---
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = (
        f"{'='*50}\n '{query}' ai_test 수집 요약 리포트\n{'-'*50}\n"
        f"총 소요 시간: {minutes}분 {seconds}초\n"
        f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"신규 저장: {stats['saved']}개\n"
        f"스킵/실패: {stats['skipped']}개\n"
        f"총 댓글 수: {stats['comments']}개\n{'='*50}\n"
    )

    print("\n" + report)

    if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)
    file_path = os.path.join(SAVE_FOLDER, f"ai_test_리포트_{query}_{now_str}.txt")
    with open(file_path, "w", encoding="utf-8") as f: f.write(report)
    print(f"리포트 저장 완료: {file_path}")

if __name__ == "__main__":
    asyncio.run(main_crawler("등록금 인상"))