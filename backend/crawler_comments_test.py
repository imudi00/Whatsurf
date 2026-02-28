import os
import requests
import trafilatura
import re
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Supabase 설정
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_naver_comments(url):
    try:
        # 1. oid, aid 추출 강화 (숫자만 정확히)
        oid_match = re.search(r"article/(\d+)", url)
        aid_match = re.search(r"article/\d+/(\d+)", url)
        
        if not oid_match or not aid_match:
            print("⚠️ URL 형식이 잘못되었습니다.")
            return []

        oid = oid_match.group(1)
        aid = aid_match.group(1)
        
        # 2. 네이버 댓글 API 주소 (파라미터 풀 버전)
        # ticket=news, templateId=view_main 등이 누락되면 안 됩니다.
        api_url = "https://apis.naver.com/comment/all/static/v2/getContents"
        params = {
            "ticket": "news",
            "templateId": "view_main",
            "pool": "cbox5",
            "lang": "ko",
            "country": "KR",
            "objectId": f"news{oid},{aid}",
            "pageSize": 10,
            "page": 1,
            "sort": "FAVORITE"
        }
        
        headers = {
            "Referer": url,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        print(f"📡 API 요청 중: news{oid},{aid}")
        response = requests.get(api_url, params=params, headers=headers)
        
        if response.status_code != 200:
            print(f"❌ API 연결 실패: {response.status_code}")
            return []
            
        data = response.json()
        
        # 데이터가 잘 왔는지 구조 확인용 출력
        if not data.get("success"):
            print(f"⚠️ 네이버 서버 응답 메시지: {data.get('message')}")
            return []

        comment_list = data.get("result", {}).get("commentList", [])
        comments = [c.get("contents").replace("\n", " ").strip() for c in comment_list if c.get("contents")]
        
        return comments
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return []

def final_check(url):
    print(f"🚀 [최종 검증 시작] {url}")
    
    # 본문 추출
    downloaded = trafilatura.fetch_url(url)
    body = trafilatura.extract(downloaded)
    
    # 댓글 추출
    comments = get_naver_comments(url)
    
    if comments:
        print(f"✅ 수집 성공! ({len(comments)}개)")
        for i, c in enumerate(comments[:3], 1):
            print(f"   [{i}] {c[:50]}...")
            
        # DB 저장
        item = {
            "title": "댓글 수집 최종 테스트",
            "url": url,
            "body": body[:500] if body else "본문 없음",
            "media": "네이버뉴스",
            "published": datetime.now().strftime("%Y-%m-%d"),
            "comments": comments 
        }
        try:
            supabase.table("news").insert([item]).execute()
            print("💾 Supabase 저장 완료!")
        except Exception as e:
            print(f"💾 DB 저장 에러: {e}")
    else:
        print("❌ 여전히 댓글이 0개로 나옵니다. (기사 식별은 되었으나 데이터가 없음)")

if __name__ == "__main__":
    url = "https://n.news.naver.com/article/448/0000591505?iid=1439"
    final_check(url)