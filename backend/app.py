import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 1. 현재 파일(app.py)의 위치를 기준으로 한 칸 위(프로젝트 루트) 경로 계산
root_path = Path(__file__).resolve().parents[1]

# 2. 그 경로가 파이썬이 파일을 찾는 명단(sys.path)에 없으면 추가
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
    
#ai 파이프라인 갖고오기
from AI.pipeline.run_pipeline import run_pipeline

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from supabase import create_client, Client
import math
import asyncio

#crawler.py에서 main_crawler 함수를 가져옵니다.
from crawler import main_crawler

app = FastAPI(title="Whatsurf API Server")

#CORS 설정
origins = [
    "http://localhost:3000", #로컬용
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    # 3000번 포트 허용
    allow_credentials=True,
    allow_methods=["*"],      # 모든 HTTP 메서드(GET, POST 등) 허용
    allow_headers=["*"],      # 모든 헤더 허용
)

# Supabase 설정
SUPABASE_URL = os.getenv("SUPABASE_URL_backend")
SUPABASE_KEY = os.getenv("SUPABASE_KEY_backend")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Request Models ---
class QueryRequest(BaseModel):
    query_text: str

# --- API Endpoints ---

# 1. 수집과 분석을 순차적으로 실행하는 통합 비동기 함수
async def run_full_process(query_text: str, query_id: int):
    try:
        # Step A: 크롤러 실행 (비동기 함수이므로 await 필수)
        print(f"--- [Step 1] Crawler 시작: {query_text} (ID: {query_id}) ---")
        await main_crawler(query_text, query_id) 
        
        # Step B: AI 파이프라인 시작 
        # run_pipeline이 일반 함수(sync)라면 그대로 호출, 
        # 만약 내부에서 대기 시간이 길다면 별도 스레드에서 돌리는 방법도 있지만 
        # 우선은 직관적으로 호출합니다.
        print(f"--- [Step 2] AI Pipeline 시작 (ID: {query_id}) ---")
        
        # 동기 함수를 비동기 루프에서 안전하게 실행 (권장)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: run_pipeline(
            query_id=query_id, 
            steps=["clustering", "feature_map", "timeline"]
        ))
        
        print(f"--- [Success] 모든 공정 완료 (ID: {query_id}) ---")
        
    except Exception as e:
        import traceback
        print(f"--- [Error] 작업 중 오류 발생 (ID: {query_id}): {e} ---")
        traceback.print_exc()


# 1. 검색어 입력 및 쿼리 ID 생성 [cite: 3]
@app.post("/api/queries")
async def create_query(request: QueryRequest, background_tasks: BackgroundTasks):
    try:
        # 1. [DB 저장] queries 테이블에 먼저 데이터를 넣어서 'id'를 발급받습니다.
        query_data = supabase.table("queries").insert({
            "query_text": request.query_text,
            "requested_at": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }).execute()
        
        # 발급된 ID와 검색어 추출
        query_id = query_data.data[0]['id']
        query_text = query_data.data[0]['query_text']

        # 통합 프로세스 등록
        background_tasks.add_task(run_full_process, query_text, query_id)
        
        # 3. [응답] 사용자에게는 바로 ID를 돌려줍니다. (수집은 백그라운드에서 진행)
        return {
            "status": "success", 
            "data": {
                "id": query_id,
                "query_text": query_text,
                "message": "수집 및 분석이 시작되었습니다."
            }
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="쿼리 생성 및 크롤러 시작 실패")

# 2. 클러스터 목록 조회 (vw_cluster_list)
@app.get("/api/queries/{query_id}/clusters")
async def get_cluster_list(query_id: int):
    # 칼럼: query_id, cluster_label, cluster_count
    res = supabase.table("vw_cluster_list").select("*").eq("query_id", query_id).execute()
    return {"status": "success", "data": {"id": query_id, "clusters": res.data}}

# 3. 클러스터별 기사 요약 조회 (vw_cluster_detail)
@app.get("/api/queries/{query_id}/clusters/{cluster_label}")
async def get_cluster_detail(query_id: int, cluster_label: int):
    try:
        # 칼럼: query_id, cluster_label, cluster_title, cluster_summary, rep_article_url, article_count, leading_sources, representative_comments
        # 이 뷰 하나에 언론사 비중과 대표 댓글이 모두 포함되어 있어 효율적입니다.
        res = supabase.table("vw_cluster_detail") \
            .select("*") \
            .eq("query_id", query_id) \
            .eq("cluster_label", cluster_label) \
            .execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail="Cluster data not found")
        
        return {
            "status": "success",
            "data": res.data[0]
        }
    except Exception as e:
        print(f"Error in 3번 기능: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 4. 클러스터별 피처맵 조회 (vw_cluster_feature_ratio)
@app.get("/api/queries/{query_id}/clusters/{cluster_label}/feature-map")
async def get_feature_map(query_id: int, cluster_label: int):
    # 칼럼: query_id, cluster_label, feature_map(jsonb)
    res = supabase.table("vw_cluster_feature_ratio") \
        .select("*") \
        .eq("query_id", query_id) \
        .eq("cluster_label", cluster_label) \
        .execute()
    
    return {"status": "success", "data": res.data[0] if res.data else {}}

# 5. 타임라인 조회 (vw_timeline_with_article)
@app.get("/api/queries/{query_id}/timeline")
async def get_timeline(query_id: int):
    # 칼럼: query_id, timeline_date, article_title, article_url
    res = supabase.table("vw_timeline_with_article").select("*").eq("query_id", query_id).execute()
    return {"status": "success", "data": {"id": query_id, "timelines": res.data}}

# 6. 언론사 분포 좌표평면 조회 (vw_article_bias_point)
@app.get("/api/queries/{query_id}/bias-plane")
async def get_bias_plane(query_id: int):
    # 칼럼: query_id, source_name, center_x, center_y, spread_radius
    res = supabase.table("vw_article_bias_point").select("*").eq("query_id", query_id).execute()
    return {"status": "success", "data": {"id": query_id, "media_distribution": res.data}}

# 7. 좌표평면 수치별 기사 직접 조회 (articles 테이블)
@app.get("/api/queries/{query_id}/bias-plane/match")
async def match_article_by_vector(query_id: int, x: float, y: float):
    try:
        # 1. article_features 테이블에서 해당 쿼리의 좌표 데이터들을 가져옵니다.
        # 칼럼명 반영: bias_x, bias_y
        # 기사 정보(title, url)를 함께 가져오기 위해 articles 테이블과 join하여 조회합니다.
        res = supabase.table("article_features") \
            .select("bias_x, bias_y, articles(id, title, url)") \
            .eq("articles.query_id", query_id) \
            .execute()
        
        if not res.data:
            return {"status": "success", "data": {"matched_article": None}}

        # 2. 유클리드 거리 계산 로직 (가장 가까운 기사 찾기) 
        # 거리 = sqrt((x2-x1)^2 + (y2-y1)^2)
        matched_article = None
        min_distance = float('inf')

        for item in res.data:
            # 1. 먼저 articles 데이터를 가져옵니다.
            article_info = item.get("articles")
            
            # [수정 핵심] article_info가 None이 아닐 때만 계산을 진행합니다.
            if article_info:
                # article_features의 bias_x, bias_y 사용
                article_x = item.get("bias_x", 0)
                article_y = item.get("bias_y", 0)
                
                # 입력값(x, y)와의 거리 계산
                distance = math.sqrt((x - article_x)**2 + (y - article_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    # matched_article 구성을 if문 안으로 넣어서 안전하게 처리합니다.
                    matched_article = {
                        "id": article_info.get("id"),
                        "title": article_info.get("title"),
                        "url": article_info.get("url"),
                        "bias_vector1": article_x,
                        "bias_vector2": article_y,
                        "distance": round(distance, 4)
                    }
            else:
                # 기사 정보가 없는 데이터는 무시하고 다음으로 넘어갑니다.
                continue
            
        return {
            "status": "success", 
            "data": {
                "id": query_id,
                "input_vector": {"x": x, "y": y},
                "matched_article": matched_article
            }
        }
    except Exception as e:
        print(f"Match Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)