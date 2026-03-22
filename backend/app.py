import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
import math

#crawler.py에서 main_crawler 함수를 가져옵니다.
from crawler import main_crawler

load_dotenv()

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
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Request Models ---
class QueryRequest(BaseModel):
    query_text: str

# --- API Endpoints ---

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

        # 2. [크롤러 호출] 발급받은 query_id를 크롤러에게 넘겨줍니다.
        # 이제 crawler.py의 main_crawler는 이 ID를 사용해 기사를 저장합니다.
        background_tasks.add_task(main_crawler, query_text, query_id)
        
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
            # article_features의 bias_x, bias_y 사용
            article_x = item.get("bias_x", 0)
            article_y = item.get("bias_y", 0)
            
            # 입력값(x, y)와의 거리 계산
            distance = math.sqrt((x - article_x)**2 + (y - article_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                # 명세서 응답 형식에 맞게 데이터 재구성 
                article_info = item.get("articles", {})
                matched_article = {
                    "id": article_info.get("id"),
                    "title": article_info.get("title"),
                    "url": article_info.get("url"),
                    "bias_vector1": article_x,
                    "bias_vector2": article_y,
                    "distance": round(distance, 4)
                }

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