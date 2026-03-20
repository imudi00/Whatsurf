import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI(title="Whatsurf API Server")

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
async def create_query(request: QueryRequest):
    try:
        data = supabase.table("queries").insert({
            "query_text": request.query_text,
            "requested_at": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }).execute()
        
        return {"status": "success", "data": data.data[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. 클러스터 목록 조회 (vw_cluster_list 활용) [cite: 3]
@app.get("/api/queries/{query_id}/clusters")
async def get_cluster_list(query_id: int):
    try:
        # 가상 뷰 vw_cluster_list에서 해당 query_id 데이터 조회 [cite: 3]
        res = supabase.table("vw_cluster_list").select("*").eq("query_id", query_id).execute()
        if not res.data:
            return {"status": "success", "data": {"id": query_id, "clusters": []}}
            
        return {"status": "success", "data": {
            "id": query_id,
            "clusters": res.data
        }}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. 클러스터별 상세 요약 조회 (vw_cluster_detail 활용) [cite: 3]
@app.get("/api/queries/{query_id}/clusters/{cluster_label}")
async def get_cluster_detail(query_id: int, cluster_label: int):
    try:
        # 가상 뷰 vw_cluster_detail에서 상세 정보 조회 [cite: 3]
        detail_res = supabase.table("vw_cluster_detail").select("*").eq("query_id", query_id).eq("label", cluster_label).execute()
        
        # 가상 뷰 vw_cluster_source_stats에서 언론사 통계 조회 [cite: 3]
        source_res = supabase.table("vw_cluster_source_stats").select("*").eq("query_id", query_id).eq("label", cluster_label).execute()

        if not detail_res.data:
            raise HTTPException(status_code=404, detail="Cluster not found")

        return {
            "status": "success",
            "data": {
                **detail_res.data[0],
                "leading_sources": source_res.data
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. 클러스터별 피처맵(비중) 조회 (vw_query_cluster_feature_ratio 활용) [cite: 9]
@app.get("/api/queries/{query_id}/clusters/{cluster_label}/feature-map")
async def get_feature_map(query_id: int, cluster_label: int):
    try:
        res = supabase.table("vw_query_cluster_feature_ratio").select("*").eq("query_id", query_id).eq("label", cluster_label).execute()
        return {"status": "success", "data": res.data[0] if res.data else {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. 타임라인 조회 (vw_query_timeline_with_article 활용) [cite: 13]
@app.get("/api/queries/{query_id}/timeline")
async def get_timeline(query_id: int):
    try:
        res = supabase.table("vw_query_timeline_with_article").select("*").eq("query_id", query_id).execute()
        return {"status": "success", "data": {"id": query_id, "timelines": res.data}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. 언론사 분포 좌표평면 조회 (vw_article_bias_point 활용) [cite: 13]
@app.get("/api/queries/{query_id}/bias-plane")
async def get_bias_plane(query_id: int):
    try:
        res = supabase.table("vw_article_bias_point").select("*").eq("query_id", query_id).execute()
        return {"status": "success", "data": {"id": query_id, "media_distribution": res.data}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 7. 좌표평면 수치별 기사 직접 조회 (articles 테이블 직접 조회) 
@app.get("/api/queries/{query_id}/bias-plane/match")
async def match_article_by_vector(query_id: int, x: float, y: float):
    try:
        # 명세서에 따라 이 기능만 가상 뷰가 아닌 실제 articles 테이블에서 필터링하여 가져옵니다. 
        # x, y 좌표값과 가장 가까운 기사를 찾는 로직은 DB 함수(RPC)를 쓰거나 앱 단에서 처리 가능합니다.
        # 여기서는 단순 쿼리 예시를 보여드립니다.
        res = supabase.table("articles") \
            .select("id, title, url, bias_vector1, bias_vector2") \
            .eq("query_id", query_id) \
            .execute()
        
        # (참고) 실제 구현 시에는 유클리드 거리가 가장 가까운 1개를 반환하는 로직이 추가되어야 합니다.
        return {"status": "success", "data": {"matched_article": res.data[0] if res.data else None}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)