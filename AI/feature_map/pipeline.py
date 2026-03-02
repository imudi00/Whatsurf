# pipeline.py
from keyword_extractor import extract_features
from body_depth import compute_body_depth
from omission_risk import compute_omission_risk
from llm_features import extract_tone_features 

def analyze_article(text: str, cluster_articles: list[str] = None) -> dict:
    # Step 1: 키워드 추출
    features = extract_features(text)

    # Step 2: 맥락 피처 (규칙 기반)
    depth = compute_body_depth(text, features)
    omission = "low"
    if cluster_articles:
        omission = compute_omission_risk(text, cluster_articles, extract_features)

    # Step 3: 논조 피처 (LLM, 단일 호출)
    tone = extract_tone_features(text, features)

    return {
        #키워드
        "keywords": features,
        # 논조
        "frame": tone["frame"],
        "logic": tone["logic"],
        "stance_score": tone["stance_score"],
        # 맥락
        "body_depth": depth,
        "omission_risk": omission,
        # 디버깅
        "_entities": [e['word'] for e in features["entities"][:10]],
    }

# 테스트
if __name__ == "__main__":
    sample = """
李, 싱가포르 총리에게 “제가 낚시를” 언급 이유는…
가포르를 국빈 방문 중인 이재명 대통령은 2일(현지 시간) 로렌스 웡 싱가포르 총리와 오찬을 가졌다.2일 이 대통령의 공식 유튜브 채널에는 “잼며들었웡”이라는 제목의 영상이 올라왔다. 영상에는 이 대통령과 김혜경 여사가 웡 총리 내외와 친교 오찬을 갖는 모습이 담겼다.영상에서 이 대통령은 차량에서 내려 마중 나온 웡 총리와 악수했다. 오찬장으로 이동한 이 대통령과 김 여사는 웡 총리 내외와 싱가포르의 전통 음식인 생선회 샐러드 유생을 함께 비볐다. 유생은 먹는 사람들이 재료 하나씩을 더하며 덕담을 나누는 음식이다.웡 총리는 유생에 대해 “중국의 생선회 샐러드에서 발전이 됐다”고 설명했다. 이어 웡 총리는 이 대통령 부부와 유생을 함께 비비며 “보다 좋은 한국과 싱가포르의 관계를 위하여”라고 말했다. 이 대통령은 “총리님과 영부인님의 행복과 건강을 위하여”라고 화답했다. 이 대통령은 비벼진 유생을 보며 “이거 아주 맛있을 거 같다”고 덧붙였다.이후 이 대통령은 웡 총리에게 “제가 낚시를 좋아한다”고 말했다. ‘어떤 낚시를 좋아하느냐’는 물음엔 “호수 낚시를 좋아한다”고 답했다. 이어 입을 벌린 채 입질을 기다린다며 상황을 재현하기도 했다. 이 대통령은 오찬을 마친 뒤 웡 총리와 악수하며 “자주 보길 바란다”고 말했다.
"""
    result = analyze_article(sample)
    print(result)
    # 예상 출력:
    # {
    #   "frame": "경제적가치",
    #   "logic": "파급효과",       ← 전문가 경고 + 파급 설명이 중심
    #   "stance_score": -0.5,      ← 비판적 논조
    #   "body_depth": 0.58,
    #   "omission_risk": "low",
    #   "_entities": ["산업통상자원부", "한국전력", "소상공인", ...]
    # }