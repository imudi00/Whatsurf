# pipeline.py
from keyword_extractor import extract_features
from preprocessor import preprocess
from body_depth import compute_body_depth
from omission_risk import compute_omission_risk
from llm_frame import extract_frame
from llm_logic import extract_logic
from llm_stance import extract_stance

def analyze_article(text: str, cluster_articles: list[str] = None) -> dict:
    # Step 1: 키워드 추출 (NER)
    features = extract_features(text)
    
    # Step 2: 구조 분해 (전처리)
    struct = preprocess(text)
    
    # Step 3: 맥락 피처 (규칙 기반, 빠름)
    depth = compute_body_depth(text, features)
    omission = compute_omission_risk(text, cluster_articles, extract_features) \
               if cluster_articles else "low"
    
    # Step 4: 논조 피처 (LLM, 순서대로 — 앞 결과가 뒤에 맥락으로 전달)
    frame_result  = extract_frame(struct)
    logic_result  = extract_logic(struct)
    stance_result = extract_stance(struct, frame_result["frame"], logic_result["logic"])
    
    return {
        # 최종 피처
        "frame":        frame_result["frame"],
        "logic":        logic_result["logic"],
        "stance_score": stance_result["stance_score"],
        "body_depth":   depth,
        "omission_risk": omission,
        # 설명 (디버깅/검증용)
        "_frame_reason":   frame_result.get("reason"),
        "_logic_reason":   logic_result.get("reason"),
        "_dominant_tone":  stance_result.get("dominant_tone"),
        "_key_evidence":   stance_result.get("key_evidence"),
        "_entities":       [e['word'] for e in features["entities"][:5]],
    }

if __name__ == "__main__":
    sample = """"대구·경북 통합해달라"‥그러면 충남·대전은? 외통수 몰린 국민의힘◀ 앵커 ▶
어제 무제한 토론을 돌연 중단한 국민의힘은 더불어민주당에 대구·경북 행정통합을 요구하고 있는데요.
하지만 민주당은 대구·경북 통합을 위해서는 충남·대전 통합법 처리에 협조하라며 국민의힘을 압박하고 있습니다.
이재욱 기자가 보도합니다.
◀ 리포트 ▶
대구·경북 통합을 요구하며 무제한 토론을 끝낸 국민의힘.
민주당을 향해 빨리 법사위를 열어 법안을 처리해 달라고 요구했습니다.
[송언석/국민의힘 원내대표]
"오늘이라도 법사위와 원포인트 본회의를 열어서 대구·경북 특별법을 처리할 것을 촉구합니다."
여당 주도로 전남·광주 통합법만 처리되면서 지역 민심이 이탈할 조짐이 일자, 사실상 백기를 든 셈입니다.
민주당은 쉽게 내주지 않겠다면서 국민의힘에 두 가지를 요구했습니다.
대구·경북 통합에 반대했다가 일주일 만에 찬성으로 돌아서며 혼선을 일으킨 데 대해 대국민 사과를 하고, 충남·대전 통합에 대해서도 찬성인지 반대인지 당론을 명확히 하라는 겁니다.
[한병도/더불어민주당 원내대표]
"대구·경북에 단일한 의견을 만들어 오고, 대전·충남에 단일한 의견을 만들어 와야 됩니다."
민주당은 대구·경북 통합 법안 처리를 지렛대 삼아, 내친김에 논의가 지지부진한 충남·대전 통합까지 이뤄내겠다는 심산입니다.
하지만 국민의힘은 민주당의 요구에 쉽게 응하지 못하고 있습니다.
가뜩이나 수세에 몰린 상황에서 대국민 사과가 부담스럽고, 자당 출신 시도지사의 의견을 들어 대전·충남 통합에 반대했다가는 지방선거 역풍이 뻔하기 때문입니다.
여기에 애초 행정 통합의 판을 자신들이 엎은 탓에 대구·경북만 밀어붙이기에도 명분 찾기가 쉽지 않은 상황입니다.
[정청래/더불어민주당 대표(어제)]
"대전·충남 통합이 무산되면 그 책임은 100% 국민의힘에게 있다는 사실을 분명히 말씀드립니다."
오는 6월 지방선거에서 통합선거를 치르려면 대구·경북 행정통합 법안 처리 일정은 매우 촉박한 상황입니다.
정치적으로 궁지에 몰린 국민의힘이 어떤 선택을 할지 관심이 쏠리고 있습니다. """
    import json
    result = analyze_article(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))