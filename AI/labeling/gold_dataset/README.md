[골든 데이터셋 사용 및 설명]

1. 목적
   이 데이터셋은 특정 뉴스 기사 1건에 대한 AI 성능 평가 및 품질 개선을 위한 기준 데이터셋(Golden Dataset)이다.
   AI가 생성한 결과와 비교하여 정확도 및 loss를 계산하고, 이를 기반으로 모델을 튜닝하는 데 사용한다.

---

2. 데이터 구성

(1) 원본 데이터

* raw_data/article_1.json
  → 실제 뉴스 기사 원문 + 댓글 원본

(2) 기사 골든 데이터

* article/gold_article_1.json
  → 기사에 대한 정답 라벨

포함 항목:

* frame: 기사 프레임 유형 (예: 경제적가치)
* logic: 논리 유형 (사실 전달, 주장 등)
* stance_score: 기사 논조 점수 (-1 ~ 1)
* bias_x, bias_y: 편향 좌표
* art_words: 편향 표현 단어
* body_depth: 기사 정보 밀도
* omission_risk: 정보 누락 위험도

→ 기사 전체의 구조적/의미적 특성을 정답 기준으로 정의한 데이터

(3) 댓글 골든 데이터

* comments/gold_comments_1.json
  → 댓글 전체에 대한 감정 정답

포함 항목:

* overall_sentiment: 전체 댓글 분위기
* dominant_emotion: 주요 감정
* distribution: 감정 분포
* examples: 대표 댓글과 감정 라벨

※ 댓글은 세부 감정이 아닌 “큰 감정 카테고리” 기준으로 라벨링
(분노/비난, 불안/경고, 냉소/조롱, 중립/조언)

---

3. 식별 기준

* article_id = 1
* keyword = "삼성전자 주가"
* title 기준으로 동일 기사 확인 가능

→ 반드시 동일 기사 기준으로 비교해야 함

---

4. 평가 방법

AI 결과(label_results 폴더)와 gold_dataset을 비교하여 평가

(1) 기사 평가

* frame / logic → 정확도 (accuracy)
* stance_score → 절대 오차 (L1 loss)
* bias_x, bias_y → 좌표 차이 (distance)
* omission_risk → 수치 비교

(2) 댓글 평가

* 전체 감정 (overall_sentiment) 비교
* 주요 감정 (dominant_emotion) 비교
* 감정 분포 (distribution) 비교

---

5. 활용

* 모델 성능 평가
* 프롬프트 개선
* LLM 모델 비교 실험
* 클러스터링 및 타임라인 품질 검증 기준

---

6. 특징

* 실제 뉴스 데이터를 기반으로 한 수작업 라벨링
* 기사 vs 댓글 간 인식 차이(프레이밍 vs 대중 반응) 반영
* 실서비스 적용 가능한 평가 기준 제공

---

7. 주의사항

* 반드시 동일 article_id 기준으로 AI 결과와 비교해야 함
* 댓글은 세부 감정이 아닌 “큰 감정 카테고리” 기준으로 평가
* raw_data(원본)와 gold 데이터셋을 함께 사용해야 정확한 평가 가능

---

✔ 본 데이터셋은 AI 결과의 "정답 기준(Ground Truth)"으로 사용됨