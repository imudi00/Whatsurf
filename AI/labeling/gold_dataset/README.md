[골든 데이터셋 사용 및 설명]

1. 데이터 구조
AI/
└─ labeling/
   └─ gold_dataset/
      ├─ raw_data/
      │  └─ article_1.json ... 뉴스 기사 원문, 댓글 원문
      ├─ article/
      │  └─ gold_article_1.json ... 기사 골든 데이터
      ├─ comments/
      │  └─ gold_comments_1.json ... 댓글 골든 데이터
      └─ evaluation/
         ├─ compare_article.py
         └─ compare_comments.py

---

2. 기사 골든 데이터
article/gold_article_1.json

기사 전체에 대한 정답 라벨:

stance_score : 기사 논조 (-1 ~ 1)
stance_label : 긍정 / 중립 / 부정
body_depth : 정보 밀도
body_depth_label : low / medium / high
art_words : 편향 표현 단어
loaded_word_density : 감정적 표현 밀도
is_biased : 편향 여부

---

3. 감정 분류 기준

댓글 감정은 아래 5개 대분류로 통일:

anger : 분노, 비난, 공격
sadness : 낙담, 우울, 체념
fear : 불안, 경고, 위험 인식
hurt : 억울함, 피해감, 배신감
joy : 긍정, 기대, 낙관

---