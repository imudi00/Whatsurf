# 파이프라인 (뉴스 분석 + 댓글 감정)
python llm/run_pipeline.py --keyword 탄핵 --limit 10 --batch_size 3

# 자동 라벨링 전체
python labeling/run_labeling.py --keyword 탄핵 --features all --limit 50

# 특정 피처만
python labeling/run_labeling.py --keyword 탄핵 --features frame logic bias
python labeling/run_labeling.py --keyword 탄핵 --features omission  # Pro 100건 차감
```

**추가 필요한 `.env` 키**
```
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_PRO_MODEL=gemini-2.5-pro