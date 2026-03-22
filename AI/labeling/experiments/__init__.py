# labeling/experiments/__init__.py
"""
피처 실험 프레임워크

Phase 1 — 정답 수집
    python -m labeling.experiments.utils.data_loader annotate \
        --labeled_dir labeling/output \
        --out experiments/data/ground_truth.jsonl

Phase 2 — 규칙 기반 피처 튜닝 (정답 없이도 실행 가능)
    python labeling/experiments/rule_based/body_depth_tuner.py \
        --articles experiments/data/ground_truth.jsonl

Phase 3 — 모델 성능 평가 (정답 필요)
    python labeling/experiments/model_eval/frame_logic_eval.py \
        --ground_truth experiments/data/ground_truth.jsonl \
        --predictions labeling/output
"""
