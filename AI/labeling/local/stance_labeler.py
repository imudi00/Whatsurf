# labeling/local/stance_labeler.py
"""
로컬 HuggingFace — stance_score 자동 라벨링
권장 모델: snunlp/KR-FinBert-SC (감성분석) 또는 파인튜닝 체크포인트

모델 출력 → [-1.0 ~ +1.0] 연속값으로 변환
  긍정(positive) 확률 → +1.0 방향
  부정(negative) 확률 → -1.0 방향
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 모델 설정 ──────────────────────────────────────────────
# snunlp/KR-FinBert-SC : label 0=negative, 1=neutral, 2=positive
DEFAULT_MODEL  = "snunlp/KR-FinBert-SC"
NEG_IDX, NEU_IDX, POS_IDX = 0, 1, 2


@dataclass
class StanceLabel:
    id:           str
    stance_score: float   # -1.0 ~ +1.0
    dominant_tone: str    # 비판적 / 중립 / 우호적


class LocalStanceLabeler:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"  [LocalStance] 모델 로드: {model_name} ({self.device})")

    @torch.no_grad()
    def label_batch(self, items: List[dict]) -> List[StanceLabel]:
        """items: [{"id":..., "text":...}]"""
        results = []
        for item in items:
            enc    = self.tokenizer([item["text"]], padding=True, truncation=True,
                                    max_length=256, return_tensors="pt")
            logits = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device)
            ).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # stance_score: pos - neg ([-1,1] 범위)
            score = float(probs[POS_IDX]) - float(probs[NEG_IDX])
            if score > 0.15:   tone = "우호적"
            elif score < -0.15: tone = "비판적"
            else:              tone = "중립"

            results.append(StanceLabel(
                id            = str(item["id"]),
                stance_score  = round(score, 4),
                dominant_tone = tone,
            ))
        return results


_labeler: Optional[LocalStanceLabeler] = None

def get_labeler(model_name: str = DEFAULT_MODEL) -> LocalStanceLabeler:
    global _labeler
    if _labeler is None:
        _labeler = LocalStanceLabeler(model_name)
    return _labeler


def label_stance(items: List[dict], model_name: str = DEFAULT_MODEL) -> List[dict]:
    """편의 함수. items=[{"id":...,"text":...}]"""
    return [
        {"id": o.id, "stance_score": o.stance_score, "dominant_tone": o.dominant_tone}
        for o in get_labeler(model_name).label_batch(items)
    ]


def label_stance_batch(texts: List[str], model_name: str = DEFAULT_MODEL) -> List[dict]:
    """
    run_labeling.py 인터페이스. texts = [str, ...]
    반환: [{"stance_score": float, "stance_label": str}]
    """
    items = [{"id": str(i), "text": t} for i, t in enumerate(texts)]
    raw = label_stance(items, model_name)
    return [
        {"stance_score": r["stance_score"],
         "stance_label": r["dominant_tone"]}
        for r in raw
    ]
