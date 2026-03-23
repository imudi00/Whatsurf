# labeling/local/emotion_labeler.py
"""
로컬 HuggingFace — 한국어 감정 분류 (댓글용)

기본 모델: hun3359/klue-bert-base-sentiment
  - KLUE-BERT 기반, 한국어 감정 데이터셋으로 파인튜닝
  - 라벨: 모델 config.id2label 에서 자동 로드 (Korean labels)
  - GPU 없으면 CPU로 자동 전환

.env 설정:
    EMOTION_MODEL=hun3359/klue-bert-base-sentiment  (기본값)
"""
import os, sys
from pathlib import Path
from typing import List, Optional

# ── 설정 ─────────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("EMOTION_MODEL", "hun3359/klue-bert-base-sentiment")

# 한국어 감정 레이블 → 영어 매핑 (모델 라벨이 한글인 경우 변환)
_KO_TO_EN = {
    "기쁨":   "joy",     "행복":   "joy",    "즐거움": "joy",
    "슬픔":   "sadness", "우울":   "sadness",
    "분노":   "anger",   "화남":   "anger",
    "두려움": "fear",    "불안":   "fear",    "공포":   "fear",
    "놀람":   "surprise","당혹감": "surprise",
    "혐오":   "disgust", "싫음":   "disgust",
    "상처":   "hurt",
    "중립":   "neutral", "보통":   "neutral",
    # 영어 레이블 (이미 영어인 경우 그대로)
    "joy": "joy", "sadness": "sadness", "anger": "anger",
    "fear": "fear", "surprise": "surprise", "disgust": "disgust",
    "neutral": "neutral", "positive": "positive", "negative": "negative",
}


# ── 모델 로더 ─────────────────────────────────────────────────

class LocalEmotionLabeler:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"  [LocalEmotion] 모델 로드 중: {model_name} ({self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # 모델의 실제 라벨 목록 (config.id2label 우선)
        id2label = getattr(self.model.config, "id2label", {})
        if id2label:
            self.labels = [id2label[i] for i in range(len(id2label))]
        else:
            self.labels = [str(i) for i in range(self.model.config.num_labels)]

        print(f"  [LocalEmotion] 라벨: {self.labels}")

    def _map_label(self, raw: str) -> str:
        """한국어/영어 → 통일된 영어 레이블. 매핑 없으면 원본 반환."""
        return _KO_TO_EN.get(raw, raw)

    def label_batch(self , texts: List[str]) -> List[dict]:
        import torch, numpy as np
        results = []
        for text in texts:
            enc = self.tokenizer(
                text, padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self.model(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device),
                ).logits[0]
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            best_i = int(np.argmax(probs))
            raw_label = self.labels[best_i]
            results.append({
                "primary_emotion":   self._map_label(raw_label),
                "raw_label":         raw_label,          # 모델 원본 라벨 보존
                "emotion_intensity": round(float(np.max(probs)), 4),
                "emotion_probs":     {
                    self._map_label(l): round(float(probs[i]), 4)
                    for i, l in enumerate(self.labels)
                },
            })
        return results


# ── 싱글턴 ───────────────────────────────────────────────────

_labeler: Optional[LocalEmotionLabeler] = None

def _get_labeler(model_name: str = DEFAULT_MODEL) -> LocalEmotionLabeler:
    global _labeler
    if _labeler is None or _labeler.model_name != model_name:
        _labeler = LocalEmotionLabeler(model_name)
    return _labeler


# ── 퍼블릭 인터페이스 ─────────────────────────────────────────

def label_emotion_batch(texts: List[str], model_name: str = DEFAULT_MODEL) -> List[dict]:
    """
    run_labeling.py 인터페이스.
    반환: [{"primary_emotion", "raw_label", "emotion_intensity", "emotion_probs"}]
    """
    labeler = _get_labeler(model_name)
    return labeler.label_batch(texts)
