# emotion_model.py
"""
BERT 기반 감정 분류 모델 (9클래스)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class EmotionOutput:
    label: str
    probs: Dict[str, float]
    intensity: float


# ──────────────────────────────────────────────
# 모델 정의
# ──────────────────────────────────────────────

class BertEmotionClassifier(nn.Module):
    def __init__(self, base_model: str, label_list: List[str]):
        super().__init__()
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, self.num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)


# ──────────────────────────────────────────────
# 서비스 클래스
# ──────────────────────────────────────────────

DEFAULT_LABELS = [
    "anger",        # 분노
    "disgust",      # 역겨움
    "fear",         # 두려움
    "anticipation", # 기대
    "sadness",      # 슬픔
    "surprise",     # 놀람
    "prediction",   # 예측
    "trust",        # 믿음
    "neutral",      # 중립
]


class EmotionService:
    def __init__(
        self,
        base_model: str = "klue/bert-base",
        label_list: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.base_model = base_model
        self.label_list = label_list or DEFAULT_LABELS
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertEmotionClassifier(base_model, self.label_list).to(self.device)

    # ── 학습 ──────────────────────────────────

    def train(
        self,
        texts: List[str],
        labels: List[str],
        epochs: int = 1,
        lr: float = 2e-5,
        batch_size: int = 2,
        max_len: int = 128,
    ) -> None:
        """레이블 데이터로 파인튜닝"""
        label_to_id = {l: i for i, l in enumerate(self.label_list)}
        y = [label_to_id[l] for l in labels]

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        n = len(texts)
        idxs = np.arange(n)

        for ep in range(1, epochs + 1):
            np.random.shuffle(idxs)
            total_loss, steps = 0.0, 0

            for start in range(0, n, batch_size):
                b = idxs[start : start + batch_size]
                batch_texts = [texts[i] for i in b]
                batch_y = torch.tensor([y[i] for i in b], dtype=torch.long, device=self.device)

                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids, attn)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                steps += 1

            print(f"[emotion] epoch={ep}  loss={total_loss / max(1, steps):.4f}")

    # ── 추론 ──────────────────────────────────

    @torch.no_grad()
    def predict(self, text: str, max_len: int = 256) -> EmotionOutput:
        """단일 텍스트 감정 예측"""
        self.model.eval()
        enc = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        logits = self.model(input_ids, attn)[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        probs_dict = {lab: float(probs[i]) for i, lab in enumerate(self.label_list)}
        best_i = int(np.argmax(probs))
        return EmotionOutput(
            label=self.label_list[best_i],
            probs=probs_dict,
            intensity=float(np.max(probs)),
        )

    @torch.no_grad()
    def predict_batch(self, texts: List[str], max_len: int = 256) -> List[EmotionOutput]:
        """다수 텍스트 일괄 예측"""
        return [self.predict(t, max_len) for t in texts]
