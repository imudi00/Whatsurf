from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

# NOTE:
# - BERT 논문 취지: pretrained encoder + task head fine-tune.
# - 여기서는 KLUE-BERT encoder 위에 분류 head를 올려 학습/추론.


@dataclass
class EmotionsOutput:
    label: str
    probs: Dict[str, float]
    intensity: float  # e.g., max prob


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
        cls = out.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(cls)
        return logits


class EmotionsService:
    def __init__(
        self,
        base_model: str = "klue/bert-base",
        label_list: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.base_model = base_model
        self.label_list = label_list or ["anger", "anxiety", "fear", "joy", "neutral", "sadness"]
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertEmotionClassifier(base_model, self.label_list).to(self.device)

    def train_simple(
        self,
        texts: List[str],
        labels: List[str],
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 8,
        max_len: int = 128,
    ) -> Dict[str, float]:
        # simple supervised single-label (softmax CE)
        label_to_id = {l: i for i, l in enumerate(self.label_list)}
        y = [label_to_id[l] for l in labels]

        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        n = len(texts)
        idxs = np.arange(n)

        for ep in range(1, epochs + 1):
            np.random.shuffle(idxs)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                batch_idx = idxs[start : start + batch_size]
                batch_texts = [texts[i] for i in batch_idx]
                batch_y = torch.tensor([y[i] for i in batch_idx], dtype=torch.long, device=self.device)

                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device)

                optim.zero_grad()
                logits = self.model(input_ids, attn)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optim.step()

                total_loss += float(loss.item())

            avg_loss = total_loss / max(1, (n // batch_size))
            print(f"[emotions] epoch={ep} loss={avg_loss:.4f}")

        return {"epochs": float(epochs)}

    @torch.no_grad()
    def predict(self, text: str, max_len: int = 256) -> EmotionsOutput:
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
        label = self.label_list[best_i]
        intensity = float(np.max(probs))

        return EmotionsOutput(label=label, probs=probs_dict, intensity=intensity)

    def article_vector(self, sentences: List[str]) -> Dict[str, float]:
        # 문장별 확률 예측 -> 평균 pooling => 기사 감정 벡터
        if not sentences:
            return {lab: 0.0 for lab in self.label_list}

        mats = []
        for s in sentences:
            out = self.predict(s)
            mats.append([out.probs[lab] for lab in self.label_list])

        arr = np.array(mats, dtype=np.float32)
        mean = arr.mean(axis=0)
        return {lab: float(mean[i]) for i, lab in enumerate(self.label_list)}