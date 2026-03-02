from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

@dataclass
class BiasVectorOutput:
    tone_dem: float
    tone_rep: float
    lr_score: float
    rationale: Optional[str] = None

class DualAxisRegressor(nn.Module):
    def __init__(self, base_model: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)

class BiasVectorService:
    def __init__(self, base_model: str = "klue/bert-base", device: Optional[str] = None):
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualAxisRegressor(base_model).to(self.device)

    def train_simple(self, texts, tone_dem, tone_rep, epochs: int = 1, lr: float = 2e-5, batch_size: int = 2, max_len: int = 128):
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()

        n = len(texts)
        idxs = np.arange(n)
        y = np.stack([tone_dem, tone_rep], axis=1).astype(np.float32)

        for ep in range(1, epochs + 1):
            np.random.shuffle(idxs)
            total, steps = 0.0, 0
            for start in range(0, n, batch_size):
                b = idxs[start:start+batch_size]
                batch_texts = [texts[i] for i in b]
                batch_y = torch.tensor(y[b], dtype=torch.float32, device=self.device)

                enc = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device)

                optim.zero_grad()
                pred = self.model(input_ids, attn)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optim.step()

                total += float(loss.item())
                steps += 1

            print(f"[bias_vector] epoch={ep} loss={total/max(1,steps):.4f}")

    @torch.no_grad()
    def predict(self, text: str, rationale: Optional[str] = None) -> BiasVectorOutput:
        self.model.eval()
        enc = self.tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        pred = self.model(input_ids, attn)[0].detach().cpu().numpy()
        tone_dem, tone_rep = float(pred[0]), float(pred[1])
        return BiasVectorOutput(tone_dem=tone_dem, tone_rep=tone_rep, lr_score=float(tone_rep - tone_dem), rationale=rationale)
