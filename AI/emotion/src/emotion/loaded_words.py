from __future__ import annotations
from dataclasses import dataclass
from typing import List

DEFAULT_BIAS_LEXICON = ["뻔뻔", "거짓말", "무책임", "과도", "최악", "충격", "참담", "막장", "선동", "조작"]

@dataclass
class LoadedWordsResult:
    is_biased: bool
    bias_terms: List[str]
    masked_text: str

class LoadedWordsPipeline:
    def __init__(self, bias_lexicon: List[str] | None = None):
        self.bias_lexicon = bias_lexicon or DEFAULT_BIAS_LEXICON

    def detect(self, text: str) -> bool:
        return any(w in text for w in self.bias_lexicon)

    def recognize(self, text: str) -> List[str]:
        return sorted(list({w for w in self.bias_lexicon if w in text}))

    def mask(self, text: str, terms: List[str]) -> str:
        masked = text
        for t in terms:
            masked = masked.replace(t, "[MASK]")
        return masked

    def run(self, text: str) -> LoadedWordsResult:
        is_biased = self.detect(text)
        terms = self.recognize(text) if is_biased else []
        masked = self.mask(text, terms) if terms else text
        return LoadedWordsResult(is_biased=is_biased, bias_terms=terms, masked_text=masked)
