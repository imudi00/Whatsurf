from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re

# 개발테스트용 baseline:
# - Detection: is_biased (간단 룰/어휘 기반) => 나중에 DistilBERT 분류기로 대체
# - Recognition: bias_terms 스팬 추출(어휘 매칭) => 나중에 RoBERTa 토큰 분류기로 대체
# - Masking: 스팬을 [MASK]로 치환 => 나중에 BERT MLM으로 대체 후보 생성

DEFAULT_BIAS_LEXICON = [
    "뻔뻔", "거짓말", "무책임", "과도", "최악", "충격", "참담", "막장", "선동", "조작"
]


@dataclass
class LoadedWordsResult:
    is_biased: bool
    bias_terms: List[str]
    masked_text: str


class LoadedWordsPipeline:
    def __init__(self, bias_lexicon: List[str] | None = None):
        self.bias_lexicon = bias_lexicon or DEFAULT_BIAS_LEXICON

    # 1) Bias Detection
    def detect(self, text: str) -> bool:
        hits = sum(1 for w in self.bias_lexicon if w in text)
        return hits > 0

    # 2) Bias Recognition (단어/구 찾기)
    def recognize(self, text: str) -> List[str]:
        terms = []
        for w in self.bias_lexicon:
            if w in text:
                terms.append(w)
        return sorted(list(set(terms)))

    # 3) Bias Masking
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