import re

_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+|(?<=다\.)\s+|(?<=다\?)\s+|(?<=다\!)\s+")

def normalize_text(s: str) -> str:
    s = s.replace("\u200b", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def split_sentences_kor(text: str):
    text = normalize_text(text)
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    sents = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 5:
            continue
        sents.append(p)
    return sents

def eojel_count_kor(sentence: str) -> int:
    sentence = normalize_text(sentence)
    return len(sentence.split()) if sentence else 0
