import json
from collections import Counter

def map_emotion(e):
    if e in ["anger"]:
        return "분노/비난"
    if e in ["fear"]:
        return "불안/경고"
    if e in ["neutral"]:
        return "중립/조언"
    return "냉소/조롱"

with open("삼성전자 주가_1_comments.json", encoding="utf-8") as f:
    data = json.load(f)

labels = [map_emotion(c["cmt_emotion"]["label"]) for c in data["comments"]]
cnt = Counter(labels)

print(cnt)