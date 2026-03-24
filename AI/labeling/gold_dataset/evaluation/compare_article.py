#JSON 비교 간단 툴
import json

# AI 결과
with open("삼성전자 주가_1_labeled.json", encoding="utf-8") as f:
    ai = json.load(f)

# GOLD 결과
with open("gold_article_1.json", encoding="utf-8") as f:
    gold = json.load(f)

# frame 비교
print("frame:", ai.get("frame"), "vs", gold["gold_article"]["frame"])

# stance loss
loss = abs(ai.get("stance_score", 0) - gold["gold_article"]["stance_score"])
print("stance loss:", loss)

# bias loss
bias_loss = abs(ai.get("bias_x", 0) - gold["gold_article"]["bias_x"])
print("bias loss:", bias_loss)