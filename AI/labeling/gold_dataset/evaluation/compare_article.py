import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[4]  # C:\Users\임진서\Desktop\AI

AI_PATH = BASE / "label_results" / "삼성전자 주가_1_labeled.json"
GOLD_PATH = BASE / "AI" / "labeling" / "gold_dataset" / "article" / "gold_article_1.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ai = load_json(AI_PATH)
    gold = load_json(GOLD_PATH)

    print("=" * 60)
    print("기사 피처 비교 결과")
    print("=" * 60)

    fields = [
        "stance_score",
        "stance_label",
        "body_depth",
        "body_depth_label",
        "art_words",
        "loaded_word_density",
        "is_biased",
    ]

    for field in fields:
        ai_val = ai.get(field)
        gold_val = gold.get(field)

        if field in ["stance_score", "body_depth", "loaded_word_density"]:
            loss = abs(float(ai_val) - float(gold_val))
            print(f"{field}:")
            print(f"  AI   = {ai_val}")
            print(f"  GOLD = {gold_val}")
            print(f"  L1 loss = {loss:.4f}")
        elif field == "art_words":
            ai_set = set(ai_val or [])
            gold_set = set(gold_val or [])
            inter = len(ai_set & gold_set)
            union = len(ai_set | gold_set)
            jaccard = inter / union if union else 1.0
            print(f"{field}:")
            print(f"  AI   = {ai_val}")
            print(f"  GOLD = {gold_val}")
            print(f"  Jaccard = {jaccard:.4f}")
        else:
            match = ai_val == gold_val
            print(f"{field}:")
            print(f"  AI   = {ai_val}")
            print(f"  GOLD = {gold_val}")
            print(f"  MATCH = {match}")

        print()

if __name__ == "__main__":
    main()