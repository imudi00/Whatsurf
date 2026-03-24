import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[4]  # C:\Users\임진서\Desktop\AI

AI_PATH = BASE / "label_results" / "삼성전자 주가_1_comments.json"
GOLD_PATH = BASE / "AI" / "labeling" / "gold_dataset" / "comments" / "gold_comments_1.json"

LABEL_MAP = {
    "anger": "anger",
    "악의적인": "anger",
    "구역질 나는": "anger",
    "노여워하는": "anger",
    "짜증내는": "anger",
    "혐오스러운": "anger",
    "한심한": "anger",
    "성가신": "anger",
    "회의적인": "anger",

    "sadness": "sadness",
    "낙담한": "sadness",
    "실망한": "sadness",
    "비통한": "sadness",
    "우울한": "sadness",
    "후회되는": "sadness",
    "눈물이 나는": "sadness",
    "염세적인": "sadness",
    "괴로워하는": "sadness",

    "fear": "fear",
    "두려운": "fear",
    "초조한": "fear",
    "안달하는": "fear",
    "걱정스러운": "fear",
    "조심스러운": "fear",
    "스트레스 받는": "fear",
    "취약한": "fear",
    "당혹스러운": "fear",
    "혼란스러운": "fear",
    "충격 받은": "fear",

    "hurt": "hurt",
    "억울한": "hurt",
    "배신당한": "hurt",
    "버려진": "hurt",
    "희생된": "hurt",
    "죄책감의": "hurt",
    "고립된": "hurt",
    "가난한 불우한": "hurt",
    "마비된": "hurt",

    "joy": "joy",
    "기쁨": "joy",
    "감사하는": "joy",
    "신뢰하는": "joy",
    "편안한": "joy",
    "만족스러운": "joy",
    "흥분": "joy",
    "느긋": "joy",
    "안도": "joy",
    "신이 난": "joy",
    "자신하는": "joy",
    "질투하는": "joy",
}

LABELS = ["anger", "sadness", "fear", "hurt", "joy"]


def normalize_ai_label(label: str) -> str:
    return LABEL_MAP.get(label, "anger")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_confusion(ai_labels: list[str], gold_labels: list[str]) -> dict:
    matrix = {g: {a: 0 for a in LABELS} for g in LABELS}
    for a, g in zip(ai_labels, gold_labels):
        matrix[g][a] += 1
    return matrix


def main():
    ai = load_json(AI_PATH)
    gold = load_json(GOLD_PATH)

    ai_comments = ai["comments"]
    gold_comments = gold["comments"]

    if len(ai_comments) != len(gold_comments):
        print(f"[경고] 댓글 개수 다름: AI={len(ai_comments)}, GOLD={len(gold_comments)}")

    n = min(len(ai_comments), len(gold_comments))

    ai_labels = []
    gold_labels = []
    mismatches = []

    for i in range(n):
        ai_text = ai_comments[i]["cmt_comment"]
        gold_text = gold_comments[i]["cmt_comment"]

        if ai_text != gold_text:
            print(f"[경고] {i+1}번째 댓글 텍스트 불일치")

        ai_label = normalize_ai_label(ai_comments[i]["cmt_emotion"]["label"])
        gold_label = gold_comments[i]["cmt_emotion"]["label"]

        ai_labels.append(ai_label)
        gold_labels.append(gold_label)

        if ai_label != gold_label:
            mismatches.append({
                "comment_id": gold_comments[i].get("comment_id", i + 1),
                "text": gold_text,
                "ai": ai_label,
                "gold": gold_label,
            })

    correct = sum(1 for a, g in zip(ai_labels, gold_labels) if a == g)
    acc = correct / n if n else 0.0

    ai_dist = Counter(ai_labels)
    gold_dist = Counter(gold_labels)
    confusion = build_confusion(ai_labels, gold_labels)

    print("=" * 60)
    print("댓글 감정 비교 결과")
    print("=" * 60)
    print(f"총 댓글 수: {n}")
    print(f"정확도: {acc:.4f} ({correct}/{n})")
    print()

    print("[AI 감정 분포]")
    for label in LABELS:
        print(f"  {label}: {ai_dist[label]}")

    print("\n[GOLD 감정 분포]")
    for label in LABELS:
        print(f"  {label}: {gold_dist[label]}")

    print("\n[Confusion Matrix] (gold -> ai)")
    for g in LABELS:
        print(f"{g}: {confusion[g]}")

    print("\n[불일치 샘플 상위 10개]")
    for row in mismatches[:10]:
        print(f"- id={row['comment_id']} | ai={row['ai']} | gold={row['gold']}")
        print(f"  {row['text'][:120]}")
        print()

if __name__ == "__main__":
    main()