import json
from pathlib import Path


HF_PATH = Path("./AI/clustering/experiments/llm_results/llm_cluster_summary_hf_kospi.json")
OLLAMA_PATH = Path("./AI/clustering/experiments/llm_results/llm_cluster_summary_ollama_kospi.json")
GEMINI_PATH = Path("./AI/clustering/experiments/llm_results/llm_cluster_summary_gemini_kospi.json")


def load_json(path: Path):
    if not path.exists():
        print(f"[경고] 파일 없음: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_results(data):
    if not data:
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return data.get("results", [])

    return []


def calc_metrics(name: str, data):
    if not data:
        return {
            "model_name": name,
            "file_exists": False,
            "cluster_count": 0,
            "success_count": 0,
            "fail_count": 0,
            "success_rate": 0.0,
            "avg_headline_len": 0.0,
            "avg_summary_len": 0.0,
            "avg_elapsed_sec": 0.0,
            "total_elapsed_sec": 0.0,
        }

    results = get_results(data)
    cluster_count = len(results)

    success_items = [r for r in results if isinstance(r, dict) and r.get("success") is True]
    fail_items = [r for r in results if not (isinstance(r, dict) and r.get("success") is True)]

    success_count = len(success_items)
    fail_count = len(fail_items)
    success_rate = (success_count / cluster_count * 100) if cluster_count > 0 else 0.0

    headline_lens = [
        len(r.get("headline", "").replace("\n", " ").strip())
        for r in success_items
        if r.get("headline", "").strip()
    ]
    summary_lens = [
        len(r.get("summary", "").replace("\n", " ").strip())
        for r in success_items
        if r.get("summary", "").strip()
    ]
    elapsed_list = [
        float(r.get("elapsed_sec", 0.0))
        for r in results
        if isinstance(r, dict)
    ]

    avg_headline_len = sum(headline_lens) / len(headline_lens) if headline_lens else 0.0
    avg_summary_len = sum(summary_lens) / len(summary_lens) if summary_lens else 0.0
    avg_elapsed_sec = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0

    model_name = name
    total_elapsed_sec = 0.0
    if isinstance(data, dict):
        model_name = data.get("model", name)
        total_elapsed_sec = float(data.get("total_elapsed_sec", 0.0))

    return {
        "model_name": model_name,
        "file_exists": True,
        "cluster_count": cluster_count,
        "success_count": success_count,
        "fail_count": fail_count,
        "success_rate": round(success_rate, 2),
        "avg_headline_len": round(avg_headline_len, 2),
        "avg_summary_len": round(avg_summary_len, 2),
        "avg_elapsed_sec": round(avg_elapsed_sec, 4),
        "total_elapsed_sec": round(total_elapsed_sec, 4),
    }


def print_metrics_table(metrics_list):
    print("\n" + "=" * 120)
    print("LLM OUTPUT COMPARISON")
    print("=" * 120)
    print(
        f"{'MODEL':30} "
        f"{'FILE':6} "
        f"{'CLUSTERS':8} "
        f"{'SUCCESS':8} "
        f"{'FAIL':6} "
        f"{'RATE(%)':8} "
        f"{'AVG_HEAD':10} "
        f"{'AVG_SUM':10} "
        f"{'AVG_T(s)':10} "
        f"{'TOTAL_T(s)':10}"
    )
    print("-" * 120)

    for m in metrics_list:
        print(
            f"{m['model_name'][:30]:30} "
            f"{str(m['file_exists']):6} "
            f"{m['cluster_count']:8} "
            f"{m['success_count']:8} "
            f"{m['fail_count']:6} "
            f"{m['success_rate']:8} "
            f"{m['avg_headline_len']:10} "
            f"{m['avg_summary_len']:10} "
            f"{m['avg_elapsed_sec']:10} "
            f"{m['total_elapsed_sec']:10}"
        )

    print("=" * 120)


def print_cluster_previews(label: str, data):
    print(f"\n[{label}]")
    if not data:
        print("파일 없음")
        return

    results = get_results(data)
    if not results:
        print("results 없음")
        return

    for r in results:
        cluster_id = r.get("cluster_id")
        success = r.get("success")
        headline = r.get("headline", "")
        summary = r.get("summary", "")
        error = r.get("error", "")
        elapsed_sec = r.get("elapsed_sec", 0.0)

        print(f"\n- cluster_id: {cluster_id}")
        print(f"  success   : {success}")
        print(f"  headline  : {headline}")
        print(f"  summary   : {summary}")
        print(f"  elapsed   : {elapsed_sec}s")
        if error:
            print(f"  error     : {error}")


def compare_by_cluster(hf_data, ollama_data, gemini_data):
    print("\n" + "=" * 120)
    print("CLUSTER-BY-CLUSTER COMPARISON")
    print("=" * 120)

    sources = {
        "HF": get_results(hf_data),
        "OLLAMA": get_results(ollama_data),
        "GEMINI": get_results(gemini_data),
    }

    cluster_ids = set()
    for items in sources.values():
        for r in items:
            cluster_ids.add(r.get("cluster_id"))

    for cluster_id in sorted(cluster_ids, key=lambda x: (x is None, x)):
        print(f"\n[cluster_id = {cluster_id}]")
        for source_name, items in sources.items():
            matched = next((x for x in items if x.get("cluster_id") == cluster_id), None)

            if not matched:
                print(f"  {source_name:8}: 결과 없음")
                continue

            success = matched.get("success")
            headline = matched.get("headline", "")
            summary = matched.get("summary", "")
            error = matched.get("error", "")
            elapsed_sec = matched.get("elapsed_sec", 0.0)

            print(f"  {source_name:8}: success={success}, elapsed={elapsed_sec}s")
            print(f"             headline={headline}")
            print(f"             summary={summary}")
            if error:
                print(f"             error={error}")


def main():
    hf_data = load_json(HF_PATH)
    ollama_data = load_json(OLLAMA_PATH)
    gemini_data = load_json(GEMINI_PATH)

    metrics_list = [
        calc_metrics("HF", hf_data),
        calc_metrics("OLLAMA", ollama_data),
        calc_metrics("GEMINI", gemini_data),
    ]

    print_metrics_table(metrics_list)

    print_cluster_previews("HF", hf_data)
    print_cluster_previews("OLLAMA", ollama_data)
    print_cluster_previews("GEMINI", gemini_data)

    compare_by_cluster(hf_data, ollama_data, gemini_data)


if __name__ == "__main__":
    main()