# src 경로 설정
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from src.clustering.data_loader import load_naver_news
from src.clustering.service_pipeline import run_service_clustering

if __name__ == "__main__":
    texts = load_naver_news()
    print(f"Documents: {len(texts)}")

    labels = run_service_clustering(texts)

    print("\nCluster Results:")
    print(labels)