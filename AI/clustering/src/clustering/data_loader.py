from datasets import load_dataset
import pandas as pd

def load_naver_news(sample_size=None):
    dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")
    df = pd.DataFrame(dataset['train'])

    if sample_size:
        df = df.sample(sample_size, random_state=42)

    texts = df['document'].fillna("").tolist()
    return texts[:1000]