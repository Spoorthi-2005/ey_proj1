import os
import pandas as pd
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

from .config import PROCESSED_DIR, RANDOM_STATE


def _load_nlp() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "reviews_nlp.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("Run NLP first to create reviews_nlp.parquet")
    return pd.read_parquet(path)


def run_topics_and_aspects(sample: int | None = None) -> tuple[str, str]:
    df = _load_nlp()
    texts = df["review_text_clean"].tolist()
    if sample and len(texts) > sample:
        texts = texts[:sample]

    topics = None
    # Try BERTopic first; if binary compat errors (numpy/numba/umap/hdbscan), fallback to sklearn pipeline
    try:
        from bertopic import BERTopic
        topic_model = BERTopic(verbose=False)
        topics, _ = topic_model.fit_transform(texts)
    except Exception:
        # Fallback: TF-IDF -> SVD -> KMeans to approximate topic clusters
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
        X = tfidf.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(100, max(2, X.shape[1]-1)), random_state=RANDOM_STATE)
        X_reduced = svd.fit_transform(X)
        k = min(10, max(2, int(len(texts) ** 0.5)))
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        topics = km.fit_predict(X_reduced)

    df_topics = pd.DataFrame({"text": texts, "topic": topics})
    topics_path = os.path.join(PROCESSED_DIR, "reviews_topics.parquet")
    df_topics.to_parquet(topics_path, index=False)

    # Aspect extraction
    kw_model = KeyBERT()
    aspects = []
    for t in texts:
        kws = kw_model.extract_keywords(t, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5)
        aspects.append(", ".join([k for k, s in kws]))
    df_aspects = pd.DataFrame({"text": texts, "aspects": aspects})
    aspects_path = os.path.join(PROCESSED_DIR, "reviews_aspects.parquet")
    df_aspects.to_parquet(aspects_path, index=False)

    return topics_path, aspects_path

if __name__ == "__main__":
    print(run_topics_and_aspects(sample=10000))
