import os
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from typing import Tuple, List, Optional

from .config import PROCESSED_DIR

# Suppress noisy warnings
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Silence specific FutureWarning about clean_up_tokenization_spaces
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\.tokenization_utils_base",
)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "bhadresh-savani/bert-base-go-emotion"


def _load_clean() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "reviews_clean.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("Run ETL first to create reviews_clean.parquet")
    return pd.read_parquet(path)


def _predict_labels_batch(texts: List[str], model_name: str, batch_size: int = 64, max_length: int = 256) -> Tuple[list, list]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    labels: List[int] = []
    scores: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits.detach().cpu().numpy()
            for row in logits:
                probs = softmax(row)
                pred = int(probs.argmax())
                labels.append(pred)
                scores.append(float(probs.max()))
    return labels, scores


def run_nlp(batch_size: int = 64, sample: Optional[int] = None) -> str:
    df = _load_clean()
    if sample is not None and len(df) > sample:
        df = df.head(sample)

    # Sentiment
    sent_labels, sent_scores = _predict_labels_batch(df["review_text_clean"].tolist(), SENTIMENT_MODEL, batch_size=batch_size)
    sent_map = {0: "negative", 1: "neutral", 2: "positive"}
    df["predicted_sentiment"] = [sent_map[i] for i in sent_labels]
    df["sentiment_confidence"] = sent_scores

    # Emotion
    # Use model config to map class ids to labels
    from transformers import AutoConfig
    emo_labels, emo_scores = _predict_labels_batch(df["review_text_clean"].tolist(), EMOTION_MODEL, batch_size=batch_size)
    emo_config = AutoConfig.from_pretrained(EMOTION_MODEL)
    id2label = getattr(emo_config, "id2label", None) or {}
    df["dominant_emotion"] = [id2label.get(i, "neutral") for i in emo_labels]
    df["emotion_confidence"] = emo_scores

    out_path = os.path.join(PROCESSED_DIR, "reviews_nlp.parquet")
    df.to_parquet(out_path, index=False)
    return out_path
