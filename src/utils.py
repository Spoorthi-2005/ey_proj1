import re
import pandas as pd
from langdetect import detect, LangDetectException

EMOJI_PATTERN = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def detect_language_safe(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def basic_text_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = HTML_TAG_PATTERN.sub(" ", text)
    t = re.sub(r"https?://\S+", " ", t)
    t = EMOJI_PATTERN.sub(" ", t)
    t = re.sub(r"[^\w\s\-\.,!?'’]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
