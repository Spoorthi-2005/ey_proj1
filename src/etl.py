import pandas as pd
from symspellpy import SymSpell, Verbosity
import pkgutil
import os
from typing import Optional

from .config import RAW_DATA, PROCESSED_DIR, MAX_ROWS
from .utils import basic_text_clean, detect_language_safe, ensure_datetime


def load_raw(limit: Optional[int] = None) -> pd.DataFrame:
    nrows = min(limit or MAX_ROWS, MAX_ROWS)
    df = pd.read_csv(RAW_DATA)
    if nrows and len(df) > nrows:
        df = df.sample(nrows, random_state=42)
    return df


def build_symspell() -> Optional[SymSpell]:
    try:
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        # Use a small frequency dictionary if available
        path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "frequency_dictionary_en_82_765.txt")
        if os.path.exists(path):
            sym_spell.load_dictionary(path, term_index=0, count_index=1)
        else:
            return None
        return sym_spell
    except Exception:
        return None


def correct_spelling(text: str, sym_spell: Optional[SymSpell]) -> str:
    if not sym_spell:
        return text
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text


def clean_and_normalize(df: pd.DataFrame, detect_lang: bool = True) -> pd.DataFrame:
    df = df.copy()
    # Required columns fallback
    if "review_text" not in df.columns:
        # Try to infer
        for c in df.columns:
            if "review" in c.lower() or "text" in c.lower():
                df = df.rename(columns={c: "review_text"})
                break
    if "rating" not in df.columns:
        for c in df.columns:
            if "rating" in c.lower() or c.lower() in {"stars", "score"}:
                df = df.rename(columns={c: "rating"})
                break
    # Clean
    df["review_text"] = df["review_text"].map(basic_text_clean)
    # Language detection (optional for speed)
    if detect_lang:
        df["lang"] = df["review_text"].map(detect_language_safe)
    else:
        df["lang"] = "unknown"
    # Spell correction (best-effort)
    sym = build_symspell()
    df["review_text_clean"] = df["review_text"].map(lambda x: correct_spelling(x, sym))
    # Timestamps (recognize common names incl. review_date)
    for c in ["timestamp", "created_at", "date_time", "date", "review_date"]:
        if c in df.columns:
            df = ensure_datetime(df, c)
            df = df.rename(columns={c: "timestamp"})
            break
    # Geo location composed if not present
    if "geo_location" not in df.columns:
        country = df["country"].astype(str) if "country" in df.columns else None
        region = df["region"].astype(str) if "region" in df.columns else None
        if country is not None or region is not None:
            if country is None:
                df["geo_location"] = region
            elif region is None:
                df["geo_location"] = country
            else:
                df["geo_location"] = (country.fillna("") + ", " + region.fillna("")).str.strip(", ")
    # Deduplicate
    keep_cols = [c for c in ["user_id", "review_text_clean", "rating", "product_category", "geo_location", "timestamp", "verified_purchase", "source", "lang"] if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["user_id", "review_text_clean"], keep="first")
    return df


def run_etl(limit: Optional[int] = None, detect_lang: bool = True) -> str:
    df = load_raw(limit)
    df = clean_and_normalize(df, detect_lang=detect_lang)
    # Prefer parquet; fallback to CSV if engine missing
    out_parquet = os.path.join(PROCESSED_DIR, "reviews_clean.parquet")
    try:
        df.to_parquet(out_parquet, index=False)
        print(f"Saved Parquet: {out_parquet}")
        return out_parquet
    except Exception as e:
        out_csv = os.path.join(PROCESSED_DIR, "reviews_clean.csv")
        df.to_csv(out_csv, index=False)
        print(f"Parquet failed ({e}); saved CSV: {out_csv}")
        return out_csv

if __name__ == "__main__":
    p = run_etl()
    print(f"Saved cleaned data to: {p}")
