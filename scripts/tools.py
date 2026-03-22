"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

License:
MIT License (see LICENSE file)

Citation:
If you use this code or dataset, please cite the IdiomX paper.
"""
from pathlib import Path
from typing import Any, Iterable, Optional
import json
import re
import pandas as pd

# Shared regex patterns

MULTISPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\w+")
LETTER_RE = re.compile(r"[A-Za-z]")


# Basic text helpers

def safe_str(x: Any) -> str:
    """
    Convert null-like values to an empty string and return stripped text.
    """
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_text(text: Any) -> str:
    """
    Normalize text by converting null-like values to an empty string,
    trimming whitespace, and collapsing repeated internal spaces.
    """
    text = safe_str(text)
    return MULTISPACE_RE.sub(" ", text)


def lower_normalized(text: Any) -> str:
    """
    Normalize text and lowercase it.
    """
    return normalize_text(text).lower()


def tokenize_words(text: Any) -> set[str]:
    """
    Tokenize text into a set of lowercase word tokens.
    Useful for approximate matching and validation checks.
    """
    return set(WORD_RE.findall(lower_normalized(text)))


def token_count(text: Any) -> int:
    """
    Count whitespace-separated tokens in a text.
    """
    return len(normalize_text(text).split())


def has_letters(text: Any) -> bool:
    """
    Return True if the text contains alphabetic characters.
    """
    return bool(LETTER_RE.search(normalize_text(text)))


# File / path helpers

def ensure_parent_dir(path: Path | str) -> Path:
    """
    Ensure the parent directory of a file path exists.
    Returns the Path object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path | str) -> Path:
    """
    Ensure a directory exists.
    Returns the Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_exists_or_raise(path: Path | str, label: str = "File") -> Path:
    """
    Validate that a file exists, otherwise raise FileNotFoundError.
    Returns the Path object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def get_mode_value(use_sample: bool, full_value: Any, sample_value: Any) -> Any:
    """
    Return a full-mode or sample-mode value based on the use_sample flag.
    """
    return sample_value if use_sample else full_value


# JSON helpers

def load_json(path: Path | str) -> dict:
    """
    Load a JSON file into a dictionary.
    """
    path = file_exists_or_raise(path, "JSON file")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path | str, ensure_ascii: bool = False, indent: int = 2) -> Path:
    """
    Save a Python object as JSON.
    """
    path = ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
    return path


def extract_json_object(text: str) -> Optional[dict]:
    """
    Extract a JSON object from raw text.
    Returns None if parsing fails.
    """
    text = safe_str(text)
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def extract_json_array(text: str) -> list:
    """
    Extract a JSON array from raw text.
    Returns an empty list if parsing fails.
    """
    text = safe_str(text)
    if not text:
        return []

    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass

    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                return arr
        except Exception:
            return []

    return []


# DataFrame helpers

def ensure_columns(df: pd.DataFrame, columns: Iterable[str], fill_value: str = "") -> pd.DataFrame:
    """
    Ensure that all listed columns exist in the dataframe.
    Missing columns are created with fill_value.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df


def normalize_dataframe_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Normalize selected dataframe columns by:
    - creating missing columns
    - filling nulls
    - converting to string
    - trimming whitespace
    """
    df = ensure_columns(df, columns)
    df = df.copy()

    for col in columns:
        df[col] = df[col].fillna("").astype(str).map(normalize_text)

    return df


def build_dedup_key(*parts: Any, sep: str = " || ") -> str:
    """
    Build a normalized lowercase deduplication key from one or more text parts.
    """
    return sep.join(lower_normalized(p) for p in parts)


def deduplicate_on_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop exact duplicates based on the given columns.
    """
    return df.drop_duplicates(subset=columns).reset_index(drop=True)


def deduplicate_idiom_meaning(
    df: pd.DataFrame,
    idiom_col: str = "idiom",
    meaning_col: str = "meaning_en",
    key_col: str = "dedup_key",
) -> pd.DataFrame:
    """
    Remove duplicates based on normalized idiom + meaning pairs.
    """
    df = df.copy()
    df[key_col] = df.apply(
        lambda r: build_dedup_key(r.get(idiom_col, ""), r.get(meaning_col, "")),
        axis=1
    )
    df = df.drop_duplicates(subset=[key_col]).drop(columns=[key_col]).reset_index(drop=True)
    return df


def value_counts_dict(series: pd.Series) -> dict:
    """
    Convert pandas value_counts output into a plain Python dictionary.
    """
    return {str(k): int(v) for k, v in series.value_counts().to_dict().items()}


# CSV / parquet convenience helpers

def read_csv_utf8(path: Path | str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file using UTF-8-SIG encoding by default.
    """
    path = file_exists_or_raise(path, "CSV file")
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def write_csv_utf8(df: pd.DataFrame, path: Path | str, index: bool = False, **kwargs) -> Path:
    """
    Write a dataframe to CSV using UTF-8-SIG encoding by default.
    """
    path = ensure_parent_dir(path)
    df.to_csv(path, index=index, encoding="utf-8-sig", **kwargs)
    return path


def read_parquet_safe(path: Path | str, **kwargs) -> pd.DataFrame:
    """
    Read a parquet file with existence checking.
    """
    path = file_exists_or_raise(path, "Parquet file")
    return pd.read_parquet(path, **kwargs)


def write_parquet_safe(df: pd.DataFrame, path: Path | str, index: bool = False, **kwargs) -> Path:
    """
    Write a dataframe to parquet with parent directory creation.
    """
    path = ensure_parent_dir(path)
    df.to_parquet(path, index=index, **kwargs)
    return path


# Batch / response helpers

def extract_output_text(body: dict) -> Optional[str]:
    """
    Extract the first output_text content from a batch response body.
    Returns None if unavailable.
    """
    try:
        outputs = body.get("output", [])
        for item in outputs:
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text")
    except Exception:
        return None
    return None


def load_batch_id(batch_info_file: Path | str) -> str:
    """
    Load batch_id from a stored batch info JSON file.
    """
    info = load_json(batch_info_file)
    batch_id = info.get("batch_id")
    if not batch_id:
        raise ValueError(f"'batch_id' not found in batch info file: {batch_info_file}")
    return str(batch_id)

# Export control

__all__ = [
    "safe_str",
    "normalize_text",
    "lower_normalized",
    "tokenize_words",
    "token_count",
    "has_letters",
    "ensure_parent_dir",
    "ensure_dir",
    "file_exists_or_raise",
    "get_mode_value",
    "load_json",
    "save_json",
    "extract_json_object",
    "extract_json_array",
    "ensure_columns",
    "normalize_dataframe_columns",
    "build_dedup_key",
    "deduplicate_on_columns",
    "deduplicate_idiom_meaning",
    "value_counts_dict",
    "read_csv_utf8",
    "write_csv_utf8",
    "read_parquet_safe",
    "write_parquet_safe",
    "extract_output_text",
    "load_batch_id",
]