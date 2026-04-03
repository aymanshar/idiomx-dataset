#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
en_08_final_dataset_statistics_v2.py

Generate final dataset statistics for IdiomX v2.

Outputs:
- data/stats/idiomx_dataset_stats_v2.json
- data/stats/idiomx_dataset_stats_v2.csv
- data/stats/idiomx_dataset_summary_v2.md
- data/stats/validation_status_distribution_v2.png
- data/stats/query_type_distribution_v2.png              (if query_type exists)
- data/stats/source_distribution_v2.png                  (if source exists)
- data/stats/example_length_distribution_v2.png          (if text column exists)
- data/stats/idioms_per_source_v2.csv                    (if source exists)
- data/stats/examples_per_idiom_v2.csv                   (top counts table)

Usage:
    python en_08_final_dataset_statistics_v2.py

Optional:
    python en_08_final_dataset_statistics_v2.py --input path/to/file.csv --output_dir path/to/output
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Helpers
# ============================================================

def safe_div(num: float, den: float) -> float:
    return round(num / den, 6) if den else 0.0


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def count_non_null(df: pd.DataFrame, col: Optional[str]) -> int:
    if not col:
        return 0
    return int(df[col].notna().sum())


def count_non_empty_string(df: pd.DataFrame, col: Optional[str]) -> int:
    if not col:
        return 0
    s = df[col].fillna("").astype(str).str.strip()
    return int((s != "").sum())


def unique_non_empty(df: pd.DataFrame, col: Optional[str]) -> int:
    if not col:
        return 0
    s = df[col].fillna("").astype(str).str.strip()
    s = s[s != ""]
    return int(s.nunique())


def text_lengths(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    return s.str.len()


def word_counts(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    return s.apply(lambda x: len(re.findall(r"\S+", x)))


def series_value_counts(series: pd.Series) -> Dict[str, int]:
    vc = series.fillna("MISSING").astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_bar_chart(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    rotation: int = 0,
    top_n: Optional[int] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = series.copy()
    if top_n is not None:
        data = data.head(top_n)

    plt.figure(figsize=(10, 6))
    ax = data.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height):,}",
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    bins: int = 50,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = values.dropna()
    if clean.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.hist(clean, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def summarize_numeric(series: pd.Series) -> Dict[str, Any]:
    clean = series.dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": 0,
            "std": 0,
            "min": 0,
            "p25": 0,
            "median": 0,
            "p75": 0,
            "max": 0,
        }

    return {
        "count": int(clean.count()),
        "mean": round(float(clean.mean()), 4),
        "std": round(float(clean.std()), 4) if clean.count() > 1 else 0.0,
        "min": round(float(clean.min()), 4),
        "p25": round(float(clean.quantile(0.25)), 4),
        "median": round(float(clean.median()), 4),
        "p75": round(float(clean.quantile(0.75)), 4),
        "max": round(float(clean.max()), 4),
    }


# ============================================================
# Main statistics
# ============================================================

def build_stats(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Optional[str]]]:
    # Try to resolve important columns from likely names
    col_idiom = first_existing(df, [
        "idiom_canonical",
        "target_idiom",
        "idiom_text",
        "idiom",
    ])

    col_example = first_existing(df, [
        "idiom_in_example",
        "example",
        "input_text",
        "query_text",
        "context_sentence",
    ])

    col_status = first_existing(df, [
        "validation_status",
        "status",
    ])

    col_generated = first_existing(df, [
        "is_generated_example",
        "generated",
    ])

    col_source = first_existing(df, [
        "source",
        "data_source",
    ])

    col_query_type = first_existing(df, [
        "query_type",
    ])

    col_meaning_en = first_existing(df, [
        "idiom_in_example_meaning_en",
        "meaning_en",
        "idiom_canonical_meaning",
        "idiom_canonical_meaning_en",
    ])

    col_meaning_ar = first_existing(df, [
        "idiom_in_example_meaning_arabic",
        "meaning_ar",
        "idiom_canonical_meaning_arabic",
        "idiom_canonical_meaning_ar",
    ])

    col_surface = first_existing(df, [
        "idiom_surface",
    ])

    col_pos = first_existing(df, [
        "pos",
        "part_of_speech",
    ])

    col_label = first_existing(df, [
        "example_usage_label",
        "label",
        "label_text",
        "is_example_idiom",
    ])

    col_ambiguity = first_existing(df, [
        "ambiguity_flag",
        "ambiguity",
    ])

    col_compositionality = first_existing(df, [
        "compositionality",
    ])

    col_domain = first_existing(df, [
        "domain",
        "idiom_domain",
    ])

    col_difficulty = first_existing(df, [
        "learner_difficulty",
        "difficulty",
    ])

    resolved = {
        "idiom": col_idiom,
        "example": col_example,
        "validation_status": col_status,
        "is_generated_example": col_generated,
        "source": col_source,
        "query_type": col_query_type,
        "meaning_en": col_meaning_en,
        "meaning_ar": col_meaning_ar,
        "surface": col_surface,
        "pos": col_pos,
        "label": col_label,
        "ambiguity": col_ambiguity,
        "compositionality": col_compositionality,
        "domain": col_domain,
        "difficulty": col_difficulty,
    }

    stats: Dict[str, Any] = {}

    # Core counts
    total_rows = int(len(df))
    stats["total_rows"] = total_rows
    stats["n_columns"] = int(df.shape[1])
    stats["columns"] = list(df.columns)

    stats["unique_idioms"] = unique_non_empty(df, col_idiom)
    stats["non_empty_examples"] = count_non_empty_string(df, col_example)
    stats["non_empty_meaning_en"] = count_non_empty_string(df, col_meaning_en)
    stats["non_empty_meaning_ar"] = count_non_empty_string(df, col_meaning_ar)
    stats["non_empty_surface_forms"] = count_non_empty_string(df, col_surface)

    # Generated ratio
    if col_generated:
        gen_series = df[col_generated]
        if str(gen_series.dtype).lower() in {"bool"}:
            generated_count = int(gen_series.fillna(False).sum())
        else:
            normalized = gen_series.fillna("").astype(str).str.strip().str.lower()
            generated_count = int(normalized.isin({"1", "true", "yes", "generated"}).sum())
        stats["generated_examples_count"] = generated_count
        stats["generated_examples_ratio"] = safe_div(generated_count, total_rows)
    else:
        stats["generated_examples_count"] = None
        stats["generated_examples_ratio"] = None

    # Validation status
    if col_status:
        vc = df[col_status].fillna("MISSING").astype(str).value_counts(dropna=False)
        stats["validation_status_distribution"] = {str(k): int(v) for k, v in vc.items()}
        corrected_count = int(vc.get("corrected", 0))
        stats["correction_rate"] = safe_div(corrected_count, total_rows)
    else:
        stats["validation_status_distribution"] = {}
        stats["correction_rate"] = None

    # Source
    if col_source:
        stats["source_distribution"] = series_value_counts(df[col_source])

    # Query type
    if col_query_type:
        stats["query_type_distribution"] = series_value_counts(df[col_query_type])

    # Label distribution
    if col_label:
        stats["label_distribution"] = series_value_counts(df[col_label])

    # Linguistic metadata
    if col_ambiguity:
        stats["ambiguity_distribution"] = series_value_counts(df[col_ambiguity])
    if col_compositionality:
        stats["compositionality_distribution"] = series_value_counts(df[col_compositionality])
    if col_domain:
        stats["domain_distribution"] = series_value_counts(df[col_domain])
    if col_difficulty:
        stats["difficulty_distribution"] = series_value_counts(df[col_difficulty])
    if col_pos:
        stats["pos_distribution"] = series_value_counts(df[col_pos])

    # Example length stats
    if col_example:
        char_len = text_lengths(df[col_example])
        word_len = word_counts(df[col_example])
        stats["example_length_char_stats"] = summarize_numeric(char_len)
        stats["example_length_word_stats"] = summarize_numeric(word_len)
    else:
        stats["example_length_char_stats"] = {}
        stats["example_length_word_stats"] = {}

    # Examples per idiom
    if col_idiom:
        non_empty_idiom = df[col_idiom].fillna("").astype(str).str.strip()
        idiom_counts = non_empty_idiom[non_empty_idiom != ""].value_counts()
        if not idiom_counts.empty:
            stats["examples_per_idiom_stats"] = summarize_numeric(idiom_counts)
            stats["top_20_idioms_by_examples"] = [
                {"idiom": str(idx), "count": int(val)}
                for idx, val in idiom_counts.head(20).items()
            ]
        else:
            stats["examples_per_idiom_stats"] = {}
            stats["top_20_idioms_by_examples"] = []
    else:
        stats["examples_per_idiom_stats"] = {}
        stats["top_20_idioms_by_examples"] = []

    # Missingness
    missing_summary = {}
    for key, col in resolved.items():
        if col:
            missing_summary[col] = {
                "missing_count": int(df[col].isna().sum()),
                "missing_ratio": safe_div(int(df[col].isna().sum()), total_rows),
            }
    stats["missing_summary"] = missing_summary

    return stats, resolved


def stats_to_dataframe(stats: Dict[str, Any]) -> pd.DataFrame:
    flat_rows: List[Dict[str, Any]] = []

    def add_row(section: str, metric: str, value: Any) -> None:
        flat_rows.append({
            "section": section,
            "metric": metric,
            "value": value,
        })

    core_keys = [
        "total_rows",
        "n_columns",
        "unique_idioms",
        "non_empty_examples",
        "non_empty_meaning_en",
        "non_empty_meaning_ar",
        "non_empty_surface_forms",
        "generated_examples_count",
        "generated_examples_ratio",
        "correction_rate",
    ]
    for k in core_keys:
        add_row("core", k, stats.get(k))

    for group_key in [
        "validation_status_distribution",
        "source_distribution",
        "query_type_distribution",
        "label_distribution",
        "ambiguity_distribution",
        "compositionality_distribution",
        "domain_distribution",
        "difficulty_distribution",
        "pos_distribution",
    ]:
        group = stats.get(group_key, {})
        if isinstance(group, dict):
            for sub_k, sub_v in group.items():
                add_row(group_key, sub_k, sub_v)

    for group_key in [
        "example_length_char_stats",
        "example_length_word_stats",
        "examples_per_idiom_stats",
    ]:
        group = stats.get(group_key, {})
        if isinstance(group, dict):
            for sub_k, sub_v in group.items():
                add_row(group_key, sub_k, sub_v)

    return pd.DataFrame(flat_rows)


def build_markdown_summary(stats: Dict[str, Any], input_path: Path) -> str:
    lines: List[str] = []
    lines.append("# IdiomX Final Dataset Statistics v2")
    lines.append("")
    lines.append(f"- **Input file:** `{input_path}`")
    lines.append(f"- **Total rows:** {stats.get('total_rows', 0):,}")
    lines.append(f"- **Unique idioms:** {stats.get('unique_idioms', 0):,}")
    lines.append(f"- **Non-empty examples:** {stats.get('non_empty_examples', 0):,}")
    lines.append(f"- **English meaning coverage:** {stats.get('non_empty_meaning_en', 0):,}")
    lines.append(f"- **Arabic meaning coverage:** {stats.get('non_empty_meaning_ar', 0):,}")
    lines.append("")

    if stats.get("generated_examples_count") is not None:
        lines.append("## Generated Examples")
        lines.append("")
        lines.append(f"- **Generated examples count:** {stats['generated_examples_count']:,}")
        lines.append(f"- **Generated examples ratio:** {stats['generated_examples_ratio']:.2%}")
        lines.append("")

    val_dist = stats.get("validation_status_distribution", {})
    if val_dist:
        lines.append("## Validation Status Distribution")
        lines.append("")
        for k, v in val_dist.items():
            lines.append(f"- **{k}:** {v:,}")
        correction_rate = stats.get("correction_rate")
        if correction_rate is not None:
            lines.append(f"- **Correction rate:** {correction_rate:.2%}")
        lines.append("")

    lines.append("## Example Length Statistics (Words)")
    lines.append("")
    word_stats = stats.get("example_length_word_stats", {})
    if word_stats:
        for k in ["count", "mean", "std", "min", "p25", "median", "p75", "max"]:
            lines.append(f"- **{k}:** {word_stats.get(k)}")
    lines.append("")

    epi = stats.get("examples_per_idiom_stats", {})
    if epi:
        lines.append("## Examples per Idiom")
        lines.append("")
        for k in ["count", "mean", "std", "min", "p25", "median", "p75", "max"]:
            lines.append(f"- **{k}:** {epi.get(k)}")
        lines.append("")

    for section_name, title in [
        ("ambiguity_distribution", "Ambiguity Distribution"),
        ("compositionality_distribution", "Compositionality Distribution"),
        ("difficulty_distribution", "Difficulty Distribution"),
        ("domain_distribution", "Domain Distribution"),
        ("source_distribution", "Source Distribution"),
        ("query_type_distribution", "Query Type Distribution"),
        ("label_distribution", "Label Distribution"),
    ]:
        group = stats.get(section_name, {})
        if isinstance(group, dict) and group:
            lines.append(f"## {title}")
            lines.append("")
            for k, v in group.items():
                lines.append(f"- **{k}:** {v:,}")
            lines.append("")

    top_idioms = stats.get("top_20_idioms_by_examples", [])
    if top_idioms:
        lines.append("## Top 20 Idioms by Number of Examples")
        lines.append("")
        lines.append("| Idiom | Count |")
        lines.append("|---|---:|")
        for row in top_idioms:
            lines.append(f"| {row['idiom']} | {row['count']:,} |")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final dataset statistics for IdiomX v2.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/enriched/idiomx_enriched_full_final_v2.csv",
        help="Path to final v2 CSV dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/stats",
        help="Directory where statistics and charts will be saved.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    print("Building statistics...")
    stats, resolved = build_stats(df)

    # Save JSON
    json_path = output_dir / "idiomx_dataset_stats_v2.json"
    save_json(json_path, stats)

    # Save flat CSV
    csv_path = output_dir / "idiomx_dataset_stats_v2.csv"
    stats_df = stats_to_dataframe(stats)
    stats_df.to_csv(csv_path, index=False, encoding="utf-8")

    # Save markdown summary
    md_path = output_dir / "idiomx_dataset_summary_v2.md"
    md_text = build_markdown_summary(stats, input_path)
    save_markdown(md_path, md_text)

    # Extra CSV: examples per idiom
    col_idiom = resolved["idiom"]
    if col_idiom:
        idiom_counts = (
            df[col_idiom]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        idiom_counts = idiom_counts[idiom_counts != ""].value_counts().reset_index()
        idiom_counts.columns = ["idiom", "count"]
        idiom_counts.to_csv(
            output_dir / "examples_per_idiom_v2.csv",
            index=False,
            encoding="utf-8",
        )

    # Extra CSV: idioms per source
    col_source = resolved["source"]
    if col_idiom and col_source:
        idioms_per_source = (
            df[[col_idiom, col_source]]
            .copy()
            .dropna(subset=[col_idiom, col_source])
        )
        idioms_per_source[col_idiom] = idioms_per_source[col_idiom].astype(str).str.strip()
        idioms_per_source[col_source] = idioms_per_source[col_source].astype(str).str.strip()
        idioms_per_source = idioms_per_source[
            (idioms_per_source[col_idiom] != "") & (idioms_per_source[col_source] != "")
        ]
        idioms_per_source = (
            idioms_per_source.groupby(col_source)[col_idiom]
            .nunique()
            .sort_values(ascending=False)
            .reset_index()
        )
        idioms_per_source.columns = ["source", "unique_idioms"]
        idioms_per_source.to_csv(
            output_dir / "idioms_per_source_v2.csv",
            index=False,
            encoding="utf-8",
        )

    # Charts
    col_example = resolved["example"]
    col_status = resolved["validation_status"]
    col_query_type = resolved["query_type"]

    if col_status:
        vc = df[col_status].fillna("MISSING").astype(str).value_counts(dropna=False)
        make_bar_chart(
            vc,
            title="Validation Status Distribution",
            xlabel="Validation Status",
            ylabel="Count",
            out_path=output_dir / "validation_status_distribution_v2.png",
            rotation=0,
        )

    if col_query_type:
        vc = df[col_query_type].fillna("MISSING").astype(str).value_counts(dropna=False)
        make_bar_chart(
            vc,
            title="Query Type Distribution",
            xlabel="Query Type",
            ylabel="Count",
            out_path=output_dir / "query_type_distribution_v2.png",
            rotation=0,
        )

    if col_source:
        vc = df[col_source].fillna("MISSING").astype(str).value_counts(dropna=False)
        make_bar_chart(
            vc,
            title="Source Distribution",
            xlabel="Source",
            ylabel="Count",
            out_path=output_dir / "source_distribution_v2.png",
            rotation=30,
        )

    if col_example:
        char_len = text_lengths(df[col_example])
        make_histogram(
            char_len,
            title="Example Length Distribution (Characters)",
            xlabel="Length (characters)",
            ylabel="Count",
            out_path=output_dir / "example_length_distribution_v2.png",
            bins=50,
        )

        word_len = word_counts(df[col_example])
        make_histogram(
            word_len,
            title="Example Length Distribution (Words)",
            xlabel="Length (words)",
            ylabel="Count",
            out_path=output_dir / "example_word_count_distribution_v2.png",
            bins=50,
        )

    # Console summary
    print("\nResolved columns:")
    for k, v in resolved.items():
        print(f"  {k:20s}: {v}")

    print("\nCore summary:")
    print(f"  Total rows           : {stats.get('total_rows', 0):,}")
    print(f"  Unique idioms        : {stats.get('unique_idioms', 0):,}")
    print(f"  Non-empty examples   : {stats.get('non_empty_examples', 0):,}")
    print(f"  English meanings     : {stats.get('non_empty_meaning_en', 0):,}")
    print(f"  Arabic meanings      : {stats.get('non_empty_meaning_ar', 0):,}")

    if stats.get("generated_examples_count") is not None:
        print(f"  Generated examples   : {stats['generated_examples_count']:,}")
        print(f"  Generated ratio      : {stats['generated_examples_ratio']:.2%}")

    val_dist = stats.get("validation_status_distribution", {})
    if val_dist:
        print("\nValidation summary:")
        for k, v in val_dist.items():
            print(f"  {k:20s}: {v:,}")
        if stats.get("correction_rate") is not None:
            print(f"  Correction rate      : {stats['correction_rate']:.2%}")

    print("\nSaved outputs:")
    print(f"  JSON     : {json_path}")
    print(f"  CSV      : {csv_path}")
    print(f"  Markdown : {md_path}")
    print(f"  Folder   : {output_dir}")


if __name__ == "__main__":
    main()