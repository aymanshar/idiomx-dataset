from pathlib import Path
import pandas as pd
from rdflib import Graph, Literal
import re
import requests
import zipfile
import io


LIDIOMS_REPO_ZIP = "https://github.com/dice-group/LIdioms/archive/refs/heads/master.zip"
# Supported RDF-like extensions
RDF_EXTS = {".ttl", ".nt", ".rdf", ".owl", ".n3", ".xml"}

def has_rdf_files(root: Path) -> bool:
    if not root.exists():
        return False
    return any(p.suffix.lower() in RDF_EXTS for p in root.rglob("*"))


def ensure_lidioms_dataset(lidioms_root: Path):
    """
    Ensure the local LIdioms English RDF dataset exists.
    If the target folder is missing or empty, download and extract it.
    """
    lidioms_root = Path(lidioms_root)

    if has_rdf_files(lidioms_root):
        print(f"LIdioms dataset already available: {lidioms_root}")
        return

    raw_dir = lidioms_root.parent.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading LIdioms dataset...")
    r = requests.get(LIDIOMS_REPO_ZIP, timeout=120)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(raw_dir)

    extracted_root = raw_dir / "LIdioms-master"
    final_root = raw_dir / "LIdioms"

    if final_root.exists() and not has_rdf_files(final_root):
        import shutil
        shutil.rmtree(final_root)

    if extracted_root.exists() and not final_root.exists():
        extracted_root.rename(final_root)

    if not has_rdf_files(lidioms_root):
        raise FileNotFoundError(
            f"LIdioms download completed, but no RDF files were found under: {lidioms_root}"
        )

    print(f"LIdioms dataset ready: {lidioms_root}")
# ============================================================
# Default project paths
# ============================================================

BASE_DIR = Path("..")
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESS_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

# Expected local LIdioms folder
DEFAULT_LIDIOMS_ROOT = DATA_RAW_DIR / "LIdioms" / "en"

# Output normalized CSV
DEFAULT_OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_source_lidioms_normalized.csv"

# Predicate hints used to detect relevant fields in RDF
LABEL_HINTS = {"label", "prefLabel", "writtenRep", "canonicalForm"}
DEF_HINTS = {"definition", "senseDefinition", "gloss", "note", "description"}
EXAMPLE_HINTS = {"example", "usage", "usageExample"}


def is_english_literal(obj):
    """Keep only English or language-neutral RDF literals."""
    return isinstance(obj, Literal) and (obj.language in (None, "en", "en-gb", "en-us"))


def pred_name(pred):
    """Extract a readable predicate name from an RDF URI."""
    txt = str(pred)
    if "#" in txt:
        return txt.split("#")[-1]
    return txt.rstrip("/").split("/")[-1]


def norm(x):
    """Normalize whitespace and convert null-like values to empty strings."""
    if x is None:
        return ""
    return " ".join(str(x).strip().split())


def collect_rdf_files(root: Path):
    """Recursively collect RDF-like files from the LIdioms folder."""
    return [p for p in root.rglob("*") if p.suffix.lower() in RDF_EXTS]


def parse_file(path: Path):
    """Parse one RDF file into an rdflib Graph."""
    g = Graph()
    try:
        g.parse(path)
        return g
    except Exception as e:
        print(f"Failed to parse {path}: {e}")
        return None


def extract_records_from_graph(g: Graph):
    """
    Extract idiom-like records from one RDF graph.
    We try to detect:
    - label/canonical form -> idiom
    - definition/gloss -> meaning
    - example/usage -> example
    """
    label_map = {}
    def_map = {}
    ex_map = {}

    for s, p, o in g:
        pname = pred_name(p)

        if is_english_literal(o):
            text = norm(o)
            if not text:
                continue

            if any(h.lower() == pname.lower() or h.lower() in pname.lower() for h in LABEL_HINTS):
                label_map.setdefault(s, []).append(text)

            if any(h.lower() == pname.lower() or h.lower() in pname.lower() for h in DEF_HINTS):
                def_map.setdefault(s, []).append(text)

            if any(h.lower() == pname.lower() or h.lower() in pname.lower() for h in EXAMPLE_HINTS):
                ex_map.setdefault(s, []).append(text)

    rows = []
    subjects = set(label_map.keys()) | set(def_map.keys()) | set(ex_map.keys())

    for s in subjects:
        labels = label_map.get(s, [])
        defs = def_map.get(s, [])
        exs = ex_map.get(s, [])

        idiom = labels[0] if labels else ""
        meaning = defs[0] if defs else ""
        example = exs[0] if exs else ""

        # Keep only multi-word expressions
        if idiom and " " in idiom:
            rows.append({
                "idiom": idiom,
                "meaning_en": meaning,
                "example": example,
                "source": "lidioms",
                "source_type": "linked_dataset",
                "pos": "",
                "tags": "",
                "idiom_confidence": "high",
                "source_url": "",
            })

    return rows


def extract_lidioms_en(
    lidioms_root=DEFAULT_LIDIOMS_ROOT,
    output_file=DEFAULT_OUTPUT_FILE,
):
    """
    Extract English idiom-like entries from the local LIdioms RDF dataset.

    Parameters
    ----------
    lidioms_root : path-like
        Root folder containing English RDF files.
    output_file : path-like
        Output CSV path.

    Returns
    -------
    pandas.DataFrame
        Extracted normalized dataset.
    """
    lidioms_root = Path(lidioms_root)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ensure_lidioms_dataset(lidioms_root)

    files = collect_rdf_files(lidioms_root)
    if not lidioms_root.exists():
        raise FileNotFoundError(
            f"LIdioms root not found: {lidioms_root}\n"
            f"Expected something like: data/raw/LIdioms/en"
        )

    files = collect_rdf_files(lidioms_root)

    if not files:
        raise FileNotFoundError(
            f"No RDF files found under: {lidioms_root}"
        )

    print(f"Found {len(files)} RDF-like files")

    rows = []

    for f in files:
        g = parse_file(f)
        if g is None:
            continue
        rows.extend(extract_records_from_graph(g))

    df = pd.DataFrame(rows)

    if df.empty:
        print("No rows extracted. Inspect the RDF predicates in the English files.")
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        return df

    for col in [
        "idiom", "meaning_en", "example", "source", "source_type",
        "pos", "tags", "idiom_confidence", "source_url"
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Keep only valid multi-word idiom candidates
    df = df[
        (df["idiom"] != "") &
        (df["idiom"].str.contains(" "))
    ].copy()

    # Deduplicate by idiom + meaning
    df["dedup_key"] = (
        df["idiom"].str.lower().str.strip()
        + " || " +
        df["meaning_en"].str.lower().str.strip()
    )

    df = (
        df.drop_duplicates(subset=["dedup_key"])
          .drop(columns=["dedup_key"])
          .reset_index(drop=True)
    )

    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved:", output_file)
    print("Rows:", len(df))
    print("Unique idioms:", df["idiom"].nunique())

    return df


def main():
    df = extract_lidioms_en()
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()