import os
import re
import glob
import csv
import argparse
from typing import Dict, List, Optional

import pandas as pd


TAGS: List[str] = [
    "has_helmet",
    "is_colorful",
    "is_dark",
    "is_female",
    "is_robot",
    "is_animal_themed",
]

# Regex keyword rules 
PATTERNS: Dict[str, List[str]] = {
    "has_helmet": [
        r"\bhelmet\b",
        r"\bheadgear\b",
        r"\bvisor\b",
        r"\bface\s*mask\b",
        r"\bmask\b",
        r"\bgas\s*mask\b",
    ],
    "is_robot": [
        r"\brobot\b",
        r"\bandroid\b",
        r"\bcyborg\b",
        r"\bmech\b",
        r"\bmechanical\b",
        r"\bmetal(?:lic)?\b",
        r"\bmachine\b",
    ],
    "is_animal_themed": [
        r"\banimal\b",
        r"\bcat\b",
        r"\bdog\b",
        r"\bwolf\b",
        r"\bfox\b",
        r"\bbear\b",
        r"\brabbit\b",
        r"\bbunny\b",
        r"\btiger\b",
        r"\blion\b",
        r"\bpanda\b",
        r"\bbird\b",
        r"\bowl\b",
        r"\bdragon\b",
        r"\bdinosaur\b",
        r"\bdeer\b",
        r"\bshark\b",
        r"\bfrog\b",
    ],
    "is_female": [
        r"\bfemale\b",
        r"\bwoman\b",
        r"\bgirl\b",
        r"\blady\b",
        r"\bprincess\b",
        
    ],
    "is_dark": [
        r"\bdark\b",
        r"\bblack\b",
        r"\bshadow\b",
        r"\bnight\b",
        r"\bgoth\b",
        r"\bemo\b",
        r"\bcharcoal\b",
        r"\bmidnight\b",
    ],
    "is_colorful": [
        r"\bcolorful\b",
        r"\brainbow\b",
        r"\bmulticolor(?:ed)?\b",
        r"\bvibrant\b",
        r"\bbright\b",
        r"\bneon\b",
    ],
}

COMPILED: Dict[str, List[re.Pattern]] = {
    tag: [re.compile(pat) for pat in pats] for tag, pats in PATTERNS.items()
}


def pick_caption(row: pd.Series) -> str:
    """Pick best available caption-like field."""
    for key in ("text", "description", "title"):
        if key in row and pd.notna(row[key]):
            s = str(row[key]).strip()
            if s:
                return s
    return ""


def extract_features(caption: str) -> Dict[str, int]:
    """Return strict binary features for the given caption."""
    c = caption.lower()
    feats: Dict[str, int] = {tag: 0 for tag in TAGS}
    for tag in TAGS:
        regexes = COMPILED.get(tag, [])
        feats[tag] = int(any(r.search(c) for r in regexes))
    return feats


def iter_rows_in_parquet(pf: str, chunk_rows: Optional[int]) -> pd.DataFrame:
    """
    Read parquet with pandas. If chunk_rows is provided, we will still read full file
    (pandas doesn't stream parquet easily), but we can subsample rows to test quickly.
    """
    df = pd.read_parquet(pf)
    if chunk_rows is not None and len(df) > chunk_rows:
        df = df.head(chunk_rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_glob",
        type=str,
        default=os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--summykai--minecraft-skins-captioned-900k/snapshots/*/data/*.parquet"
        ),
        help="Glob for parquet shards (default points to HF cache).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="minecraft_features.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--limit_per_file",
        type=int,
        default=0,
        help="For quick testing: limit rows per parquet file (0 = no limit).",
    )
    args = parser.parse_args()

    parquet_files = sorted(glob.glob(args.parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for pattern:\n{args.parquet_glob}")

    limit_per_file = args.limit_per_file if args.limit_per_file > 0 else None

    # Write CSV progressively to not hold 800k+ rows in memory
    out_cols = ["hash", "file_name", *TAGS]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    total = 0
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()

        for pf in parquet_files:
            print(f"Reading: {pf}")
            df = iter_rows_in_parquet(pf, chunk_rows=limit_per_file)

            has_hash = "hash" in df.columns
            has_file = "file_name" in df.columns

            # Iterate rows
            for _, row in df.iterrows():
                caption = pick_caption(row)
                feats = extract_features(caption)

                out_row = {
                    "hash": str(row["hash"]) if has_hash and pd.notna(row["hash"]) else "",
                    "file_name": str(row["file_name"]) if has_file and pd.notna(row["file_name"]) else "",
                    **feats,
                }
                # Ensure strict 0/1
                for tag in TAGS:
                    out_row[tag] = int(bool(out_row[tag]))

                writer.writerow(out_row)
                total += 1

            print(f"  wrote rows so far: {total}")

    print(f"\nDone. Wrote {total} rows to {args.out_csv}")
    print("Columns:", out_cols)


if __name__ == "__main__":
    main()
