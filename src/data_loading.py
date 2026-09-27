"""
data_loading.py — Load and validate TSV files for the entity resolution pipeline.

All files are tab-separated. Commas appear inside address and ID-list fields,
so sep='\t' is always used explicitly. Columns in source files:
  [entity_id, business_name, business_address, country]
Ground truth columns:
  [source1_entity_id, matched_entity_ids]  (matched_entity_ids is comma-separated)
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns for source files and ground truth
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def load_source(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load a source TSV file (S1, S2, or S3) into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the TSV file.
    nrows : int, optional
        If set, load only the first N data rows (for quick sanity checks).

    Returns
    -------
    pd.DataFrame with columns [entity_id, business_name, business_address, country].
    All columns are kept as strings; missing values are filled with empty string.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    logger.info(f"Loading source file: {path.name} ...")
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        nrows=nrows,
        keep_default_na=False,  # treat empty cells as "" not NaN
        on_bad_lines="warn",    # skip malformed lines but warn
    )

    # Validate schema
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Source file {path.name} is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Keep only the four expected columns (drop extras if any)
    df = df[SOURCE_COLUMNS].copy()

    # Fill NaN produced by empty cells
    df.fillna("", inplace=True)

    # Strip leading/trailing whitespace from the entity_id column
    df["entity_id"] = df["entity_id"].str.strip()

    logger.info(f"  Loaded {len(df):,} records from {path.name}")
    return df


def load_ground_truth(path: str | Path) -> Dict[str, List[str]]:
    """
    Load the ground truth TSV into a dict mapping S1 entity_id → list of matched IDs.

    An empty matched_entity_ids cell means the S1 entity has no matches (singleton).

    Parameters
    ----------
    path : str or Path
        Path to train_ground_truth.tsv.

    Returns
    -------
    dict  {s1_entity_id: [matched_s2_s3_ids]}
          Singletons map to an empty list [].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    logger.info(f"Loading ground truth: {path.name} ...")
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        on_bad_lines="warn",
    )

    missing = [c for c in GT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Ground truth {path.name} is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    gt: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        raw_ids = row["matched_entity_ids"].strip()
        if raw_ids:
            # Split on comma; strip whitespace from each ID
            matched = [x.strip() for x in raw_ids.split(",") if x.strip()]
        else:
            matched = []
        gt[s1_id] = matched

    n_singletons = sum(1 for v in gt.values() if not v)
    logger.info(
        f"  Loaded {len(gt):,} GT rows | "
        f"singletons (no match): {n_singletons:,} | "
        f"with matches: {len(gt) - n_singletons:,}"
    )
    return gt


def load_all_train(base_dir: str | Path):
    """
    Convenience loader: load all four train files from base_dir.

    Parameters
    ----------
    base_dir : str or Path
        Directory containing train_source1.tsv, train_source2.tsv,
        train_source3.tsv, train_ground_truth.tsv.

    Returns
    -------
    tuple (s1_df, s2_df, s3_df, ground_truth_dict)
    """
    base = Path(base_dir)
    s1 = load_source(base / "train_source1.tsv")
    s2 = load_source(base / "train_source2.tsv")
    s3 = load_source(base / "train_source3.tsv")
    gt = load_ground_truth(base / "train_ground_truth.tsv")
    return s1, s2, s3, gt


def load_all_test(base_dir: str | Path):
    """
    Convenience loader: load all three test source files from base_dir.

    Parameters
    ----------
    base_dir : str or Path
        Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv.

    Returns
    -------
    tuple (s1_df, s2_df, s3_df)
    """
    base = Path(base_dir)
    s1 = load_source(base / "test_source1.tsv")
    s2 = load_source(base / "test_source2.tsv")
    s3 = load_source(base / "test_source3.tsv")
    return s1, s2, s3


def get_all_candidate_ids(s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Set[str]:
    """Return the full set of valid S2 and S3 entity IDs (for output validation)."""
    return set(s2_df["entity_id"]) | set(s3_df["entity_id"])
