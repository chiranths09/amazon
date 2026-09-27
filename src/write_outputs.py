"""
write_outputs.py — Write candidate_pairs.tsv and matching_results.tsv.

Output rules (enforced here, must satisfy validate_submission.py):
  - Tab-separated, UTF-8, one header row.
  - One row per Source 1 entity (even if the ID list is empty).
  - Matched/candidate IDs are comma-separated with NO quoting.
  - No duplicate IDs within a row.
  - No duplicate source1_entity_id rows.
  - Only S2-/S3- IDs that exist in the test set.
  - matched_entity_ids must be a subset of that row's candidate_entity_ids.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


def _deduplicate_ids(ids: List[str]) -> List[str]:
    """Return ids with duplicates removed while preserving order."""
    seen = set()
    result = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def write_outputs(
    candidate_pairs: Dict[str, List[str]],
    matching_results: Dict[str, List[str]],
    s1_ids: List[str],
    valid_cand_ids: Optional[Set[str]] = None,
    output_dir: str | Path = "output",
) -> None:
    """
    Write the two required output TSV files.

    Parameters
    ----------
    candidate_pairs : dict {s1_id: [candidate_cand_ids]}
    matching_results : dict {s1_id: [matched_cand_ids]}
    s1_ids : ordered list of ALL S1 entity IDs (determines row order in output)
    valid_cand_ids : optional set of valid S2/S3 IDs — used to filter out invalid IDs
    output_dir : directory to write output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"

    # Build rows for both files simultaneously
    candidate_rows = []
    matching_rows = []

    seen_s1_ids = set()
    n_filtered = 0

    for s1_id in s1_ids:
        # Enforce no duplicate S1 rows
        if s1_id in seen_s1_ids:
            logger.warning(f"Duplicate S1 entity_id in s1_ids: {s1_id} — skipping")
            continue
        seen_s1_ids.add(s1_id)

        # Get candidate IDs for this entity
        raw_cands = candidate_pairs.get(s1_id, [])
        cand_ids = _deduplicate_ids(raw_cands)

        # Filter to valid S2/S3 IDs if provided
        if valid_cand_ids is not None:
            before = len(cand_ids)
            cand_ids = [x for x in cand_ids if x in valid_cand_ids]
            n_filtered += before - len(cand_ids)

        # Get matched IDs — must be subset of candidate IDs
        raw_matched = matching_results.get(s1_id, [])
        matched_ids = _deduplicate_ids(raw_matched)

        # Enforce matched ⊆ candidates
        cand_set = set(cand_ids)
        matched_ids = [x for x in matched_ids if x in cand_set]

        # Filter matched to valid IDs too
        if valid_cand_ids is not None:
            matched_ids = [x for x in matched_ids if x in valid_cand_ids]

        # Format as comma-separated strings (empty string if no IDs)
        cand_str = ",".join(cand_ids)
        matched_str = ",".join(matched_ids)

        candidate_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_str})
        matching_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})

    if n_filtered > 0:
        logger.warning(f"  Filtered out {n_filtered} invalid (non-S2/S3) candidate IDs")

    # Write candidate_pairs.tsv
    cand_df = pd.DataFrame(candidate_rows)
    cand_df.to_csv(candidate_path, sep="\t", index=False, encoding="utf-8")
    logger.info(f"  Wrote {len(cand_df):,} rows to {candidate_path}")

    # Write matching_results.tsv
    match_df = pd.DataFrame(matching_rows)
    match_df.to_csv(matching_path, sep="\t", index=False, encoding="utf-8")
    n_nonempty = (match_df["matched_entity_ids"] != "").sum()
    logger.info(
        f"  Wrote {len(match_df):,} rows to {matching_path} "
        f"({n_nonempty:,} non-empty matches)"
    )
