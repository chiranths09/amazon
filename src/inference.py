"""
inference.py — Full inference pipeline for the test set.

Steps:
  1. Load test source files (S1, S2, S3).
  2. Normalize all texts.
  3. Generate blocking candidates (same strategies as training).
  4. Score all candidate pairs with the trained LightGBM model.
  5. Apply the tuned threshold (and top_k if configured).
  6. Return candidate_pairs and matching_results dicts.

The country field is handled as an open string throughout — France (or any
new country) appearing only in the test set is correctly handled.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_inference(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    model_dir: str | Path = "model",
    use_token_blocking: bool = True,
    use_tfidf_blocking: bool = True,
    use_phonetic_blocking: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Run the full inference pipeline on provided DataFrames.

    Parameters
    ----------
    s1_df, s2_df, s3_df : normalized DataFrames
        Must already have norm_name, norm_address, norm_combined columns.
    model_dir : directory containing lgbm_model.pkl and threshold.json

    Returns
    -------
    (candidate_pairs_dict, matching_results_dict)
    Both are {s1_id: [cand_ids]} with every S1 entity present (even if empty).
    """
    from src.blocking import generate_candidates
    from src.features import build_feature_matrix, FEATURE_COLS
    from src.train_matcher import load_model
    from src.threshold_tuning import apply_threshold, load_threshold

    model_dir = Path(model_dir)

    # 1. Combine S2 + S3 into a single candidate DataFrame
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    logger.info(f"  Total candidates: {len(cand_df):,} (S2={len(s2_df):,}, S3={len(s3_df):,})")

    # 2. Generate blocking candidates
    logger.info("  Generating blocking candidates ...")
    candidate_set = generate_candidates(
        s1_df, cand_df,
        use_token=use_token_blocking,
        use_tfidf=use_tfidf_blocking,
        use_phonetic=use_phonetic_blocking,
    )

    # Build candidate_pairs dict (all candidates, before threshold)
    candidate_pairs_dict: Dict[str, List[str]] = {
        s1_id: [] for s1_id in s1_df["entity_id"]
    }
    for s1_id, cand_id in candidate_set:
        candidate_pairs_dict[s1_id].append(cand_id)

    logger.info(
        f"  Candidate pairs: {len(candidate_set):,} "
        f"(avg {len(candidate_set)/max(1,len(s1_df)):.1f} per S1 entity)"
    )

    # 3. Build feature matrix for all candidate pairs
    if not candidate_set:
        logger.warning("  No candidate pairs generated! Check blocking configuration.")
        matching_results_dict = {s1_id: [] for s1_id in s1_df["entity_id"]}
        return candidate_pairs_dict, matching_results_dict

    pairs_list = list(candidate_set)
    logger.info("  Building feature matrix for candidate pairs ...")
    feat_df = build_feature_matrix(pairs_list, s1_df, cand_df)

    # 4. Score with model
    logger.info("  Scoring with LightGBM model ...")
    clf = load_model(model_dir)
    X = feat_df[FEATURE_COLS]
    scores = clf.predict_proba(X)[:, 1]
    feat_df["score"] = scores

    scores_df = feat_df[["s1_id", "cand_id", "score"]].copy()

    # 5. Apply tuned threshold
    threshold, top_k = load_threshold(model_dir)
    logger.info(f"  Applying threshold={threshold:.2f}, top_k={top_k} ...")
    matching_results_dict = apply_threshold(scores_df, threshold, top_k)

    # Ensure every S1 entity has an entry (even if no match predicted)
    for s1_id in s1_df["entity_id"]:
        if s1_id not in matching_results_dict:
            matching_results_dict[s1_id] = []

    # Sanity: matched IDs must be subset of candidate IDs
    n_violations = 0
    for s1_id, matched in matching_results_dict.items():
        cands = set(candidate_pairs_dict.get(s1_id, []))
        for m in matched:
            if m not in cands:
                n_violations += 1
    if n_violations > 0:
        logger.warning(
            f"  {n_violations} matched IDs not in candidate set! "
            "This should not happen — check pipeline."
        )

    n_predicted = sum(1 for v in matching_results_dict.values() if v)
    logger.info(
        f"  Predictions: {n_predicted:,} S1 entities matched "
        f"({n_predicted/len(s1_df)*100:.1f}%)"
    )

    return candidate_pairs_dict, matching_results_dict
