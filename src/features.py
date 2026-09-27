"""
features.py — Pairwise similarity feature engineering for entity resolution.

For each (s1_id, cand_id) pair, computes:
  - rapidfuzz string similarity on names and addresses (ratio, token_sort, token_set)
  - Jaccard token overlap on names and addresses
  - TF-IDF cosine similarity on normalized names and addresses
  - Exact-match-after-normalization flags
  - country_match: boolean via string equality (NOT one-hot, generalizes to unseen countries)
  - Length and structural difference features
  - Source identifier (2 or 3, as ordinal integer)

Designed for scale: builds all features in batches using vectorized operations
where possible, falls back to row-wise rapidfuzz for string similarity.
"""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

logger = logging.getLogger(__name__)

# Feature column names in the order they will appear in the feature matrix
FEATURE_COLS = [
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "name_jaccard",
    "addr_jaccard",
    "name_tfidf_cos",
    "addr_tfidf_cos",
    "name_exact_match",
    "addr_exact_match",
    "country_match",
    "name_len_diff",
    "addr_len_diff",
    "name_token_count_diff",
    "addr_token_count_diff",
    "source_num",           # 2 or 3 (ordinal, not one-hot)
]


# ---------------------------------------------------------------------------
# Jaccard similarity on token sets
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between token sets of two strings."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity — fast: fit once on unique texts, indexed lookup
# ---------------------------------------------------------------------------

def _compute_tfidf_cosine_all(
    texts_a: np.ndarray,
    texts_b: np.ndarray,
    max_features: int = 80_000,
    ngram_range: Tuple[int, int] = (2, 4),
) -> np.ndarray:
    """
    Compute cosine similarity for paired arrays texts_a and texts_b via TF-IDF.

    Optimized for large pair counts: fits the vectorizer once on UNIQUE texts,
    then uses integer index arrays to assemble per-pair rows — avoids re-fitting
    the vectorizer for every batch.

    Returns a 1-D float32 array of cosine similarities.
    """
    n = len(texts_a)
    # Deduplicate texts to fit vectorizer only once
    unique_texts = list(dict.fromkeys(list(texts_a) + list(texts_b)))
    if not any(t.strip() for t in unique_texts):
        return np.zeros(n, dtype=np.float32)

    text_to_idx = {t: i for i, t in enumerate(unique_texts)}

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    try:
        mat = vec.fit_transform(unique_texts)          # sparse (n_unique, vocab)
        mat = sk_normalize(mat, norm="l2", copy=False)

        # Index arrays: which row in mat corresponds to each pair element
        idx_a = np.array([text_to_idx[t] for t in texts_a])
        idx_b = np.array([text_to_idx[t] for t in texts_b])

        # Row-wise dot product via element-wise sparse multiply + sum
        # Process in chunks to avoid memory explosion on huge pair sets
        chunk = 50_000
        cosines = np.empty(n, dtype=np.float32)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            rows_a = mat[idx_a[start:end]]
            rows_b = mat[idx_b[start:end]]
            cosines[start:end] = np.array(
                rows_a.multiply(rows_b).sum(axis=1)
            ).flatten()

        return np.clip(cosines, 0.0, 1.0)
    except ValueError:
        return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main: build_feature_matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    pairs: List[Tuple[str, str]],
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the pairwise feature matrix for a list of (s1_id, cand_id) pairs.

    Parameters
    ----------
    pairs : list of (s1_id, cand_id)
    s1_df : DataFrame with columns [entity_id, norm_name, norm_address, country, ...]
    cand_df : DataFrame with columns [entity_id, norm_name, norm_address, country, ...]
    batch_tfidf_size : int
        Number of pairs per TF-IDF batch (controls memory).

    Returns
    -------
    pd.DataFrame with columns = FEATURE_COLS + ['s1_id', 'cand_id']
    """
    # Import here to keep the top of the file clean; rapidfuzz is a required dep
    from rapidfuzz import fuzz

    if not pairs:
        return pd.DataFrame(columns=FEATURE_COLS + ["s1_id", "cand_id"])

    logger.info(f"  Building features for {len(pairs):,} pairs ...")

    # Index source DataFrames by entity_id for fast lookup
    s1_idx = s1_df.set_index("entity_id")
    cand_idx = cand_df.set_index("entity_id")

    # --- Collect raw columns ---
    s1_ids, cand_ids = zip(*pairs)
    s1_ids = list(s1_ids)
    cand_ids = list(cand_ids)

    # Bulk-fetch the required columns
    s1_names    = s1_idx.loc[s1_ids, "norm_name"].values
    s1_addrs    = s1_idx.loc[s1_ids, "norm_address"].values
    s1_countries = s1_idx.loc[s1_ids, "country"].values

    cand_names    = cand_idx.loc[cand_ids, "norm_name"].values
    cand_addrs    = cand_idx.loc[cand_ids, "norm_address"].values
    cand_countries = cand_idx.loc[cand_ids, "country"].values

    n = len(pairs)
    logger.info(f"    Computing rapidfuzz features (bulk cdist) ...")

    # --- rapidfuzz features: use cdist bulk API (C-level, much faster than Python loop) ---
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import Indel

    # Convert to Python str lists once
    sn_list = [str(x) for x in s1_names]
    cn_list = [str(x) for x in cand_names]
    sa_list = [str(x) for x in s1_addrs]
    ca_list = [str(x) for x in cand_addrs]

    def _cdist_pairwise(queries, choices, scorer) -> np.ndarray:
        """
        Compute scorer(queries[i], choices[i]) for each i.
        rapidfuzz.process.cdist returns an (m x n) matrix; we take the diagonal.
        For large N we process in chunks to avoid materialising a huge matrix.
        """
        chunk = 5_000
        out = np.empty(len(queries), dtype=np.float32)
        for start in range(0, len(queries), chunk):
            end = min(start + chunk, len(queries))
            q_chunk = queries[start:end]
            c_chunk = choices[start:end]
            # cdist: rows=queries, cols=choices — returns (chunk x chunk) square
            # We only need the diagonal: pair (i, i)
            mat = process.cdist(q_chunk, c_chunk, scorer=scorer, score_cutoff=0)
            out[start:end] = np.diag(mat).astype(np.float32) / 100.0
        return out

    name_ratio            = _cdist_pairwise(sn_list, cn_list, fuzz.ratio)
    name_token_sort_ratio = _cdist_pairwise(sn_list, cn_list, fuzz.token_sort_ratio)
    name_token_set_ratio  = _cdist_pairwise(sn_list, cn_list, fuzz.token_set_ratio)
    addr_ratio            = _cdist_pairwise(sa_list, ca_list, fuzz.ratio)
    addr_token_sort_ratio = _cdist_pairwise(sa_list, ca_list, fuzz.token_sort_ratio)
    addr_token_set_ratio  = _cdist_pairwise(sa_list, ca_list, fuzz.token_set_ratio)

    logger.info(f"    Computing Jaccard features ...")

    # --- Jaccard features (vectorised via set operations) ---
    name_jaccard = np.array([
        _jaccard(sn_list[i], cn_list[i]) for i in range(n)
    ], dtype=np.float32)
    addr_jaccard = np.array([
        _jaccard(sa_list[i], ca_list[i]) for i in range(n)
    ], dtype=np.float32)

    logger.info(f"    Computing TF-IDF cosine features (single fit over unique texts) ...")

    # --- TF-IDF cosine: fit once on all unique texts, then index ---
    name_tfidf_cos = _compute_tfidf_cosine_all(s1_names, cand_names)
    addr_tfidf_cos = _compute_tfidf_cosine_all(s1_addrs, cand_addrs)

    logger.info(f"    Computing exact-match and structural features ...")

    # --- Exact-match flags ---
    name_exact = (s1_names == cand_names).astype(np.float32)
    addr_exact = (s1_addrs == cand_addrs).astype(np.float32)

    # --- Country match: string equality (never one-hot; handles unseen values) ---
    country_match = (s1_countries == cand_countries).astype(np.float32)

    # --- Length difference features ---
    name_len_diff = np.abs(
        np.array([len(s) for s in s1_names], dtype=np.float32) -
        np.array([len(s) for s in cand_names], dtype=np.float32)
    )
    addr_len_diff = np.abs(
        np.array([len(s) for s in s1_addrs], dtype=np.float32) -
        np.array([len(s) for s in cand_addrs], dtype=np.float32)
    )
    name_tok_cnt_diff = np.abs(
        np.array([len(s.split()) for s in s1_names], dtype=np.float32) -
        np.array([len(s.split()) for s in cand_names], dtype=np.float32)
    )
    addr_tok_cnt_diff = np.abs(
        np.array([len(s.split()) for s in s1_addrs], dtype=np.float32) -
        np.array([len(s.split()) for s in cand_addrs], dtype=np.float32)
    )

    # --- Source indicator (2 or 3, ordinal integer) ---
    source_num = np.array(
        [2 if cid.startswith("S2-") else 3 for cid in cand_ids],
        dtype=np.float32,
    )

    # --- Assemble DataFrame ---
    feat_df = pd.DataFrame({
        "s1_id":                s1_ids,
        "cand_id":              cand_ids,
        "name_ratio":           name_ratio,
        "name_token_sort_ratio": name_token_sort_ratio,
        "name_token_set_ratio": name_token_set_ratio,
        "addr_ratio":           addr_ratio,
        "addr_token_sort_ratio": addr_token_sort_ratio,
        "addr_token_set_ratio": addr_token_set_ratio,
        "name_jaccard":         name_jaccard,
        "addr_jaccard":         addr_jaccard,
        "name_tfidf_cos":       name_tfidf_cos,
        "addr_tfidf_cos":       addr_tfidf_cos,
        "name_exact_match":     name_exact,
        "addr_exact_match":     addr_exact,
        "country_match":        country_match,
        "name_len_diff":        name_len_diff,
        "addr_len_diff":        addr_len_diff,
        "name_token_count_diff": name_tok_cnt_diff,
        "addr_token_count_diff": addr_tok_cnt_diff,
        "source_num":           source_num,
    })

    logger.info(f"  Feature matrix shape: {feat_df.shape}")
    return feat_df
