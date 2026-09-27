"""
blocking.py — Candidate pair generation for entity resolution.

Strategy (all offline, no external APIs):
  1. Country-scoped exact token blocking: pairs sharing ≥1 significant name token
     within the same country bucket.
  2. TF-IDF char n-gram (3–5 gram) ANN per country: sklearn TfidfVectorizer +
     NearestNeighbors(metric='cosine') to find approximate near-neighbors.
  3. Phonetic blocking: Soundex on first token of business name.

Results are unioned and deduplicated.

Scale note: with 2M+ S1 and 5M+ candidate records, we need memory-efficient
approaches. We use:
  - sparse TF-IDF matrices (cosine dot-product in blocks)
  - inverted index for token blocking
  - chunked processing where needed
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TFIDF_TOP_K = 10          # top-k TF-IDF neighbors per S1 entity
TFIDF_SIMILARITY_THRESH = 0.20  # minimum cosine similarity to keep
TFIDF_CHUNK_SIZE = 50_000  # number of S1 rows processed per chunk (memory)
TOKEN_BLOCK_MIN_LEN = 3    # minimum token length for token blocking
PHONETIC_TOP_K = 5         # max S2/S3 per Soundex bucket (to bound size)
MAX_PER_TOKEN_BUCKET = 200 # max candidates per token bucket (limits noisy common tokens)

# Stopwords to exclude from token blocking (language-agnostic minimal set)
_STOPWORDS = {
    "the", "of", "and", "a", "an", "in", "at", "to", "for", "on",
    "by", "with", "from", "is", "are", "was", "be",
    # Common business words that appear everywhere (add more to reduce blocking noise)
    "company", "companies", "corporation", "incorporated", "limited",
    "private", "public", "group", "services", "solutions", "systems",
    "technology", "international", "national", "associates",
}

# ---------------------------------------------------------------------------
# Soundex (minimal, stdlib-free implementation)
# ---------------------------------------------------------------------------

_SOUNDEX_MAP = str.maketrans(
    "aehiouwy bfpvcgjkqsxzdtlmnr",
    "000000000111122222222334556",
)


def _soundex(word: str) -> str:
    """Compute a 4-character Soundex code for a word."""
    if not word:
        return "0000"
    word = word.lower()
    first = word[0]
    coded = word.translate(_SOUNDEX_MAP)
    # Remove consecutive duplicates and zeros
    result = first
    prev = coded[0]
    for c in coded[1:]:
        if c != "0" and c != prev:
            result += c
        prev = c
    # Pad or trim to 4 chars
    result = (result + "0000")[:4]
    return result


# ---------------------------------------------------------------------------
# Helper: get significant tokens from a normalized name
# ---------------------------------------------------------------------------

def _sig_tokens(text: str) -> Set[str]:
    """Return significant (non-stopword, length >= min) tokens from text."""
    return {
        t for t in text.split()
        if len(t) >= TOKEN_BLOCK_MIN_LEN and t not in _STOPWORDS
    }


# ---------------------------------------------------------------------------
# Blocking stage 1: Token inverted-index blocking (per country)
# ---------------------------------------------------------------------------

def _token_block(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> Set[Tuple[str, str]]:
    """
    Build an inverted index on significant name tokens, scoped by country string.
    Returns (s1_id, cand_id) pairs that share ≥1 significant token.

    Country is used as a plain string bucket — unseen values work automatically.
    """
    logger.info("  [Token blocking] Building inverted index ...")

    # Build: country -> token -> list of cand_ids (capped at MAX_PER_TOKEN_BUCKET)
    token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for _, row in cand_df.iterrows():
        country = row["country"]
        for tok in _sig_tokens(row["norm_name"]):
            bucket = token_index[country][tok]
            if len(bucket) < MAX_PER_TOKEN_BUCKET:
                bucket.append(row["entity_id"])

    pairs: Set[Tuple[str, str]] = set()
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        country = row["country"]
        cand_bucket = token_index.get(country, {})
        for tok in _sig_tokens(row["norm_name"]):
            for cand_id in cand_bucket.get(tok, set()):
                pairs.add((s1_id, cand_id))

    logger.info(f"  [Token blocking] Generated {len(pairs):,} pairs")
    return pairs


# ---------------------------------------------------------------------------
# Blocking stage 2: TF-IDF char n-gram ANN per country
# ---------------------------------------------------------------------------

def _tfidf_block(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    top_k: int = TFIDF_TOP_K,
    sim_thresh: float = TFIDF_SIMILARITY_THRESH,
    chunk_size: int = TFIDF_CHUNK_SIZE,
) -> Set[Tuple[str, str]]:
    """
    For each country bucket, fit a TF-IDF char n-gram vectorizer on candidates,
    then find the top-k nearest neighbors for each S1 entity using cosine similarity.

    Processes in country-chunks to keep memory bounded. Uses sparse matrix
    dot-product to avoid dense intermediates.
    """
    logger.info("  [TF-IDF blocking] Running per-country TF-IDF ANN ...")
    pairs: Set[Tuple[str, str]] = set()

    # Get all unique country values (open string — no fixed set)
    countries = s1_df["country"].unique()

    for country in countries:
        s1_sub = s1_df[s1_df["country"] == country]
        cand_sub = cand_df[cand_df["country"] == country]

        if len(s1_sub) == 0 or len(cand_sub) == 0:
            continue

        logger.info(
            f"    Country='{country}': {len(s1_sub):,} S1, {len(cand_sub):,} candidates"
        )

        # Fit vectorizer on candidate texts
        cand_texts = cand_sub["norm_combined"].tolist()
        cand_ids = cand_sub["entity_id"].tolist()
        s1_texts = s1_sub["norm_combined"].tolist()
        s1_ids = s1_sub["entity_id"].tolist()

        try:
            vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=200_000,  # cap vocabulary for memory
                sublinear_tf=True,
            )
            cand_matrix = vec.fit_transform(cand_texts)  # sparse (n_cand, vocab)
            cand_matrix_norm = sk_normalize(cand_matrix, norm="l2", copy=True)

            # Process S1 in chunks to avoid huge dense products
            actual_k = min(top_k, len(cand_ids))
            for start in range(0, len(s1_texts), chunk_size):
                chunk_texts = s1_texts[start : start + chunk_size]
                chunk_ids = s1_ids[start : start + chunk_size]
                s1_matrix = vec.transform(chunk_texts)
                s1_matrix_norm = sk_normalize(s1_matrix, norm="l2", copy=True)

                # Cosine similarity via sparse dot product → dense chunk
                sim_block = (s1_matrix_norm @ cand_matrix_norm.T).toarray()

                for i, s1_id in enumerate(chunk_ids):
                    row_sim = sim_block[i]
                    # Get top-k indices above threshold
                    top_idx = np.argpartition(row_sim, -actual_k)[-actual_k:]
                    for idx in top_idx:
                        if row_sim[idx] >= sim_thresh:
                            pairs.add((s1_id, cand_ids[idx]))

        except ValueError as e:
            # Vocabulary too small (e.g., country with 1 record)
            logger.warning(f"    TF-IDF skipped for country='{country}': {e}")

    logger.info(f"  [TF-IDF blocking] Generated {len(pairs):,} pairs")
    return pairs


# ---------------------------------------------------------------------------
# Blocking stage 3: Phonetic blocking (Soundex on first name token)
# ---------------------------------------------------------------------------

def _phonetic_block(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    max_per_bucket: int = PHONETIC_TOP_K,
) -> Set[Tuple[str, str]]:
    """
    Build a (country, soundex_code) → cand_ids index.
    Match S1 entities against candidates with the same (country, soundex) key.
    """
    logger.info("  [Phonetic blocking] Building Soundex index ...")

    # Build index: (country, soundex) -> list of cand_ids
    soundex_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for _, row in cand_df.iterrows():
        tokens = row["norm_name"].split()
        if not tokens:
            continue
        code = _soundex(tokens[0])
        key = (row["country"], code)
        bucket = soundex_index[key]
        if len(bucket) < max_per_bucket:
            bucket.append(row["entity_id"])

    pairs: Set[Tuple[str, str]] = set()
    for _, row in s1_df.iterrows():
        tokens = row["norm_name"].split()
        if not tokens:
            continue
        code = _soundex(tokens[0])
        key = (row["country"], code)
        for cand_id in soundex_index.get(key, []):
            pairs.add((row["entity_id"], cand_id))

    logger.info(f"  [Phonetic blocking] Generated {len(pairs):,} pairs")
    return pairs


# ---------------------------------------------------------------------------
# Main API: generate_candidates
# ---------------------------------------------------------------------------

def generate_candidates(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    use_token: bool = True,
    use_tfidf: bool = True,
    use_phonetic: bool = True,
) -> Set[Tuple[str, str]]:
    """
    Generate candidate (s1_id, cand_id) pairs using the union of blocking strategies.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Source 1 records (must have norm_name, norm_combined, country columns).
    cand_df : pd.DataFrame
        Combined S2 + S3 records (same columns).
    use_token, use_tfidf, use_phonetic : bool
        Toggle individual blocking strategies on/off.

    Returns
    -------
    set of (s1_id, cand_id) tuples — the union of all enabled strategies.
    """
    pairs: Set[Tuple[str, str]] = set()

    if use_token:
        pairs |= _token_block(s1_df, cand_df)

    if use_tfidf:
        pairs |= _tfidf_block(s1_df, cand_df)

    if use_phonetic:
        pairs |= _phonetic_block(s1_df, cand_df)

    logger.info(f"  Total candidate pairs after union & dedup: {len(pairs):,}")
    return pairs


# ---------------------------------------------------------------------------
# Recall evaluation
# ---------------------------------------------------------------------------

def compute_blocking_recall(
    candidate_pairs: Set[Tuple[str, str]],
    ground_truth: Dict[str, List[str]],
    s1_ids_in_split: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Compute blocking recall: what fraction of true GT pairs are retained.

    Parameters
    ----------
    candidate_pairs : set of (s1_id, cand_id)
    ground_truth : dict {s1_id: [matched_ids]}
    s1_ids_in_split : optional set
        If provided, restrict evaluation to this subset of S1 entities.

    Returns
    -------
    dict with keys: recall, n_true_pairs, n_retained, n_total_candidates, n_s1_entities
    """
    # Build fast lookup
    cand_set = candidate_pairs

    total_true = 0
    retained = 0

    for s1_id, matches in ground_truth.items():
        if s1_ids_in_split and s1_id not in s1_ids_in_split:
            continue
        for m in matches:
            total_true += 1
            if (s1_id, m) in cand_set:
                retained += 1

    recall = retained / total_true if total_true > 0 else 1.0
    return {
        "recall": recall,
        "n_true_pairs": total_true,
        "n_retained": retained,
        "n_total_candidates": len(cand_set),
        "pair_ratio": len(cand_set) / max(1, total_true),
    }
