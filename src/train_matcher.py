"""
train_matcher.py — Train a LightGBM binary classifier for entity matching.

Pipeline:
  1. Load train data; normalize; split 80/20 by S1 entity_id (no leakage).
  2. Run blocking on the train split to get the candidate pool.
  3. Label: positive = (s1_id, matched_id) pairs from ground truth;
     negative = remaining blocking candidates.
  4. Build pairwise feature matrix.
  5. Train LightGBM with class_weight handling (positives are rare).
  6. Evaluate on val split (AUC, precision@threshold).
  7. Save model + metadata to model/ directory.

Scale note: for very large datasets (2M+ S1), we train on a SAMPLE of
blocking candidates to fit in memory and reasonable training time.
The sample size is configurable via MAX_TRAIN_PAIRS.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

# Maximum number of (candidate) pairs to use for training.
# Set lower to speed up training; higher for better coverage.
MAX_TRAIN_PAIRS = 500_000
MAX_NEG_RATIO = 20  # max negatives per positive (to balance)
RANDOM_SEED = 42


def split_s1_ids(
    s1_df: pd.DataFrame,
    val_frac: float = 0.20,
    random_seed: int = RANDOM_SEED,
) -> Tuple[Set[str], Set[str]]:
    """
    Split S1 entity IDs into train and validation sets.
    Split is on entity_id so no S1 entity appears in both splits.
    """
    ids = s1_df["entity_id"].tolist()
    np.random.seed(random_seed)
    np.random.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    logger.info(f"  Train S1 IDs: {len(train_ids):,} | Val S1 IDs: {len(val_ids):,}")
    return train_ids, val_ids


def build_training_data(
    candidate_pairs: Set[Tuple[str, str]],
    ground_truth: Dict[str, List[str]],
    s1_ids_subset: Set[str],
    max_pairs: int = MAX_TRAIN_PAIRS,
    max_neg_ratio: int = MAX_NEG_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """
    From blocking candidates, label positives and sample negatives.

    Returns
    -------
    (pairs_list, labels_array) where labels are 0/1.
    """
    # Build ground truth lookup for the given S1 subset
    gt_set: Set[Tuple[str, str]] = set()
    for s1_id in s1_ids_subset:
        if s1_id in ground_truth:
            for cand_id in ground_truth[s1_id]:
                gt_set.add((s1_id, cand_id))

    # Partition candidates into positives (in GT) and negatives
    positives = []
    negatives = []
    for pair in candidate_pairs:
        s1_id, cand_id = pair
        if s1_id not in s1_ids_subset:
            continue
        if pair in gt_set:
            positives.append(pair)
        else:
            negatives.append(pair)

    logger.info(
        f"  Positives in blocking candidates: {len(positives):,} / {len(gt_set):,} GT pairs"
    )
    logger.info(f"  Raw negatives: {len(negatives):,}")

    # Sample negatives to control the ratio
    rng = np.random.RandomState(random_seed)
    max_neg = min(len(negatives), len(positives) * max_neg_ratio)
    if len(negatives) > max_neg:
        neg_indices = rng.choice(len(negatives), size=max_neg, replace=False)
        negatives = [negatives[i] for i in neg_indices]

    # If total still exceeds max_pairs, subsample proportionally
    total = len(positives) + len(negatives)
    if total > max_pairs:
        keep_frac = max_pairs / total
        n_pos = max(1, int(len(positives) * keep_frac))
        n_neg = max_pairs - n_pos
        pos_idx = rng.choice(len(positives), size=n_pos, replace=False)
        neg_idx = rng.choice(len(negatives), size=min(n_neg, len(negatives)), replace=False)
        positives = [positives[i] for i in pos_idx]
        negatives = [negatives[i] for i in neg_idx]

    pairs = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)

    # Shuffle
    shuffle_idx = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in shuffle_idx]
    labels = labels[shuffle_idx]

    logger.info(
        f"  Final training set: {len(labels):,} pairs "
        f"({labels.sum():,} pos, {(1-labels).sum():,} neg)"
    )
    return pairs, labels


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_dir: str | Path = "model",
) -> object:
    """
    Train a LightGBM binary classifier.

    Parameters
    ----------
    X_train, X_val : DataFrames of features (FEATURE_COLS columns)
    y_train, y_val : binary label arrays
    model_dir : directory to save model and metadata

    Returns
    -------
    Trained LGBMClassifier.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "lightgbm is not installed. Run: pip install lightgbm"
        )

    from src.features import FEATURE_COLS

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Scale pos_weight for imbalanced labels
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(1, n_pos)
    logger.info(f"  Training LightGBM | scale_pos_weight={scale_pos_weight:.2f}")

    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )

    # Ensure only feature columns are used (drop s1_id, cand_id, label if present)
    X_tr = X_train[FEATURE_COLS]
    X_vl = X_val[FEATURE_COLS]

    t0 = time.time()
    clf.fit(
        X_tr, y_train,
        eval_set=[(X_vl, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=100)],
    )
    elapsed = time.time() - t0
    logger.info(f"  Training done in {elapsed:.1f}s | best_iteration={clf.best_iteration_}")

    # Evaluation
    val_proba = clf.predict_proba(X_vl)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    val_ap = average_precision_score(y_val, val_proba)
    logger.info(f"  Val AUC={val_auc:.4f} | Val AP={val_ap:.4f}")

    # Feature importances
    importances = sorted(
        zip(FEATURE_COLS, clf.feature_importances_),
        key=lambda x: -x[1],
    )
    logger.info("  Top-10 feature importances:")
    for feat, imp in importances[:10]:
        logger.info(f"    {feat:35s}: {imp:6.0f}")

    # Save model
    model_path = model_dir / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"  Model saved to {model_path}")

    # Save metadata
    meta = {
        "feature_cols": FEATURE_COLS,
        "val_auc": val_auc,
        "val_ap": val_ap,
        "best_iteration": clf.best_iteration_,
        "scale_pos_weight": scale_pos_weight,
        "n_train": len(y_train),
        "n_val": len(y_val),
    }
    meta_path = model_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Metadata saved to {meta_path}")

    return clf


def load_model(model_dir: str | Path = "model"):
    """Load a saved LightGBM model from disk."""
    model_path = Path(model_dir) / "lgbm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}. Run training first.")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return clf
