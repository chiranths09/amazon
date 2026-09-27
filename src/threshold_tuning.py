"""
threshold_tuning.py — Tune the decision threshold to maximize macro F_0.5.

Approach:
  1. Score all (s1_id, cand_id) validation pairs with the trained model.
  2. Grid-search threshold in [0.05, 0.95] step 0.02.
  3. For each threshold, apply per-entity prediction logic and compute macro F_0.5.
  4. Also tune an optional top-k cap per entity (prevent over-prediction).
  5. Save the best (threshold, top_k) to model/threshold.json.
  6. Save a threshold vs. F_0.5 plot to model/threshold_curve.png.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_threshold(
    scores_df: pd.DataFrame,
    threshold: float,
    top_k: Optional[int] = None,
) -> Dict[str, List[str]]:
    """
    Convert a DataFrame of (s1_id, cand_id, score) into a prediction dict.

    Parameters
    ----------
    scores_df : DataFrame with columns [s1_id, cand_id, score]
    threshold : float, minimum score to predict a match
    top_k : int, optional. If set, keep at most top_k predictions per S1 entity.

    Returns
    -------
    dict {s1_id: [matched_cand_ids]}  (list may be empty if no candidate passes threshold)
    """
    predictions: Dict[str, List[str]] = {}

    for s1_id, group in scores_df.groupby("s1_id"):
        # Filter by threshold
        passing = group[group["score"] >= threshold]

        if top_k is not None and len(passing) > top_k:
            # Keep the top-k highest-scoring candidates
            passing = passing.nlargest(top_k, "score")

        predictions[s1_id] = passing["cand_id"].tolist()

    return predictions


def tune_threshold(
    scores_df: pd.DataFrame,
    ground_truth: Dict[str, List[str]],
    s1_ids_in_val: Set[str],
    threshold_grid: Optional[np.ndarray] = None,
    top_k_grid: Optional[List[Optional[int]]] = None,
    model_dir: str | Path = "model",
) -> Tuple[float, Optional[int]]:
    """
    Grid-search over (threshold, top_k) to maximize macro F_0.5 on the validation split.

    Parameters
    ----------
    scores_df : DataFrame with columns [s1_id, cand_id, score]
    ground_truth : dict {s1_id: [true_matched_ids]}
    s1_ids_in_val : set of S1 IDs in the validation split
    threshold_grid : array of thresholds to try (default: 0.05 to 0.95, step 0.02)
    top_k_grid : list of top_k values to try (default: [None, 5, 10, 20])
    model_dir : directory to save threshold.json and threshold_curve.png

    Returns
    -------
    (best_threshold, best_top_k)
    """
    from src.evaluate import compute_f05_macro

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if threshold_grid is None:
        threshold_grid = np.arange(0.05, 0.96, 0.02)

    if top_k_grid is None:
        top_k_grid = [None, 5, 10, 20]

    # Restrict scores to validation S1 entities
    val_scores = scores_df[scores_df["s1_id"].isin(s1_ids_in_val)].copy()
    val_s1_ids = list(s1_ids_in_val)

    logger.info(
        f"  Tuning threshold over {len(threshold_grid)} thresholds x "
        f"{len(top_k_grid)} top_k values = {len(threshold_grid) * len(top_k_grid)} combos"
    )

    best_f05 = -1.0
    best_thresh = 0.5
    best_top_k = None
    all_results = []  # (threshold, top_k, f05) for plotting

    for top_k in top_k_grid:
        f05_by_thresh = []
        for thresh in threshold_grid:
            preds = apply_threshold(val_scores, thresh, top_k)
            # Ensure every val S1 entity has an entry (even if empty)
            for s1_id in val_s1_ids:
                if s1_id not in preds:
                    preds[s1_id] = []

            metrics = compute_f05_macro(preds, ground_truth, val_s1_ids)
            f05 = metrics["macro_f05"]
            f05_by_thresh.append(f05)
            all_results.append((float(thresh), top_k, f05))

            if f05 > best_f05:
                best_f05 = f05
                best_thresh = float(thresh)
                best_top_k = top_k

        logger.info(
            f"  top_k={top_k}: best F_0.5={max(f05_by_thresh):.4f} "
            f"@ thresh={threshold_grid[np.argmax(f05_by_thresh)]:.2f}"
        )

    logger.info(
        f"\n  *** Best: threshold={best_thresh:.2f}, top_k={best_top_k}, "
        f"F_0.5={best_f05:.4f} ***\n"
    )

    # Save threshold config
    config = {
        "threshold": best_thresh,
        "top_k": best_top_k,
        "val_macro_f05": best_f05,
    }
    config_path = model_dir / "threshold.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"  Threshold config saved to {config_path}")

    # Save curve plot
    _save_threshold_curve(all_results, best_thresh, model_dir)

    return best_thresh, best_top_k


def _save_threshold_curve(
    results: List[Tuple[float, Optional[int], float]],
    best_thresh: float,
    model_dir: Path,
) -> None:
    """Save a threshold vs. F_0.5 plot to model/threshold_curve.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by top_k
        from itertools import groupby
        results_sorted = sorted(results, key=lambda x: (str(x[1]), x[0]))
        for top_k_val, group in groupby(results_sorted, key=lambda x: x[1]):
            group = list(group)
            threshs = [r[0] for r in group]
            f05s = [r[2] for r in group]
            label = f"top_k={top_k_val}" if top_k_val else "top_k=unlimited"
            ax.plot(threshs, f05s, marker=".", label=label)

        ax.axvline(best_thresh, color="red", linestyle="--", label=f"best={best_thresh:.2f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Macro F_0.5")
        ax.set_title("Threshold vs. Macro F_0.5 (validation)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = model_dir / "threshold_curve.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
        logger.info(f"  Threshold curve saved to {plot_path}")
    except Exception as e:
        logger.warning(f"Could not save threshold curve: {e}")


def load_threshold(model_dir: str | Path = "model") -> Tuple[float, Optional[int]]:
    """Load saved threshold and top_k from model/threshold.json."""
    config_path = Path(model_dir) / "threshold.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No threshold config found at {config_path}. Run threshold tuning first."
        )
    with open(config_path) as f:
        config = json.load(f)
    threshold = config["threshold"]
    top_k = config.get("top_k")
    logger.info(f"Loaded threshold={threshold:.2f}, top_k={top_k}")
    return threshold, top_k
