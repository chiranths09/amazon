"""
evaluate.py — F_0.5 macro evaluation for entity resolution.

F_0.5 formula (per S1 entity):
    F_0.5 = 1.25 * P * R / (0.25 * P + R)

Macro average: unweighted mean across all S1 entities in the evaluation set.

Singleton handling:
  - If S1 entity has no ground truth matches (singleton) AND no predictions → score 1.0
  - If S1 entity has no ground truth matches BUT has predictions → score 0.0
    (any false positive on a true singleton = 0)
  - If S1 entity has ground truth matches → standard F_0.5

This module is used for local validation before submitting to the leaderboard.
"""

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def f05_score(precision: float, recall: float) -> float:
    """
    Compute F_0.5 from precision and recall.
    Beta=0.5 → precision weighted more heavily than recall.
    """
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return 1.25 * precision * recall / denom


def per_entity_f05(
    predicted: Set[str],
    ground_truth: Set[str],
) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F_0.5 for a single S1 entity.

    Parameters
    ----------
    predicted : set of predicted matched IDs
    ground_truth : set of true matched IDs

    Returns
    -------
    (precision, recall, f05)
    """
    is_singleton = len(ground_truth) == 0
    no_prediction = len(predicted) == 0

    if is_singleton and no_prediction:
        return 1.0, 1.0, 1.0  # perfect: correctly predicted no match

    if is_singleton and not no_prediction:
        return 0.0, 0.0, 0.0  # false positive on a singleton = 0

    if no_prediction and not is_singleton:
        # All misses
        return 0.0, 0.0, 0.0

    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, f05_score(precision, recall)


def compute_f05_macro(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    s1_ids: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute macro-averaged F_0.5 across all S1 entities.

    Parameters
    ----------
    predictions : dict {s1_id: [predicted_matched_ids]}
    ground_truth : dict {s1_id: [true_matched_ids]}
    s1_ids : list of S1 entity IDs to evaluate (defaults to all in ground_truth).
    verbose : bool
        If True, print per-entity breakdown for the worst predictions.

    Returns
    -------
    dict with keys:
        macro_f05, macro_precision, macro_recall,
        singleton_accuracy,
        n_entities, n_singletons, n_with_matches,
        n_true_positive_pairs, n_false_positive_pairs, n_false_negative_pairs
    """
    if s1_ids is None:
        s1_ids = list(ground_truth.keys())

    f05_scores = []
    precisions = []
    recalls = []
    singleton_correct = 0
    n_singletons = 0
    n_with_matches = 0
    total_tp = total_fp = total_fn = 0
    worst_entities = []  # for verbose output

    for s1_id in s1_ids:
        gt_set = set(ground_truth.get(s1_id, []))
        pred_set = set(predictions.get(s1_id, []))

        p, r, f = per_entity_f05(pred_set, gt_set)
        f05_scores.append(f)
        precisions.append(p)
        recalls.append(r)

        is_singleton = len(gt_set) == 0
        if is_singleton:
            n_singletons += 1
            if not pred_set:
                singleton_correct += 1
        else:
            n_with_matches += 1

        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if verbose and f < 0.5:
            worst_entities.append((s1_id, gt_set, pred_set, f))

    if verbose and worst_entities:
        worst_entities.sort(key=lambda x: x[3])
        logger.info("  Worst 10 predictions (F_0.5 < 0.5):")
        for s1_id, gt, pred, f in worst_entities[:10]:
            logger.info(f"    {s1_id}: GT={list(gt)[:3]} | Pred={list(pred)[:3]} | F0.5={f:.3f}")

    macro_f05 = np.mean(f05_scores) if f05_scores else 0.0
    macro_p = np.mean(precisions) if precisions else 0.0
    macro_r = np.mean(recalls) if recalls else 0.0
    singleton_acc = singleton_correct / max(1, n_singletons)

    return {
        "macro_f05": float(macro_f05),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "singleton_accuracy": float(singleton_acc),
        "n_entities": len(s1_ids),
        "n_singletons": n_singletons,
        "n_with_matches": n_with_matches,
        "n_true_positive_pairs": total_tp,
        "n_false_positive_pairs": total_fp,
        "n_false_negative_pairs": total_fn,
    }


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation") -> None:
    """Pretty-print the metrics dict returned by compute_f05_macro."""
    logger.info(f"\n{'='*55}")
    logger.info(f"  {title}")
    logger.info(f"{'='*55}")
    logger.info(f"  Macro F_0.5          : {metrics['macro_f05']:.4f}")
    logger.info(f"  Macro Precision      : {metrics['macro_precision']:.4f}")
    logger.info(f"  Macro Recall         : {metrics['macro_recall']:.4f}")
    logger.info(f"  Singleton Accuracy   : {metrics['singleton_accuracy']:.4f}")
    logger.info(f"  Entities evaluated   : {metrics['n_entities']:,}")
    logger.info(f"  Singletons           : {metrics['n_singletons']:,}")
    logger.info(f"  Entities w/ matches  : {metrics['n_with_matches']:,}")
    logger.info(f"  TP / FP / FN pairs   : {metrics['n_true_positive_pairs']:,} / "
                f"{metrics['n_false_positive_pairs']:,} / {metrics['n_false_negative_pairs']:,}")
    logger.info(f"{'='*55}\n")
