"""
run_pipeline.py -- CLI entry point for the Business Entity Resolution pipeline.

Runs all stages end-to-end with clear stage-by-stage logging and timing.

Usage:
    # Full pipeline (train + infer):
    python run_pipeline.py --mode all

    # Train only:
    python run_pipeline.py --mode train

    # Inference only (requires a trained model):
    python run_pipeline.py --mode infer

    # Quick sanity check on a small sample:
    python run_pipeline.py --mode all --sample 5000
"""

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path

# Make sure src/ is importable when run from project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def _setup_logging() -> None:
    """Configure logging with timestamps. Force UTF-8 on Windows stdout."""
    import io
    # On Windows the default stdout is cp1252 which can't handle many chars.
    # Wrap it in a UTF-8-safe writer so log messages never crash.
    if hasattr(sys.stdout, "buffer"):
        safe_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    else:
        safe_stdout = sys.stdout

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(safe_stdout)],
        force=True,
    )


def _banner(msg: str) -> None:
    """Print a clearly visible stage banner."""
    bar = "=" * 65
    logging.getLogger().info("\n%s\n  %s\n%s", bar, msg, bar)


@contextlib.contextmanager
def _timed(label: str):
    """Context manager that prints elapsed time for a code block."""
    t0 = time.time()
    yield
    logging.getLogger().info("  [done] %s in %.1fs", label, time.time() - t0)


# ---------------------------------------------------------------------------
# STAGE 0: Dependency check
# ---------------------------------------------------------------------------

def check_deps() -> None:
    log = logging.getLogger(__name__)
    missing = []
    for pkg in ["lightgbm", "rapidfuzz", "matplotlib"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error("Missing packages: %s  ->  pip install %s", missing, " ".join(missing))
        sys.exit(1)


# ---------------------------------------------------------------------------
# STAGE 1: Data loading
# ---------------------------------------------------------------------------

def stage_load_train(train_dir: Path, sample):
    from src.data_loading import load_source, load_ground_truth

    log = logging.getLogger(__name__)
    _banner("STAGE 1 -- Load Training Data")

    s1 = load_source(train_dir / "train_source1.tsv", nrows=sample)
    s2 = load_source(train_dir / "train_source2.tsv", nrows=sample * 3 if sample else None)
    s3 = load_source(train_dir / "train_source3.tsv", nrows=sample * 3 if sample else None)
    gt_full = load_ground_truth(train_dir / "train_ground_truth.tsv")

    # In sample mode restrict GT to loaded S1 entities so blocking recall is meaningful
    loaded_s1_ids = set(s1["entity_id"])
    gt = {k: v for k, v in gt_full.items() if k in loaded_s1_ids}
    log.info(
        "  GT filtered to loaded S1: %d entities (%d with matches, %d singletons)",
        len(gt),
        sum(1 for v in gt.values() if v),
        sum(1 for v in gt.values() if not v),
    )

    log.info("  S1=%d  S2=%d  S3=%d  GT=%d", len(s1), len(s2), len(s3), len(gt))

    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        assert list(df.columns) == ["entity_id", "business_name", "business_address", "country"], \
            f"{name} schema mismatch"
    log.info("  Schema validation: OK")

    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        counts = df["country"].value_counts().head(5).to_dict()
        log.info("  %s country distribution: %s", name, counts)

    return s1, s2, s3, gt


def stage_load_test(test_dir: Path, sample):
    from src.data_loading import load_source

    log = logging.getLogger(__name__)
    _banner("STAGE 1b -- Load Test Data")

    s1 = load_source(test_dir / "test_source1.tsv", nrows=sample)
    s2 = load_source(test_dir / "test_source2.tsv", nrows=sample * 3 if sample else None)
    s3 = load_source(test_dir / "test_source3.tsv", nrows=sample * 3 if sample else None)

    log.info("  Test S1=%d  S2=%d  S3=%d", len(s1), len(s2), len(s3))
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        counts = df["country"].value_counts().head(8).to_dict()
        log.info("  Test %s countries: %s", name, counts)

    return s1, s2, s3


# ---------------------------------------------------------------------------
# STAGE 2: Normalization
# ---------------------------------------------------------------------------

def stage_normalize(s1, s2, s3):
    from src.normalization import normalize_df

    log = logging.getLogger(__name__)
    _banner("STAGE 2 -- Text Normalization")

    with _timed("Normalize S1"):
        s1 = normalize_df(s1)
    with _timed("Normalize S2"):
        s2 = normalize_df(s2)
    with _timed("Normalize S3"):
        s3 = normalize_df(s3)

    log.info("  Sample normalization (first 3 S1 rows):")
    for _, row in s1.head(3).iterrows():
        log.info(
            "    [%s] '%s' -> '%s'",
            row["country"],
            str(row["business_name"])[:50],
            str(row["norm_name"])[:50],
        )

    return s1, s2, s3


# ---------------------------------------------------------------------------
# STAGE 3: Blocking
# ---------------------------------------------------------------------------

def stage_blocking(s1_train, s2_train, s3_train, gt, train_ids, val_ids):
    from src.blocking import generate_candidates, compute_blocking_recall
    import pandas as pd

    log = logging.getLogger(__name__)
    _banner("STAGE 3 -- Blocking (Candidate Generation)")

    cand_df = pd.concat([s2_train, s3_train], ignore_index=True)

    s1_train_sub = s1_train[s1_train["entity_id"].isin(train_ids)]
    s1_val_sub   = s1_train[s1_train["entity_id"].isin(val_ids)]

    log.info("  Generating blocking candidates for TRAIN S1 (%d entities)...", len(s1_train_sub))
    with _timed("Train blocking"):
        train_candidates = generate_candidates(s1_train_sub, cand_df)

    log.info("  Generating blocking candidates for VAL S1 (%d entities)...", len(s1_val_sub))
    with _timed("Val blocking"):
        val_candidates = generate_candidates(s1_val_sub, cand_df)

    train_r = compute_blocking_recall(train_candidates, gt, train_ids)
    val_r   = compute_blocking_recall(val_candidates,   gt, val_ids)

    log.info("\n  -- Blocking Recall Report --")
    log.info(
        "  TRAIN: recall=%.4f (%d/%d GT pairs retained) | %d total candidates | ratio=%.1fx",
        train_r["recall"], train_r["n_retained"], train_r["n_true_pairs"],
        train_r["n_total_candidates"], train_r["pair_ratio"],
    )
    log.info(
        "  VAL  : recall=%.4f (%d/%d GT pairs retained) | %d total candidates",
        val_r["recall"], val_r["n_retained"], val_r["n_true_pairs"],
        val_r["n_total_candidates"],
    )

    if train_r["recall"] < 0.70:
        log.warning(
            "  [WARN] Blocking recall %.2f%% < 70%%. "
            "Consider adjusting TFIDF_TOP_K or sim threshold in blocking.py.",
            train_r["recall"] * 100,
        )

    return train_candidates, val_candidates, cand_df


# ---------------------------------------------------------------------------
# STAGE 4: Feature engineering + model training
# ---------------------------------------------------------------------------

def stage_train(train_candidates, val_candidates, gt, train_ids, val_ids,
                s1_train, cand_df, model_dir: Path):
    from src.features import build_feature_matrix, FEATURE_COLS
    from src.train_matcher import build_training_data, train_model

    log = logging.getLogger(__name__)
    _banner("STAGE 4 -- Feature Engineering + Model Training")

    log.info("  Building training pairs...")
    train_pairs, train_labels = build_training_data(train_candidates, gt, train_ids)
    log.info("  Building validation pairs...")
    val_pairs, val_labels = build_training_data(val_candidates, gt, val_ids, max_pairs=200_000)

    log.info("  Building train feature matrix...")
    with _timed("Train features"):
        X_train = build_feature_matrix(train_pairs, s1_train, cand_df)

    log.info("  Building val feature matrix...")
    with _timed("Val features"):
        X_val = build_feature_matrix(val_pairs, s1_train, cand_df)

    y_train = train_labels
    y_val   = val_labels

    log.info("  X_train: %s  pos=%d neg=%d", X_train.shape, y_train.sum(), (1 - y_train).sum())
    log.info("  X_val:   %s  pos=%d neg=%d", X_val.shape, y_val.sum(), (1 - y_val).sum())

    log.info("  Training LightGBM model...")
    with _timed("Model training"):
        clf = train_model(X_train, y_train, X_val, y_val, model_dir=model_dir)

    return clf, X_val, y_val, val_pairs


# ---------------------------------------------------------------------------
# STAGE 5+6: Threshold tuning + validation evaluation
# ---------------------------------------------------------------------------

def stage_evaluate_val(clf, X_val, val_pairs, gt, val_ids, model_dir: Path):
    from src.features import FEATURE_COLS
    from src.threshold_tuning import tune_threshold, apply_threshold
    from src.evaluate import compute_f05_macro, print_metrics
    import pandas as pd

    log = logging.getLogger(__name__)
    _banner("STAGE 5+6 -- Threshold Tuning + Validation Evaluation")

    val_proba = clf.predict_proba(X_val[FEATURE_COLS])[:, 1]
    scores_df = pd.DataFrame({
        "s1_id":   list(X_val["s1_id"]),
        "cand_id": list(X_val["cand_id"]),
        "score":   val_proba,
    })

    log.info("  Tuning decision threshold on validation set...")
    with _timed("Threshold tuning"):
        best_thresh, best_top_k = tune_threshold(
            scores_df, gt, set(val_ids), model_dir=model_dir
        )

    preds = apply_threshold(scores_df, best_thresh, best_top_k)
    for s1_id in val_ids:
        if s1_id not in preds:
            preds[s1_id] = []

    metrics = compute_f05_macro(preds, gt, list(val_ids), verbose=True)
    print_metrics(metrics, title=f"VAL Evaluation @ threshold={best_thresh:.2f}")

    return best_thresh, best_top_k


# ---------------------------------------------------------------------------
# STAGE 7: Full inference on test set
# ---------------------------------------------------------------------------

def stage_infer(s1_test, s2_test, s3_test, model_dir: Path, output_dir: Path):
    from src.normalization import normalize_df
    from src.inference import run_inference
    from src.write_outputs import write_outputs
    from src.data_loading import get_all_candidate_ids

    log = logging.getLogger(__name__)
    _banner("STAGE 7 -- Inference on Test Set")

    log.info("  Normalizing test data...")
    with _timed("Normalize test S1"):
        s1_test = normalize_df(s1_test)
    with _timed("Normalize test S2"):
        s2_test = normalize_df(s2_test)
    with _timed("Normalize test S3"):
        s3_test = normalize_df(s3_test)

    valid_ids = get_all_candidate_ids(s2_test, s3_test)

    log.info("  Running inference pipeline...")
    with _timed("Full inference"):
        candidate_pairs, matching_results = run_inference(
            s1_test, s2_test, s3_test, model_dir=model_dir
        )

    log.info("  Writing output files...")
    write_outputs(
        candidate_pairs, matching_results,
        s1_ids=list(s1_test["entity_id"]),
        valid_cand_ids=valid_ids,
        output_dir=output_dir,
    )
    log.info("  Outputs written to %s/", output_dir)


# ---------------------------------------------------------------------------
# STAGE 8: Validate submission
# ---------------------------------------------------------------------------

def stage_validate(test_dir: Path, output_dir: Path, validator_path: Path) -> bool:
    import subprocess

    log = logging.getLogger(__name__)
    _banner("STAGE 8 -- Submission Validation")

    cmd = [
        sys.executable, str(validator_path),
        "--matching",  str(output_dir / "matching_results.tsv"),
        "--candidate", str(output_dir / "candidate_pairs.tsv"),
        "--test-dir",  str(test_dir),
    ]
    log.info("  Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in (result.stdout + result.stderr).strip().splitlines():
        log.info("  %s", line)

    if result.returncode == 0:
        log.info("  [PASS] Submission is valid and safe to submit!")
        return True
    else:
        log.error("  [FAIL] Fix the issues above before submitting.")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _setup_logging()
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "infer", "all"], default="all",
                        help="Pipeline mode (default: all)")
    parser.add_argument("--train-dir",
                        default="d:/p/dataset/student_resource/dataset/train",
                        help="Directory with train TSV files")
    parser.add_argument("--test-dir",
                        default="d:/p/dataset/student_resource/dataset/test",
                        help="Directory with test TSV files")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for output TSV files (default: output/)")
    parser.add_argument("--model-dir",  default="model",
                        help="Directory for model artifacts (default: model/)")
    parser.add_argument("--validator",
                        default="d:/p/dataset/student_resource/utils/validate_submission.py",
                        help="Path to validate_submission.py")
    parser.add_argument("--sample", type=int, default=None,
                        help="Load only N rows per source (sanity-check mode)")
    parser.add_argument("--val-frac", type=float, default=0.20,
                        help="Fraction of S1 entities held out for validation (default: 0.20)")
    args = parser.parse_args()

    train_dir  = Path(args.train_dir)
    test_dir   = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    model_dir  = Path(args.model_dir)
    validator  = Path(args.validator)

    t_total = time.time()
    log.info("Business Entity Resolution Pipeline | mode=%s", args.mode)
    if args.sample:
        log.warning("SAMPLE MODE: only %d rows per source (sanity-check, not full run)", args.sample)

    check_deps()

    # ---- TRAIN ---------------------------------------------------------------
    if args.mode in ("train", "all"):
        from src.train_matcher import split_s1_ids

        with _timed("Load train data"):
            s1_tr, s2_tr, s3_tr, gt = stage_load_train(train_dir, args.sample)

        with _timed("Normalize train"):
            s1_tr, s2_tr, s3_tr = stage_normalize(s1_tr, s2_tr, s3_tr)

        train_ids, val_ids = split_s1_ids(s1_tr, val_frac=args.val_frac)

        with _timed("Blocking"):
            train_cands, val_cands, cand_df_tr = stage_blocking(
                s1_tr, s2_tr, s3_tr, gt, train_ids, val_ids
            )

        clf, X_val, y_val, val_pairs = stage_train(
            train_cands, val_cands, gt, train_ids, val_ids,
            s1_tr, cand_df_tr, model_dir=model_dir,
        )

        stage_evaluate_val(clf, X_val, val_pairs, gt, val_ids, model_dir=model_dir)

    # ---- INFER ---------------------------------------------------------------
    if args.mode in ("infer", "all"):
        with _timed("Load test data"):
            s1_te, s2_te, s3_te = stage_load_test(test_dir, args.sample)

        with _timed("Inference + write"):
            stage_infer(s1_te, s2_te, s3_te, model_dir=model_dir, output_dir=output_dir)

        if validator.exists():
            stage_validate(test_dir, output_dir, validator)
        else:
            log.warning("Validator not found at %s -- skipping", validator)

    _banner(f"PIPELINE COMPLETE -- total time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
