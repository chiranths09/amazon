# Business Entity Resolution Pipeline

A complete offline Python pipeline for resolving business records from Sources 2 and 3 against a deduplicated Source 1 reference, using a supervised LightGBM classifier.

## Project Structure

```
code/business_entity_resolution/
├── run_pipeline.py          ← CLI entry point (run this)
├── requirements.txt
├── README.md
├── model/                   ← saved after training
│   ├── lgbm_model.pkl
│   ├── threshold.json
│   ├── model_meta.json
│   └── threshold_curve.png
├── output/                  ← generated at inference
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
└── src/
    ├── data_loading.py      ← load TSVs, schema validation
    ├── normalization.py     ← text normalization, abbreviation expansion
    ├── blocking.py          ← candidate generation (token + TF-IDF + phonetic)
    ├── features.py          ← pairwise similarity features
    ├── train_matcher.py     ← LightGBM training
    ├── threshold_tuning.py  ← optimize decision threshold for F_0.5
    ├── inference.py         ← full inference pipeline
    ├── write_outputs.py     ← write submission TSVs
    └── evaluate.py          ← macro F_0.5 evaluation
```

## Dataset Layout (expected)

```
dataset/student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
└── utils/
    └── validate_submission.py
```

## Step-by-Step Reproduction

### 1. Install dependencies

```powershell
cd d:\p\code\business_entity_resolution
pip install -r requirements.txt
```

### 2. Quick sanity check (small sample, ~5 min)

```powershell
python run_pipeline.py --mode all --sample 5000
```

This loads only 5000 rows per source so you can verify the pipeline runs end-to-end in minutes.

### 3. Full training run

```powershell
python run_pipeline.py --mode train
```

**What this does (with timing printed at each stage):**
- Stage 1: Loads all training TSVs (~2.2M S1, ~5M S2, ~5.3M S3)
- Stage 2: Normalizes text (lowercase, punctuation strip, abbreviation expansion)
- Stage 3: Generates blocking candidates (token index + TF-IDF char n-gram + Soundex), prints **blocking recall**
- Stage 4: Builds pairwise feature matrix, trains LightGBM with early stopping, prints **val AUC and AP**
- Stage 5+6: Grid-searches threshold to maximize **macro F_0.5**, prints final val metrics

Saved artifacts: `model/lgbm_model.pkl`, `model/threshold.json`, `model/threshold_curve.png`

### 4. Inference on test set

```powershell
python run_pipeline.py --mode infer
```

**What this does:**
- Loads test TSVs (1.7M S1, ~5M each S2/S3 including France records)
- Normalizes, blocks, featurizes, scores with saved model
- Applies tuned threshold → produces final predictions
- Writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`
- Runs `validate_submission.py` automatically and prints **PASS/FAIL**

### 5. Full pipeline (train + infer) in one command

```powershell
python run_pipeline.py --mode all
```

### 6. Manual validation check

```powershell
python d:\p\dataset\student_resource\utils\validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir d:\p\dataset\student_resource\dataset\test
```

## Output Format

Both output files are **tab-separated UTF-8**:

**`output/candidate_pairs.tsv`**
```
source1_entity_id	candidate_entity_ids
S1-714132312	S2-123456,S3-789012,S2-345678
S1-106407869	
```

**`output/matching_results.tsv`**
```
source1_entity_id	matched_entity_ids
S1-714132312	S2-123456,S3-789012
S1-106407869	
```

Rules enforced:
- One row per Source 1 test entity (including singletons with empty match list)
- Matched IDs are a subset of candidate IDs
- No duplicate IDs within a row
- Only S2-/S3- prefixed IDs

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Country handling | Open string, never one-hot | Generalizes to France (test-only) |
| Blocking | Token + TF-IDF char n-gram + Soundex | High recall, country-scoped |
| Classifier | LightGBM | Fast, handles imbalance, no internet |
| Metric | Macro F_0.5 | Precision-weighted; singletons scored 1.0/0.0 |
| Train/val split | By S1 entity_id | Prevents data leakage |

## CLI Reference

```
python run_pipeline.py [options]

Options:
  --mode {train,infer,all}   Pipeline mode (default: all)
  --train-dir PATH           Train TSV directory
  --test-dir PATH            Test TSV directory
  --output-dir PATH          Output directory (default: output/)
  --model-dir PATH           Model directory (default: model/)
  --sample N                 Load only N rows (sanity-check mode)
  --val-frac FLOAT           Val fraction of S1 IDs (default: 0.20)
  --validator PATH           Path to validate_submission.py
```
