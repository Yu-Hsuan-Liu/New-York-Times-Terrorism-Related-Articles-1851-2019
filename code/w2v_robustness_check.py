# -*- coding: utf-8 -*-
"""
w2v_robustness_check.py — Word2Vec Robustness Check (Unified min_count)
=======================================================================
Addresses reviewer concern about asymmetric hyperparameters (min_count=3/5/20).

This script retrains all 14 Word2Vec models (7 periods × 2 architectures)
using a UNIFIED min_count=5, keeping all other hyperparameters identical
to the original training. It then compares the top-25 ranked terms for
key target words ("terrorism", "terrorist") between original and robustness
models to verify that core findings are stable.

OUTPUT (saved to nlp_results/):
    - W2V_Robustness_Comparison.csv  — Side-by-side top-25 comparison
    - W2V_Robustness_Summary.txt     — Human-readable summary
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────
SEED = 123
# --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
_os.makedirs(RESULTS_ROOT, exist_ok=True)
PARQUET_PATH = _os.path.join(DATA_ROOT, 'combined_processed_df.parquet')
ANALYSIS_COLUMN = "clean_lemmatized_phrased"
OUTPUT_DIR = RESULTS_ROOT
ORIGINAL_CSV = os.path.join(OUTPUT_DIR, "FINAL_ALL_MODELS_RESULTS.csv")

UNIFIED_MIN_COUNT = 5
TARGET_KEYWORDS = ["terrorism", "terrorist"]
TOPN = 25

PERIOD_ORDER = [
    "1851-1900", "1901-1930", "1931-1950", "1951-1960",
    "1961-1980", "1981-20010910", "20010911-2019",
]

# Original hyperparameters (min_count varies; we override it)
def get_hyperparams(num_docs):
    """Return original hyperparams but with unified min_count."""
    if num_docs < 1000:
        return dict(vector_size=50, window=3, epochs=50, min_count=UNIFIED_MIN_COUNT)
    elif num_docs < 10000:
        return dict(vector_size=100, window=5, epochs=20, min_count=UNIFIED_MIN_COUNT)
    else:
        return dict(vector_size=200, window=5, epochs=10, min_count=UNIFIED_MIN_COUNT)


def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load Data ───────────────────────────────────────────────────
    print("Loading parquet data...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Full dataset: {len(df):,} articles")

    # ── 2. Load Original Results ───────────────────────────────────────
    print(f"Loading original W2V results from {ORIGINAL_CSV}...")
    orig_df = pd.read_csv(ORIGINAL_CSV)
    print(f"  Original results: {len(orig_df):,} rows")

    # ── 3. Retrain with Unified min_count ─────────────────────────────
    all_robust_data = []

    for period in PERIOD_ORDER:
        subset = df[df["label"] == period]
        num_docs = len(subset)
        params = get_hyperparams(num_docs)
        sentences = subset[ANALYSIS_COLUMN].str.split().tolist()

        print(f"\nPeriod {period} ({num_docs:,} docs)")
        print(f"  Params: vec_size={params['vector_size']}, window={params['window']}, "
              f"epochs={params['epochs']}, min_count={params['min_count']}")

        for sg_mode in [0, 1]:
            mode_name = "SkipGram" if sg_mode == 1 else "CBOW"
            print(f"  Training {mode_name}...", end=" ", flush=True)

            model = Word2Vec(
                sentences=sentences,
                vector_size=params["vector_size"],
                window=params["window"],
                min_count=params["min_count"],
                workers=1,
                sg=sg_mode,
                epochs=params["epochs"],
                seed=SEED,
            )
            print("Done.")

            for target in TARGET_KEYWORDS:
                if target in model.wv:
                    top_words = model.wv.most_similar(target, topn=TOPN)
                    temp_df = pd.DataFrame(top_words, columns=["Neighbor_Word", "Similarity"])
                    temp_df["Target_Word"] = target
                    temp_df["Period"] = period
                    temp_df["Model_Type"] = mode_name
                    temp_df["Rank"] = range(1, TOPN + 1)
                    all_robust_data.append(temp_df)
                else:
                    print(f"    ['{target}'] NOT FOUND in vocabulary.")

    robust_df = pd.concat(all_robust_data, ignore_index=True)

    # ── 4. Compare Original vs Robustness ─────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON: Original (variable min_count) vs Robustness (min_count=5)")
    print("=" * 70)

    comparison_rows = []
    summary_lines = []

    for period in PERIOD_ORDER:
        for target in TARGET_KEYWORDS:
            for model_type in ["CBOW", "SkipGram"]:
                # Original top-25
                orig_mask = (
                    (orig_df["Period"] == period)
                    & (orig_df["Target_Word"] == target)
                    & (orig_df["Model_Type"] == model_type)
                    & (orig_df["Rank"] <= TOPN)
                )
                orig_terms = set(orig_df.loc[orig_mask, "Neighbor_Word"].tolist())

                # Robustness top-25
                rob_mask = (
                    (robust_df["Period"] == period)
                    & (robust_df["Target_Word"] == target)
                    & (robust_df["Model_Type"] == model_type)
                )
                rob_terms = set(robust_df.loc[rob_mask, "Neighbor_Word"].tolist())

                overlap = orig_terms & rob_terms
                only_orig = orig_terms - rob_terms
                only_robust = rob_terms - orig_terms
                jaccard = len(overlap) / len(orig_terms | rob_terms) if (orig_terms | rob_terms) else 0

                combo_label = f"{period} | {target} | {model_type}"
                line = (
                    f"{combo_label:50s}  "
                    f"Overlap: {len(overlap):2d}/{TOPN}  "
                    f"Jaccard: {jaccard:.2f}  "
                    f"Dropped: {sorted(only_orig)}  "
                    f"New: {sorted(only_robust)}"
                )
                summary_lines.append(line)
                print(line)

                comparison_rows.append({
                    "Period": period,
                    "Target_Word": target,
                    "Model_Type": model_type,
                    "Overlap_Count": len(overlap),
                    "Jaccard": round(jaccard, 3),
                    "Shared_Terms": ", ".join(sorted(overlap)),
                    "Only_Original": ", ".join(sorted(only_orig)),
                    "Only_Robust": ", ".join(sorted(only_robust)),
                })

    comparison_df = pd.DataFrame(comparison_rows)

    # ── 5. Overall Summary ────────────────────────────────────────────
    mean_overlap = comparison_df["Overlap_Count"].mean()
    mean_jaccard = comparison_df["Jaccard"].mean()
    min_overlap = comparison_df["Overlap_Count"].min()

    print(f"\n{'='*70}")
    print(f"OVERALL: Mean overlap = {mean_overlap:.1f}/{TOPN}, "
          f"Mean Jaccard = {mean_jaccard:.2f}, "
          f"Min overlap = {min_overlap}/{TOPN}")
    print(f"{'='*70}")

    # ── 6. Save Results ───────────────────────────────────────────────
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "W2V_Robustness_Comparison.csv"), index=False)
    robust_df.to_csv(os.path.join(OUTPUT_DIR, "W2V_Robustness_FullResults.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "W2V_Robustness_Summary.txt"), "w") as f:
        f.write(f"Word2Vec Robustness Check: Unified min_count={UNIFIED_MIN_COUNT}\n")
        f.write(f"Original: variable min_count (3/5/20 by corpus size)\n")
        f.write(f"{'='*70}\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\n{'='*70}\n")
        f.write(f"OVERALL: Mean overlap = {mean_overlap:.1f}/{TOPN}, "
                f"Mean Jaccard = {mean_jaccard:.2f}, "
                f"Min overlap = {min_overlap}/{TOPN}\n")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s. Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
