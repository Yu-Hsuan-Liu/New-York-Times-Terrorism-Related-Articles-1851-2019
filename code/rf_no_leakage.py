# -*- coding: utf-8 -*-
"""
rf_no_leakage.py — Leak-Free Random Forest Classification
==========================================================
Fixes the train-test data leakage in the original pipeline.

ORIGINAL PROBLEM:
    Oversampling with replacement was applied BEFORE the train-test split,
    allowing duplicate articles to appear in both training and test sets.

FIX:
    1. Sample n=597 per period WITHOUT replacement (no duplicates)
    2. THEN perform stratified 80/20 train-test split
    3. Train RF on training set only
    4. Evaluate on clean, unseen test set

OUTPUT (saved to nlp_results/):
    - RF_Metrics.txt              — Weighted F1, Macro F1, best params
    - RF_Final_PerClass.csv       — Per-period Precision, Recall, F1, Support
    - RF_Final_Bootstrap_CI.txt   — 95% Bootstrap CI for weighted F1
    - RF_Feature_Importance.csv   — Top 50 features by Gini importance
    - RF_Confusion_Matrix.csv     — 7×7 confusion matrix
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────
SEED = 123
np.random.seed(SEED)

# --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
_os.makedirs(RESULTS_ROOT, exist_ok=True)
PARQUET_PATH = _os.path.join(DATA_ROOT, 'combined_processed_df.parquet')
ANALYSIS_COLUMN = "clean_lemmatized_phrased"
OUTPUT_DIR = RESULTS_ROOT
SAMPLE_DIR = _os.path.join(RESULTS_ROOT, 'sample_no_leak')

N_BOOTSTRAP = 1000
TEST_SIZE = 0.20

PERIOD_ORDER = [
    "1851-1900", "1901-1930", "1931-1950", "1951-1960",
    "1961-1980", "1981-20010910", "20010911-2019",
]

PARAM_GRID = {
    "n_estimators": [10, 100],
    "max_depth": [None, 1, 10],
    "criterion": ["gini", "entropy"],
    "random_state": [SEED],
}


def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # ── 1. Load Data ───────────────────────────────────────────────────
    print("Loading parquet data...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Full dataset: {len(df):,} articles")
    print(f"  Period distribution:")
    for p in PERIOD_ORDER:
        n = (df["label"] == p).sum()
        print(f"    {p}: {n:,}")

    # ── 2. Balanced Sampling WITHOUT Replacement ───────────────────────
    min_n = df.groupby("label").size().min()
    print(f"\nSampling n={min_n} per period WITHOUT replacement (seed={SEED})...")

    sampled = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min_n, replace=False, random_state=SEED))
        .reset_index(drop=True)
    )
    total_sampled = len(sampled)
    print(f"  Total sampled: {total_sampled} articles ({total_sampled // 7} per period)")

    # Verify no duplicates
    assert sampled.index.is_unique, "Index is not unique after sampling!"
    print("  Duplicate check PASSED — all articles are unique.")

    # ── 3. Vectorize ──────────────────────────────────────────────────
    print("\nVectorizing (CountVectorizer, unigrams)...")
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    X = vectorizer.fit_transform(sampled[ANALYSIS_COLUMN])
    y = sampled["label"]
    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Feature matrix: {X.shape[0]} × {X.shape[1]} (sparse)")

    # Save vectorizer
    pickle.dump(vectorizer, open(os.path.join(SAMPLE_DIR, "vectorizer.pkl"), "wb"))

    # ── 4. Stratified Train-Test Split ────────────────────────────────
    print(f"\nStratified train-test split ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}, seed={SEED})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"  Training set: {X_train.shape[0]} articles")
    print(f"  Test set:     {X_test.shape[0]} articles")
    print(f"  Train distribution:")
    for p in PERIOD_ORDER:
        print(f"    {p}: {(y_train == p).sum()}")
    print(f"  Test distribution:")
    for p in PERIOD_ORDER:
        print(f"    {p}: {(y_test == p).sum()}")

    # ── 5. GridSearchCV ───────────────────────────────────────────────
    print(f"\nRunning GridSearchCV (5-fold CV on training set)...")
    grid = GridSearchCV(
        RandomForestClassifier(random_state=SEED),
        PARAM_GRID,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    print(f"  Best CV score: {grid.best_score_:.4f}")
    print(f"  Best params: {best_params}")

    # ── 6. Train Best Model on Full Training Set ──────────────────────
    print("\nTraining final model with best params...")
    best_rf = RandomForestClassifier(**best_params)
    best_rf.fit(X_train, y_train)

    # Save model
    pickle.dump(best_rf, open(os.path.join(SAMPLE_DIR, "my_model.pkl"), "wb"))

    # ── 7. Evaluate on Test Set ───────────────────────────────────────
    y_pred = best_rf.predict(X_test)

    # Overall weighted metrics
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )

    print(f"\n{'='*60}")
    print(f"  RESULTS (Leak-Free, Without-Replacement Sampling)")
    print(f"{'='*60}")
    print(f"  Weighted Precision: {w_prec:.4f}")
    print(f"  Weighted Recall:    {w_rec:.4f}")
    print(f"  Weighted F1-Score:  {w_f1:.4f}")
    print(f"  Macro F1-Score:     {macro_f1:.4f}")
    print(f"  Best params:        {best_params['criterion']}/{best_params['max_depth']}/{best_params['n_estimators']}")

    # ── 8. Per-Class Metrics ──────────────────────────────────────────
    per_prec, per_rec, per_f1, per_sup = precision_recall_fscore_support(
        y_test, y_pred, labels=PERIOD_ORDER, average=None
    )
    per_class_df = pd.DataFrame({
        "Period": PERIOD_ORDER,
        "Precision": np.round(per_prec, 3),
        "Recall": np.round(per_rec, 3),
        "F1-Score": np.round(per_f1, 3),
        "Support": per_sup.astype(int),
    })
    print(f"\n  Per-Class F1:")
    for _, row in per_class_df.iterrows():
        print(f"    {row['Period']:20s}  P={row['Precision']:.3f}  R={row['Recall']:.3f}  F1={row['F1-Score']:.3f}  n={row['Support']}")

    # ── 9. Confusion Matrix ───────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=PERIOD_ORDER)
    cm_df = pd.DataFrame(cm, index=PERIOD_ORDER, columns=PERIOD_ORDER)
    print(f"\n  Confusion Matrix:")
    print(cm_df.to_string())

    # ── 10. Bootstrap 95% CI for Weighted F1 ─────────────────────────
    print(f"\nComputing {N_BOOTSTRAP}-iteration bootstrap CI...")
    boot_f1s = []
    rng = np.random.RandomState(SEED)
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    n_test = len(y_test_arr)

    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_test, size=n_test, replace=True)
        _, _, bf1, _ = precision_recall_fscore_support(
            y_test_arr[idx], y_pred_arr[idx], average="weighted"
        )
        boot_f1s.append(bf1)

    ci_lower = np.percentile(boot_f1s, 2.5)
    ci_upper = np.percentile(boot_f1s, 97.5)
    print(f"  95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # ── 11. Feature Importance ────────────────────────────────────────
    feature_names = vectorizer.get_feature_names_out()
    importances = best_rf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).nlargest(50)
    print(f"\n  Top 20 Features:")
    for name, imp in feat_imp.head(20).items():
        print(f"    {name:30s}  {imp:.5f}")

    # ── 12. Save All Results ──────────────────────────────────────────
    print(f"\nSaving results to {OUTPUT_DIR}/...")

    # Metrics text
    bp = best_params
    metrics_str = (
        f"Weighted Precision: {w_prec:.4f}, Weighted Recall: {w_rec:.4f}, "
        f"Weighted F1: {w_f1:.4f}, Macro F1: {macro_f1:.4f}, "
        f"Best params: {bp['criterion']}/{bp['max_depth']}/{bp['n_estimators']}"
    )
    with open(os.path.join(OUTPUT_DIR, "RF_Metrics.txt"), "w") as f:
        f.write(metrics_str)

    # Per-class CSV
    per_class_df.to_csv(os.path.join(OUTPUT_DIR, "RF_Final_PerClass.csv"), index=False)

    # Bootstrap CI text
    ci_str = f"Weighted F1: {w_f1:.4f}, 95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
    with open(os.path.join(OUTPUT_DIR, "RF_Final_Bootstrap_CI.txt"), "w") as f:
        f.write(ci_str)

    # Feature importance CSV
    feat_imp.to_csv(os.path.join(OUTPUT_DIR, "RF_Feature_Importance.csv"), header=["Importance"])

    # Confusion matrix CSV
    cm_df.to_csv(os.path.join(OUTPUT_DIR, "RF_Confusion_Matrix.csv"))

    # Full classification report
    report = classification_report(y_test, y_pred, labels=PERIOD_ORDER, digits=4)
    with open(os.path.join(OUTPUT_DIR, "RF_Classification_Report.txt"), "w") as f:
        f.write("Leak-Free Random Forest Classification Report\n")
        f.write(f"Sampling: n={min_n}/period, WITHOUT replacement, seed={SEED}\n")
        f.write(f"Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} stratified, seed={SEED}\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"Vocabulary size: {vocab_size:,}\n\n")
        f.write(report)
        f.write(f"\n95% Bootstrap CI (n={N_BOOTSTRAP}): [{ci_lower:.4f}, {ci_upper:.4f}]\n")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s. All results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
