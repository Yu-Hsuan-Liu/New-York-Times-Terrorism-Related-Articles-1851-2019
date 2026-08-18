# -*- coding: utf-8 -*-
"""
run_per_period_lda.py
---------------------
Per-period LDA optimisation using the already-cleaned updated parquet.

Inputs
------
  data/combined_processed_df_updated.parquet (repo-relative; see path block below)
    Column used: 'clean_lemmatized_phrased'   (stopwords already applied)
    Period column: 'label'  (e.g. '1851-1900', '20010911-2019', ...)

Logic mirrors run_dynamic_lda_optimization() from nyt_main_training.py:
  - Small  (<1 000 docs) : no_below=3,  K in 1-4, passes=50, chunksize=num_docs
  - Medium (1 000-9 999) : no_below=5,  K in 1-4, passes=20, chunksize=2 000
  - Large  (>=10 000)    : no_below=20, K in 1-4, passes=5,  chunksize=4 000
  - no_above=0.4 for all sizes
  - Early-stop when coherence stops increasing across the K sweep
  - LdaModel (single-core, not LdaMulticore) — avoids nested-multiprocessing on Windows
  - random_state=123, CoherenceModel c_v, processes=1

Outputs
-------
  results/Dynamic_LDA_Results_Updated.csv
    Columns: Period, Topic_ID, Keywords, Best_K, Coherence

Run with:
  /c/Users/tosea/anaconda3/python.exe run_per_period_lda.py
"""

import os
import sys
import warnings
import random
import pickle

import numpy as np
import pandas as pd

# Gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_SEED = 123
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
# --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
_os.makedirs(RESULTS_ROOT, exist_ok=True)
PARQUET_PATH = _os.path.join(DATA_ROOT, 'combined_processed_df_updated.parquet')
OUT_DIR      = RESULTS_ROOT
OUT_CSV      = os.path.join(OUT_DIR, 'Dynamic_LDA_Results_Updated.csv')

# Column in the parquet that holds the pre-cleaned text
# (same column Option A used for LDA — see rerun_lda_rf_option_a.py line 420)
ANALYSIS_COLUMN = 'clean_lemmatized_phrased'

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the updated parquet  (DO NOT re-apply stopwords — already done)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("Loading updated parquet...")
df = pd.read_parquet(PARQUET_PATH)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Validate required columns
for col in ('label', ANALYSIS_COLUMN):
    if col not in df.columns:
        raise ValueError(
            f"Required column '{col}' not found in parquet. "
            f"Available: {list(df.columns)}"
        )

# Drop rows where the text column is blank / NaN
before = len(df)
df = df[df[ANALYSIS_COLUMN].notna()]
df = df[df[ANALYSIS_COLUMN].str.strip() != ''].copy()
after = len(df)
print(f"  Dropped {before - after} empty/NaN rows.")

# Report period labels exactly as they appear in the parquet
periods = sorted(df['label'].unique())
print(f"\n  Period labels found ({len(periods)}):")
for p in periods:
    cnt = (df['label'] == p).sum()
    print(f"    '{p}': {cnt:,} docs")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-period LDA optimisation
#    Mirrors run_dynamic_lda_optimization() from nyt_main_training.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STARTING PER-PERIOD LDA OPTIMISATION")
print("=" * 80)

all_rows = []          # will accumulate CSV rows
results_summary = {}   # period -> list-of-topic-strings (for final print)

for period in periods:
    print("\n" + "=" * 80)
    print(f"Processing Period: {period}")

    subset  = df[df['label'] == period]
    docs    = subset[ANALYSIS_COLUMN].tolist()
    num_docs = len(docs)

    if num_docs < 10:
        print("  > SKIP: fewer than 10 documents.")
        continue

    # ── Corpus size profile (identical thresholds as nyt_main_training.py) ──
    if num_docs < 1000:
        my_no_below  = 3
        my_k_range   = range(1, 5)   # 1, 2, 3, 4
        my_passes    = 50
        my_chunksize = num_docs       # single chunk for tiny corpora
        dataset_type = "Small (High Precision Mode)"
    elif num_docs < 10000:
        my_no_below  = 5
        my_k_range   = range(1, 5)
        my_passes    = 20
        my_chunksize = 2000
        dataset_type = "Medium (Balanced Mode)"
    else:
        my_no_below  = 20
        my_k_range   = range(1, 5)
        my_passes    = 5
        my_chunksize = 4000
        dataset_type = "Large (High Efficiency Mode)"

    print(f"  > Profile : {dataset_type}")
    print(f"  > Docs    : {num_docs:,}  |  no_below={my_no_below}  "
          f"no_above=0.4  passes={my_passes}  chunksize={my_chunksize}")

    # ── Tokenise  (the text is already space-separated tokens) ──
    # NOTE: Do NOT re-apply the _ADDITIONAL_SW2_SET here — the updated parquet
    # column 'clean_lemmatized_phrased' was produced by rerun_lda_rf_option_a.py
    # which already applied check_additional_sw2 + handle_vio_lence_merger +
    # clean_garbage_text before saving the parquet.
    data_tokens = [str(sent).split() for sent in docs]

    # ── Gensim dictionary ──
    id2word = corpora.Dictionary(data_tokens)
    id2word.filter_extremes(no_below=my_no_below, no_above=0.4)
    corpus  = [id2word.doc2bow(text) for text in data_tokens]

    if len(id2word) == 0:
        print("  > SKIP: dictionary is empty after filter_extremes.")
        continue

    print(f"  > Dictionary size: {len(id2word):,} tokens")
    print(f"  > Searching Best K in {list(my_k_range)} ...", end="  ", flush=True)

    # ── K sweep with early-stop ──
    best_score = -1.0
    best_model = None
    best_k     = -1

    for k in my_k_range:
        try:
            model = LdaModel(
                corpus       = corpus,
                id2word      = id2word,
                num_topics   = k,
                random_state = GLOBAL_SEED,
                passes       = my_passes,
                chunksize    = my_chunksize,
                alpha        = 'auto',
                eta          = 'auto',
                eval_every   = None,   # disable perplexity log to keep output clean
            )
            cm    = CoherenceModel(
                model      = model,
                texts      = data_tokens,
                dictionary = id2word,
                coherence  = 'c_v',
                processes  = 1,        # single process — avoids Windows spawn issues
            )
            score = cm.get_coherence()

            if score > best_score:
                best_score = score
                best_model = model
                best_k     = k
                print(f"K={k}({score:.3f})↑", end="  ", flush=True)
            else:
                print(f"K={k}({score:.3f})↓ STOP")
                break   # early-stop: coherence did not improve

        except Exception as exc:
            print(f"[Error K={k}: {exc}]", end="  ", flush=True)
            continue

    print()  # newline after the inline K sweep printout

    # ── Record results ──
    if best_model is None:
        print(f"  > WARNING: No valid model found for period '{period}'.")
        continue

    print(f"  >>> Winner for {period}: K={best_k}  Coherence={best_score:.4f}")

    topic_strings = []
    for idx, topic_str in best_model.print_topics(-1):
        print(f"     T{idx}: {topic_str}")
        topic_strings.append(topic_str)
        all_rows.append({
            'Period'    : period,
            'Topic_ID'  : idx,
            'Keywords'  : topic_str,
            'Best_K'    : best_k,
            'Coherence' : round(best_score, 4),
        })

    results_summary[period] = {
        'best_k'    : best_k,
        'coherence' : round(best_score, 4),
        'topics'    : topic_strings,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. Save CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
if all_rows:
    results_df = pd.DataFrame(all_rows, columns=[
        'Period', 'Topic_ID', 'Keywords', 'Best_K', 'Coherence'
    ])
    results_df.to_csv(OUT_CSV, index=False)
    print(f"Results saved to: {OUT_CSV}")
    print(f"Total rows written: {len(results_df)}")
else:
    print("WARNING: No results were collected — CSV not written.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Clean summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: Per-Period LDA Optimisation Results")
print("=" * 80)

if results_summary:
    # Header
    col_w = [22, 7, 11]
    header = (
        f"{'Period':<{col_w[0]}}"
        f"{'Best_K':>{col_w[1]}}"
        f"{'Coherence':>{col_w[2]}}"
    )
    sep = "-" * (sum(col_w) + 2)
    print(header)
    print(sep)

    for p in periods:
        if p not in results_summary:
            print(f"{p:<{col_w[0]}}{'SKIPPED':>{col_w[1] + col_w[2] + 2}}")
            continue
        info = results_summary[p]
        print(
            f"{p:<{col_w[0]}}"
            f"{info['best_k']:>{col_w[1]}}"
            f"{info['coherence']:>{col_w[2]}.4f}"
        )

    print(sep)
    print()

    # Top-3 keywords per period for quick reading
    print("Top keywords per winning topic per period:")
    print(sep)
    for p in periods:
        if p not in results_summary:
            continue
        info   = results_summary[p]
        k      = info['best_k']
        topics = info['topics']
        print(f"\n  {p}  (K={k}, Coherence={info['coherence']:.4f})")
        for t_idx, t_str in enumerate(topics):
            # Extract just the words from the gensim topic string
            # Format: '0.123*"word" + 0.045*"other" + ...'
            words = [
                part.split('*')[-1].strip().strip('"').strip("'")
                for part in t_str.split('+')
            ]
            top5 = ', '.join(words[:5])
            print(f"    T{t_idx}: {top5}")
else:
    print("No results to display.")

print("\n" + "=" * 80)
print("DONE.")
print("=" * 80)
