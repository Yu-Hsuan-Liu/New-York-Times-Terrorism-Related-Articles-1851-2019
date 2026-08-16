# Reproducing "Analyzing Terrorism in Historical News: A Natural Language Processing Approach"

This repository holds the code, raw data, and result files behind the paper's analysis of
**78,860 New York Times articles (1851–2019)** that contain "terrorism" or "terrorist".
Two methods are run over seven historical periods: **Word2Vec** semantic modeling and
**per-period LDA** topic modeling, with a supplementary **Random Forest** period classifier.

This document maps every claim in the paper to the script and data file that produces it, so a
reviewer can re-run or spot-check any result.

---

## 1. Requirements

- Python 3.9+
- `gensim` (Word2Vec, `Phrases`, `LdaModel`/`LdaMulticore`, `CoherenceModel`)
- `spacy` + the `en_core_web_md` model (`python -m spacy download en_core_web_md`)
- `nltk` (stopwords corpus)
- `pandas`, `numpy`, `scipy` (Procrustes alignment)
- `scikit-learn` (RandomForest, CountVectorizer, TfidfVectorizer, GridSearchCV)
- `matplotlib` (figures)

A fixed random seed (`123`) is set throughout. Note: the main Word2Vec models train with
multiple worker threads, so neighbor rankings are reproducible up to minor multi-thread
nondeterminism; the Procrustes and robustness scripts use `workers=1`.

---

## 2. Run order

```
raw data  →  cleaning/preprocessing  →  Word2Vec  →  LDA  →  Procrustes + robustness  →  Random Forest
```

`nyt_main_training.py` is the master pipeline (preprocessing → Word2Vec → LDA); it imports helpers
from `my_functions_NYT_gpu.py`. The `run_*.py` scripts reproduce individual supplementary analyses.

---

## 3. Raw data (`data/`)

| File | What it is |
|---|---|
| `total_nyt_terrorism_news1850-2019.csv` | Full scrape metadata (title, pubdate, link) |
| `NYT_text_1851_1980.csv` | OCR full text, pre-1980 (NYT TimesMachine) |
| `nyt_online_texts.csv` | Online full text (post-1980) |
| `total_nyt_terrorism_news1850-1980_raw.csv` | Raw pre-1980 set |

The corpus is filtered to articles whose full text contains "terrorism" or "terrorist"
(78,860 articles; ~60.4M raw tokens, ~22M after cleaning).

---

## 4. Cleaning / preprocessing

- **`my_functions_NYT_gpu.py`** — preprocessing library:
  - `clean_text` — strip non-alphanumeric characters, lowercase
  - `rem_sw` — remove NLTK English stopwords + a domain-specific list
  - `handle_klan_merger` — fold Ku Klux Klan spelling variants into the single token `klan`
  - `check_dictionary` — spaCy `en_core_web_md`; keep in-vocabulary, alphabetic tokens; lemmatize
  - Gensim `Phrases` (`min_count=3`, `threshold=10`) — multi-word expressions
- **`nyt_main_training.py`** (preprocessing section) — applies the above and builds the processed corpus.

---

## 5. Word2Vec  → Table 3, Figure 1, §4.1, §5.2

| Script | Produces | Paper element |
|---|---|---|
| `nyt_main_training.py` (Word2Vec section) | `FINAL_ALL_MODELS_RESULTS.csv` | Table 3 neighbor ranks; CBOW + Skip-Gram per period; tiered hyperparameters; `seed=123` |
| `run_temporal_alignment.py` | `Procrustes_Displacement_Results.csv`, `Procrustes_Alignment_Summary.txt` | Temporal Stability Check (0.619 / 0.322 / 0.457 / 0.153); Table S8, Figure S5 |
| `w2v_robustness_check.py` | `W2V_Robustness_Comparison.csv`, `W2V_Robustness_Summary.txt` | Uniform min_count=5 robustness / Jaccard overlap; Table S7 |
| `w2v_klan_sensitivity.py` | — | Klan sensitivity checks |
| `make_w2v_visualization.py` | figure | Figure 1 (semantic-proximity heatmap) |

**Hyperparameter tiers (by corpus size):** small (<1,000 docs) `vector_size=50, window=3, epochs=50, min_count=3`;
medium (1,000–9,999) `100 / 5 / 20 / 5`; large (≥10,000) `200 / 5 / 10 / 20`.

---

## 6. LDA  → Tables 5, 6, §4.2

| Script | Produces | Paper element |
|---|---|---|
| `run_per_period_lda.py` | `Dynamic_LDA_Results_Updated.csv`, `LDA_Topic_Keywords.csv` | Table 6 per-period topic keywords; K ∈ {1,2,3,4} by c_v coherence |
| `run_lda_coherence.py` | `LDA_Coherence_Scores.csv` | Table 5 K* and coherence values |
| `run_normalized_lda.py` | `LDA_Normalized_TopicProportions.csv` | Normalized topic proportions |
| `nyt_main_training.py` (LDA section) | — | Global K=12 model (Supplementary S1–S3) |

Per-period training profiles: small `no_below=3`, 50 passes; medium `no_below=5`, 20 passes;
large `no_below=20`, 5 passes; `no_above=0.4`, `random_state=123`.

---

## 7. Random Forest (supplementary, §S4–S5)

Use the **leak-free** version.

| Script | Produces | Paper element |
|---|---|---|
| `rf_no_leakage.py` | leak-free RF model | Corrected period classifier |
| `run_rf_bootstrap_main.py` | `RF_Final_Bootstrap_CI.txt` | Weighted F1 = 0.666, 95% CI [0.635, 0.699]; 1,000 bootstrap iterations |
| `run_rf_comparison.py` | `RF_Comparison.csv` | CountVectorizer (F1 0.663) / TF-IDF (F1 0.687) comparison |
| `rerun_lda_rf_option_a.py`, `generate_rf_figures.py` | — | Combined rerun, figures |

---

## 8. Result files to check the paper against (`results/`)

- **Word2Vec ranks:** `FINAL_ALL_MODELS_RESULTS.csv` (columns: Neighbor_Word, Similarity, Target_Word, Period, Model_Type, Rank; use `Model_Type = CBOW` for the main-text ranks)
- **LDA:** `Dynamic_LDA_Results_Updated.csv`, `LDA_Topic_Keywords.csv`, `LDA_Coherence_Scores.csv`
- **Procrustes:** `Procrustes_Displacement_Results.csv`, `Procrustes_Alignment_Summary.txt`
- **Robustness:** `W2V_Robustness_Comparison.csv`, `W2V_Robustness_Summary.txt`
- **Random Forest:** `RF_Metrics.txt`, `RF_Final_Bootstrap_CI.txt`, `RF_Final_PerClass.csv`, `RF_Comparison.csv`

---

## 9. Known caveats for reproduction

1. **Processed corpus not included.** The fully cleaned corpus the models train on
   (`combined_processed_df.parquet`) is not in this release; it is regenerated by running the
   Section 4 cleaning pipeline end-to-end. Table 2's per-period article counts (summing to 78,860)
   are computed from that processed corpus.
2. **Data access.** Pre-1980 text was obtained via NYT TimesMachine (OCR) and 1981–2019 via
   ProQuest Historical Newspapers; both normally require institutional subscriptions.
3. **Source break at 1980/1981.** Pre-1980 text is OCR'd (TimesMachine); post-1980 is clean digital
   text (ProQuest). Comparisons across that boundary are partly confounded with this change in
   text-acquisition method (discussed in the paper's Limitations).
