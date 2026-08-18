# Analyzing Terrorism in Historical News: A Natural Language Processing Approach

Replication materials for the manuscript (Liu, Lo, Moton, and Mitnik). The study traces how the New York Times used the words "terrorism" and "terrorist" from 1851 to 2019, with per-period Word2Vec models and per-period LDA topic models over seven historical periods, plus a Random Forest period classifier as a supplementary check.

## Layout

`raw_data_processing/` holds the notebooks that collected and assembled the corpus (article-list scraping, full-text retrieval, OCR, merging); its own README walks through them in order. `data/` carries the article lists. `code/` holds the analysis scripts, `results/` their outputs, and `figures/` the figures used in the manuscript and supplement. Section 3.1 of the manuscript describes the text sources.

## Correction (August 2026)

Before publication we found that the corpus-building step in `nyt_main_training.py` had assigned 11,549 pre-1981 articles, whose digital texts the web scrape had also returned, to the 1981–2001 period; 9,480 of them were already in the corpus under their correct period through the OCR route. The script now keeps only 1981–2019 rows from the web-scrape file, and every result touching 1981–2001 was rerun with the same code, parameters, and seeds. The corrected corpus has 67,311 articles (16,921 in 1981–2001). The other six periods are unaffected. `CORRECTION_2026-08.md` lists what changed and by how much. The result files in this repository are the corrected ones.

## Data

`total_nyt_terrorism_news1850-2019.csv` is the full article list pulled from the newspaper's Terrorism topic index (177,195 records: date, title, abstract, link, author, desk). The analysis corpus is the subset whose full text contains "terrorism" or "terrorist" after the filters in `nyt_main_training.py`, 67,311 articles. `total_nyt_terrorism_news1850-1980_raw.csv` is the pre-1981 slice of that list.

The full-text files and the processed corpora exceed GitHub's file-size limit and are not included: `nyt_online_texts.csv` (427 MB), `NYT_text_1851_1980.csv` (108 MB), `nyt_terrorism_pdftexts_proquest2.csv` (100 MB), and `combined_processed_df.parquet` / `combined_processed_df_updated.parquet` (457 MB each). Contact the corresponding author for copies, or rebuild them with the collection notebooks; both text sources require subscription access.

## Rerunning the analysis

Scripts resolve paths relative to the repository: inputs from `data/`, outputs to `results/` and `figures/`. Set `NYT_NLP_DATA` or `NYT_NLP_RESULTS` to point elsewhere. Run from inside `code/`.

1. `nyt_main_training.py` builds the processed corpus from the three text files (cleaning, lemmatization, phrase detection), writes `combined_processed_df.parquet` into the data directory, and trains the per-period Word2Vec models (Table 3), the keyword-proximity checks, and the global K = 12 LDA. It imports helpers from `my_functions_NYT_gpu.py`. `rerun_lda_rf_option_a.py` re-applies the final stopword list to that parquet and writes `combined_processed_df_updated.parquet`; its Random Forest section is superseded by `rf_no_leakage.py`.
2. `run_per_period_lda.py` fits the per-period topic models and selects K by coherence (Tables 5–6). `run_lda_coherence.py` runs the K = 2–15 sweep on a balanced subsample and `run_normalized_lda.py` the period-normalized global model (Supplementary S1).
3. `run_temporal_alignment.py` retrains uniform CBOW models and computes Procrustes displacement across consecutive periods (Section 4.1; Table S8, Figure S5). `w2v_robustness_check.py` retrains all 14 models at a uniform min_count of 5 (Table S7). `w2v_klan_sensitivity.py` checks the Klan normalization in 1851–1900.
4. `task_a_bootstrap.py` and `task_a2_category_bootstrap.py` run the document-level bootstrap of neighbor ranks; `task_b_e_cooccurrence.py` produces the predicate co-occurrence and KWIC checks (Tables S9–S10). These use the pre-1980 periods only.
5. `rf_no_leakage.py` trains the leak-free Random Forest classifier with a 1,000-iteration bootstrap CI, and `run_rf_comparison.py` compares CountVectorizer with TF-IDF (Supplementary S2).
6. `make_w2v_visualization.py` (Figure 1), `make_visualization.py` (Figure S1), `generate_rf_figures.py` (Figures S2–S4) draw the figures from the files in `results/`.

Word2Vec models train with `workers=1` and `seed=123` and reproduce exactly on the same corpus; the per-period LDA and Random Forest runs are seeded as well. Requires Python 3.9+ with gensim, scikit-learn, spaCy (`en_core_web_md`), NLTK, pandas, numpy, scipy, matplotlib, seaborn, and pyarrow.

## License and citation

Code is released under the MIT License. Please cite the manuscript if you use these materials.
