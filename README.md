# Analyzing Terrorism in Historical News: A Natural Language Processing Approach

Replication materials for the article in *Violence and Victims* (Liu, Lo, Moton, and Mitnik). The study traces how the New York Times used the words "terrorism" and "terrorist" from 1851 to 2019, with per-period Word2Vec and LDA models over seven historical periods and a Random Forest classifier as a supplementary check.

The notebooks in `raw_data_processing/` collected and assembled the corpus. `data/` carries the article lists, `code/` the analysis scripts, `results/` their outputs, and `figures/` the figures in the manuscript and supplement.

The full-text files are too large for GitHub and are not included. If you need the data, contact the corresponding author.

To rerun the analysis, work inside `code/`: `nyt_main_training.py` builds the processed corpus and trains the Word2Vec models, and the remaining scripts reproduce the individual tables, figures, and robustness checks. Requires Python 3.9+ with gensim, scikit-learn, spaCy (`en_core_web_md`), NLTK, pandas, numpy, scipy, matplotlib, and seaborn.

Code is released under the MIT License. Please cite the article if you use these materials.
