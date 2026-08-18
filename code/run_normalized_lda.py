# -*- coding: utf-8 -*-
"""
Period-normalized LDA robustness check.
Samples min(n_per_period) docs from each period, fits LDA K=12,
compares topic proportions to the original global model.
Saves to nlp_results/LDA_Normalized_TopicProportions.csv
"""
import random, os
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

random.seed(123)
np.random.seed(123)

# --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
_os.makedirs(RESULTS_ROOT, exist_ok=True)
OUT_DIR = RESULTS_ROOT
os.makedirs(OUT_DIR, exist_ok=True)

print('Loading parquet...')
df = pd.read_parquet(_os.path.join(DATA_ROOT, 'combined_processed_df.parquet'))
df = df[df['clean_lemmatized_phrased'].notna()].copy()

label_col = 'label'
text_col  = 'clean_lemmatized_phrased'

periods = sorted(df[label_col].unique())
counts  = {p: len(df[df[label_col]==p]) for p in periods}
min_n   = min(counts.values())
print(f'Documents per period: {counts}')
print(f'Min n (normalization target): {min_n}')

print(f'Sampling {min_n} docs per period (no replacement)...')
sampled = pd.concat([
    df[df[label_col] == p].sample(min_n, random_state=123, replace=False)
    for p in periods
], ignore_index=True)
print(f'Total: {len(sampled)}')

texts        = [str(t).split() for t in sampled[text_col]]
period_labels = list(sampled[label_col])

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.5)
print(f'Dictionary size: {len(dictionary)}')
corpus = [dictionary.doc2bow(t) for t in texts]

print('Fitting normalized LDA (K=12, passes=20)...')
lda = LdaModel(
    corpus, num_topics=12, id2word=dictionary,
    passes=20, random_state=123
)

print('\nTopic keywords (normalized model):')
for i in range(12):
    words = [w for w, _ in lda.show_topic(i, topn=10)]
    print(f'  T{i}: {", ".join(words)}')

print('\nComputing topic proportions per period...')
doc_topics = []
for i, bow in enumerate(corpus):
    topic_dist = lda.get_document_topics(bow, minimum_probability=0)
    top_topic  = max(topic_dist, key=lambda x: x[1])[0]
    doc_topics.append({'period': period_labels[i], 'topic': top_topic})

topic_df = pd.DataFrame(doc_topics)
props = (topic_df.groupby(['period','topic'])
         .size().unstack(fill_value=0))
props_pct = (props.div(props.sum(axis=1), axis=0) * 100).round(1)

print('\nNormalized LDA Topic Proportions (%):')
print(props_pct.to_string())

# Save
out_csv = os.path.join(OUT_DIR, 'LDA_Normalized_TopicProportions.csv')
props_pct.to_csv(out_csv)

out_kw = os.path.join(OUT_DIR, 'LDA_Normalized_Keywords.txt')
with open(out_kw, 'w', encoding='utf-8') as f:
    f.write(f'Normalized LDA (K=12, n={min_n}/period, seed=123)\n\n')
    for i in range(12):
        words = [w for w, _ in lda.show_topic(i, topn=10)]
        f.write(f'T{i}: {", ".join(words)}\n')

print(f'\nSaved: {out_csv}')
print(f'Saved: {out_kw}')
