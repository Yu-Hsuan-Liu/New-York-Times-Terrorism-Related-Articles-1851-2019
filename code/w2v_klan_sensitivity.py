# -*- coding: utf-8 -*-
"""
Sensitivity check for KKK normalization in Word2Vec.
Runs W2V on 1851-1900 corpus:
  Version A - with 'klan' token (current, normalized)
  Version B - with 'klan' token removed entirely
Reports top-25 neighbors of 'terrorism' and 'terrorist' for both versions.
If KKK-related vocabulary (night_rider, freedmen, reign_terror, etc.) still
appears in Version B, the finding is robust independent of normalization.
"""
import pandas as pd
from gensim.models import Word2Vec

# --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
_os.makedirs(RESULTS_ROOT, exist_ok=True)
PARQUET = _os.path.join(DATA_ROOT, 'combined_processed_df_updated.parquet')

print("Loading parquet...")
df = pd.read_parquet(PARQUET)
df_early = df[df['label'] == '1851-1900'].copy()
print(f"  1851-1900 documents: {len(df_early)}")

def parse_tokens(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    return text.split()

corpus_all   = [parse_tokens(t) for t in df_early['clean_lemmatized_phrased']]
corpus_noklan = [[tok for tok in doc if tok != 'klan'] for doc in corpus_all]

klan_count = sum(doc.count('klan') for doc in corpus_all)
docs_with_klan = sum(1 for doc in corpus_all if 'klan' in doc)
print(f"  'klan' token occurrences (normalized): {klan_count}")
print(f"  Documents containing 'klan':           {docs_with_klan}")

# SkipGram params for small corpus (< 1000 docs)
params = dict(vector_size=50, window=3, min_count=3, workers=1, seed=123,
              sg=1, epochs=50)

for label, corpus in [('VERSION A — With klan', corpus_all),
                      ('VERSION B — Without klan', corpus_noklan)]:
    print(f'\n{"="*60}')
    print(label)
    print('='*60)
    model = Word2Vec(corpus, **params)
    for target in ['terrorism', 'terrorist']:
        if target in model.wv:
            neighbors = model.wv.most_similar(target, topn=25)
            print(f"\n  Top 25 neighbors of '{target}':")
            for rank, (word, score) in enumerate(neighbors, 1):
                print(f"    {rank:2d}. {word:<35s} {score:.4f}")
        else:
            print(f"  '{target}' NOT in vocabulary")

print('\nSensitivity check complete.')
