# -*- coding: utf-8 -*-
"""
Compares CountVectorizer vs TF-IDF Random Forest (same grid, same split).
Also computes 1000-iteration bootstrap 95% CI on weighted F1 for both.
Saves to nlp_results/RF_Comparison.csv and per-vectorizer reports.
"""
import random, os, csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report

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
print(f'Periods: {periods}')
counts = {p: len(df[df[label_col]==p]) for p in periods}
min_n = min(counts.values())
print(f'Min n per period: {min_n}')

print(f'Sampling {min_n} docs per period...')
sampled = pd.concat([
    df[df[label_col] == p].sample(min_n, random_state=123, replace=False)
    for p in periods
], ignore_index=True)

X = sampled[text_col].astype(str)
y = sampled[label_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)
print(f'Train: {len(X_train)}, Test: {len(X_test)}')

param_grid = {
    'n_estimators': [10, 100],
    'max_depth':    [None, 1, 10],
    'criterion':    ['gini', 'entropy'],
}

results_summary = []

for vec_name, Vectorizer in [('CountVectorizer', CountVectorizer), ('TF-IDF', TfidfVectorizer)]:
    print(f'\n=== {vec_name} ===')
    vec = Vectorizer()
    X_train_v = vec.fit_transform(X_train)
    X_test_v  = vec.transform(X_test)

    grid = GridSearchCV(
        RandomForestClassifier(random_state=123),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid.fit(X_train_v, y_train)
    best_params = grid.best_params_
    best = grid.best_estimator_
    print(f'Best params: {best_params}')

    y_pred = best.predict(X_test_v)
    f1_w   = f1_score(y_test, y_pred, average='weighted')
    f1_mac = f1_score(y_test, y_pred, average='macro')
    print(f'Weighted F1: {f1_w:.4f}')
    print(classification_report(y_test, y_pred))

    # Bootstrap CI (1000 iterations)
    print(f'Bootstrap CI...')
    rng = np.random.default_rng(123)
    n = len(y_test)
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    boot_f1s = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        boot_f1s.append(f1_score(
            y_test_arr[idx], y_pred_arr[idx],
            average='weighted', zero_division=0
        ))
    ci_lo = np.percentile(boot_f1s, 2.5)
    ci_hi = np.percentile(boot_f1s, 97.5)
    print(f'95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]')

    results_summary.append({
        'Vectorizer':        vec_name,
        'Weighted_F1':       round(f1_w, 4),
        'Macro_F1':          round(f1_mac, 4),
        'CI_lower_95':       round(ci_lo, 4),
        'CI_upper_95':       round(ci_hi, 4),
        'Best_n_estimators': best_params['n_estimators'],
        'Best_max_depth':    str(best_params['max_depth']),
        'Best_criterion':    best_params['criterion'],
    })

    report_path = os.path.join(OUT_DIR, f'RF_{vec_name.replace("-","_")}_FullReport.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'RF with {vec_name}\n')
        f.write(f'Best params: {best_params}\n')
        f.write(f'Weighted F1: {f1_w:.4f}  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]\n\n')
        f.write(classification_report(y_test, y_pred))
    print(f'Saved: {report_path}')

# Save comparison
out_csv = os.path.join(OUT_DIR, 'RF_Comparison.csv')
pd.DataFrame(results_summary).to_csv(out_csv, index=False)
print(f'\nSaved: {out_csv}')
print('\n=== FINAL SUMMARY ===')
for r in results_summary:
    print(f"  {r['Vectorizer']:20s}  F1={r['Weighted_F1']:.4f}  "
          f"95%CI=[{r['CI_lower_95']:.4f},{r['CI_upper_95']:.4f}]  "
          f"best: {r['Best_n_estimators']}est / depth={r['Best_max_depth']} / {r['Best_criterion']}")
