# -*- coding: utf-8 -*-
"""
TASK A supplement — category-level bootstrap for 1851-1900.
Same design as task_a_bootstrap.py (B=50, CBOW, small params, seed=123+b),
but records the FULL top-25 neighbor list per replicate for both targets,
then asks: in what share of replicates does ANY (and how many) racial/electoral-
violence or klan-variant token appear in the top-25?

Outputs -> reviewer_analyses/category_bootstrap_1851_1900.csv
           reviewer_analyses/category_bootstrap_summary.txt
"""
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
_os.makedirs(RESULTS_ROOT, exist_ok=True)
OUT = _os.path.join(RESULTS_ROOT, '')
PARQUET = _os.path.join(DATA_ROOT, 'combined_processed_df_updated.parquet')
B = 50
PARAMS = dict(vector_size=50, window=3, epochs=50, min_count=3)

RACIAL_ELECTORAL = {"fair_election","intimidation_violence","practical_disfranchisement",
    "disfranchisement","night_rider","negro","negroes","freedmen","freedman",
    "election_fraud","georgia_alabama","alabama_arkansas","louisiana_mississippi",
    "florida_louisiana","colored_voter","colored_vote","bulldozing","white_league",
    "jackson_miss","legislature_elect","riot_tumult","election_outrage","ballot_box",
    "free_fair","arkansas","mississippi","louisiana","intimidation","disfranchise",
    "election_fraud_intimidation","fraud_intimidation","fraud_violence"}
STATE_TERROR = {"repression","brute_force","reign_terror","despotism","tyranny","oppression"}

def is_klan(tok): return ("klan" in tok) or ("klux" in tok)

df = pd.read_parquet(PARQUET, columns=["label","clean_lemmatized_phrased"])
sub = df[df.label == "1851-1900"].reset_index(drop=True)
n = len(sub)
rows = []
for b in range(B):
    boot = sub.sample(n=n, replace=True, random_state=123 + b)
    sents = boot.clean_lemmatized_phrased.str.split().tolist()
    model = Word2Vec(sentences=sents, sg=0, workers=4, seed=123 + b, **PARAMS)
    wv = model.wv
    for target in ["terrorism","terrorist"]:
        if target not in wv.key_to_index:
            rows.append(dict(Replicate=b, Target=target, Top25="TARGET ABSENT")); continue
        top25 = [w for w,_ in wv.most_similar(target, topn=25)]
        rows.append(dict(Replicate=b, Target=target, Top25=", ".join(top25),
                         n_klan=sum(is_klan(w) for w in top25),
                         n_racial_electoral=sum(w in RACIAL_ELECTORAL for w in top25),
                         n_state_terror=sum(w in STATE_TERROR for w in top25)))
    if (b+1) % 10 == 0:
        print(f"replicate {b+1}/{B}", flush=True)

d = pd.DataFrame(rows)
d.to_csv(OUT+"category_bootstrap_1851_1900.csv", index=False)
lines = []
for target in ["terrorism","terrorist"]:
    g = d[(d.Target==target) & (d.Top25!="TARGET ABSENT")]
    lines.append(f"\nTarget '{target}' ({len(g)}/{B} replicates with target in vocab):")
    lines.append(f"  any klan-variant in top-25:          {(g.n_klan>0).mean():.2f}")
    lines.append(f"  any racial/electoral token in top25: {(g.n_racial_electoral>0).mean():.2f}")
    lines.append(f"  mean # racial/electoral in top-25:   {g.n_racial_electoral.mean():.2f}")
    lines.append(f"  any state-terror token in top-25:    {(g.n_state_terror>0).mean():.2f}")
    lines.append(f"  mean # state-terror in top-25:       {g.n_state_terror.mean():.2f}")
txt = "\n".join(lines)
with open(OUT+"category_bootstrap_summary.txt","w",encoding="utf-8") as f:
    f.write("Category-level bootstrap, 1851-1900, CBOW small params, B=50\n"+txt+"\n")
print(txt)
print("\nSupplement done.")
