# -*- coding: utf-8 -*-
"""
TASK A — Bootstrap rank stability for headline Word2Vec tokens.

B=50 document-level bootstrap resamples per period (with replacement, same n).
CBOW with the period's ORIGINAL hyperparameters (Section 3.3), seed = 123+replicate.
For each replicate, exact cosine-similarity rank of each headline token w.r.t.
'terrorism' / 'terrorist' (rank over full vocab, target excluded; NA if absent).

Data: D:/Jupyter/NYT_API/combined_processed_df_updated.parquet
Outputs -> reviewer_analyses/bootstrap_ranks.csv (long, per replicate)
           reviewer_analyses/bootstrap_ranks_summary.csv
           reviewer_analyses/bootstrap_summary.txt
"""
import time
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

HEADLINE = {
    "1851-1900": {
        "terrorism": ["klan","fair_election","intimidation_violence","georgia_alabama",
                      "alabama_arkansas","practical_disfranchisement"],
        "terrorist": ["repression","brute_force","reign_terror"],
    },
    "1901-1930": {
        "terrorism": ["bulgar","social_revolutionist","socialist_revolutionary","assassination"],
        "terrorist": ["night_rider"],
    },
    "1931-1950": {
        "terrorism": ["nazis","bombing","jews","reich","zionist","malaya","klan",
                      "holy_war","bombay_india"],
        "terrorist": [],
    },
    "1951-1960": {
        "terrorism": ["kill_injure","slay","cameroon","north_africans","mau_mau",
                      "counterterrorist","antiterrorist_drive","jakarta_indonesia"],
        "terrorist": ["south_vietnam"],
    },
    "1961-1980": {   # no reviewer-specified headline tokens; track top manuscript terms
        "terrorism": [],
        "terrorist": [],
    },
}
PERIODS = ["1851-1900","1901-1930","1931-1950","1951-1960","1961-1980"]

def get_params(n):
    if n < 1000:   return dict(vector_size=50,  window=3, epochs=50, min_count=3)
    if n < 10000:  return dict(vector_size=100, window=5, epochs=20, min_count=5)
    return dict(vector_size=200, window=5, epochs=10, min_count=20)

def token_rank(wv, target, token):
    """1-based rank of token in target's neighbor list (target excluded)."""
    if target not in wv.key_to_index or token not in wv.key_to_index:
        return np.nan
    v = wv.get_vector(target, norm=True)
    sims = wv.get_normed_vectors() @ v
    st = sims[wv.key_to_index[token]]
    return int((sims > st).sum())  # includes target(sim=1) => equals most_similar position

def main():
    t0 = time.time()
    df = pd.read_parquet(PARQUET, columns=["label","clean_lemmatized_phrased"])
    records = []

    for period in PERIODS:
        tokens_map = HEADLINE[period]
        all_tokens = [(tg, tk) for tg, tks in tokens_map.items() for tk in tks]
        if not all_tokens:
            print(f"\n[{period}] no headline tokens specified — skipping bootstrap for this period.")
            continue
        sub = df[df.label == period].reset_index(drop=True)
        n = len(sub)
        params = get_params(n)
        print(f"\n[{period}] n={n}, params={params}, B={B}", flush=True)

        for b in range(B):
            tb = time.time()
            boot = sub.sample(n=n, replace=True, random_state=123 + b)
            sents = boot.clean_lemmatized_phrased.str.split().tolist()
            model = Word2Vec(sentences=sents, sg=0, workers=4, seed=123 + b, **params)
            wv = model.wv
            for tg, tk in all_tokens:
                records.append(dict(Period=period, Replicate=b, Target=tg, Token=tk,
                                    Rank=token_rank(wv, tg, tk),
                                    Target_in_vocab=tg in wv.key_to_index,
                                    Vocab_size=len(wv.key_to_index)))
            if (b+1) % 5 == 0:
                print(f"  replicate {b+1}/{B} done ({time.time()-tb:.1f}s/rep)", flush=True)
                # incremental save
                pd.DataFrame(records).to_csv(OUT+"bootstrap_ranks.csv", index=False)

    long_df = pd.DataFrame(records)
    long_df.to_csv(OUT+"bootstrap_ranks.csv", index=False)

    # -------- summary --------
    rows = []
    for (period, tg, tk), g in long_df.groupby(["Period","Target","Token"]):
        r = g.Rank
        nn = r.notna()
        rows.append(dict(
            Period=period, Target=tg, Token=tk, B=len(g),
            n_in_vocab=int(nn.sum()),
            share_top25=float((r <= 25).sum() / len(g)),
            share_top100=float((r <= 100).sum() / len(g)),
            median_rank=float(r.median()) if nn.any() else np.nan,
            IQR_low=float(r.quantile(.25)) if nn.any() else np.nan,
            IQR_high=float(r.quantile(.75)) if nn.any() else np.nan,
            min_rank=float(r.min()) if nn.any() else np.nan,
            max_rank=float(r.max()) if nn.any() else np.nan,
        ))
    summ = pd.DataFrame(rows).sort_values(["Period","Target","median_rank"])
    summ.to_csv(OUT+"bootstrap_ranks_summary.csv", index=False)

    with open(OUT+"bootstrap_summary.txt","w",encoding="utf-8") as f:
        f.write(f"TASK A — Bootstrap rank stability (B={B}, CBOW, original per-period hyperparameters,\n"
                f"document resampling with replacement, seed=123+replicate, workers=4)\n"
                f"Rank = exact position in cosine-neighbor list of target (NA if token below min_count in replicate).\n"
                f"share_top25/share_top100 treat NA (absent from vocab) as NOT in top-k.\n\n")
        f.write(summ.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        f.write(f"\n\nTotal runtime: {(time.time()-t0)/60:.1f} min\n")
    print(summ.to_string(index=False))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
