# -*- coding: utf-8 -*-
"""
TASK B — Predicate-level co-occurrence (labeling validation)
TASK E — (1) KWIC for 'bulgar' 1901-1930; (2) 'terrorist' token counts per period

Uses the raw pre-1980 CSV (D:/Jupyter/NYT_API/NYT_text_1851_1980.csv) restricted
to the exact analysis corpus (same filters as nyt_main_training.py -> 15,946 rows,
verified to align 1:1 with the parquet's pre-1980 rows).
Outputs -> reviewer_analyses/
"""
import re
import numpy as np
import pandas as pd

# --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
_os.makedirs(RESULTS_ROOT, exist_ok=True)
OUT = _os.path.join(RESULTS_ROOT, '')
RAW = _os.path.join(DATA_ROOT, 'NYT_text_1851_1980.csv')
PARQUET = _os.path.join(DATA_ROOT, 'combined_processed_df_updated.parquet')

# ---------------- reproduce analysis corpus ----------------
not_contain = ["The Screen:","Book Review","The Kong and I","TV:","TV ","THE SCREEN","Screen","STAGE VIEW","Publishing:","Public and Private Games","Paperback","ON TELEVISION","News of the Theater","IN AND OUT OF BOOKS","Editors' Choice","Books: ","Books--Authors","Books of","Books and Authors","Books -- Authors","BOOK NOTES","BOOK ENDS","Best Sellers","BEHIND THE BEST SELLERS","At the Movies","Arts and Leisure Guide","About Real Estate","Advertising","Answers to Weekly Quiz","Answers to Quiz","A Listing of","TV","BOOKS OF THE TIMES","THE PLAY","MUSIC VIEW", "Books of The","GOING OUT Guide", "Movies", "Paper backs","Paperback Best Sellers", "Paperbacks","Westchester/This Week","Long Island/ This Week","Television This Week","New Jersey/This Week", "Weekly News Quiz","The Screen;", "What's in a Book Name?", "When Is a Movie So Bad","Sports Editor's Mailbox: The Animals at Shea/Scoreboard at Yankee Stadium","Sports News Briefs","Sports World Special","Sports of The Times","Sports of the Times","Stage: ", "A preview OF FALL BOOKS; ", "Books Authors; ","NEW PUBLICATONS", "NEW PUBLICATIONS", "New Books","OF MANY THINGS: ","PREVIEW of FALL BOOKS", "Movie Mailbag", "NEW BOOKS","Passionate Story of a Bandit:'Augusto Matraga' Is at 5th Avenue Cinema Movie From Brazil by Santos Arrives","Action Movie Set in Latin America", "M-G-M FILLS ROLES IN TWO NEW FILMS;","His Latest Volume of Collected Plays", "Musical Cartoon to Play At the Brooklyn Academy","STUDENTS TO GIVE PLAYS; 4 Municipal Colleges to Present Drama Festival May 12","The Theater: A Musical","'Minnie' on Music -- and Rain","Amid Trials of Croatian Nationalists, a Satirical Musical Comedy in Zagreb Evokes Laughter Through Tears","THE STADIUM SEASON; Management Overcomes Many Difficulties -- Need of Music in Wartime","Drama: Adapting Wiesel's", "HITCHCOCK: MASTER MELODRAMATIST", "Terrorism Is Drama","They Weren't Riotous Comedies", "Sports In America","'Mary Burns, Fugitive,' the Melodrama of a Girl Who Loved a Murderer, at the Paramount.","'Thunderbolt,' Entebbe Raid In Israeli Film","3 1/2-Hour Film Based on Uris' Novel Opens", "Screen: ","6 Film Studio's Vie Over Entebbe Raid", "STAGE VIEW","CHARGES TERRORISM IN FILM INDUSTRY; Head of New Jersey","FILM CONTROVERSY", "FILM VIEW", "Film Festival: ", "Film Fete:","Film:", "Kennedy Center Drops Disputed Film", "SHOP TALK","Shouldn't Suspense Films Do More Than Just Kill Time", "STAMPS","'Viva Italia!,' Starring Vittorio Gassman, Returns to Day of Separate-Sketch Film:The Cast"]
excluding_index_list = [15424, 11867, 11874, 13516, 14336, 18744, 19544, 19603, 20110,20346, 20670, 21318, 21508, 21610, 21649, 21659, 21927, 22122, 22216]

df = pd.read_csv(RAW, encoding="ISO-8859-1")
df = df[~(df.title.str.contains('|'.join(not_contain), na=False))]
df = df.drop(excluding_index_list, errors='ignore')
df = df[(df.text.str.contains("terrorist", na=False) | df.text.str.contains("terrorism", na=False))]
conditions = [((df["year"]>=1851)&(df["year"]<=1900)),((df["year"]>=1901)&(df["year"]<=1930)),
              ((df["year"]>=1931)&(df["year"]<=1950)),((df["year"]>=1951)&(df["year"]<=1960)),
              ((df["year"]>=1961)&(df["year"]<=1980))]
values = ['1851-1900','1901-1930','1931-1950','1951-1960','1961-1980']
df['label'] = np.select(conditions, values, default='Unknown')
df = df.reset_index(drop=True)
assert len(df) == 15946, len(df)
print(f"Analysis corpus reproduced: {len(df)} articles")

def datestr(row):
    for c in ('date','pubdate_x','pubdate_y'):
        v = row.get(c)
        if isinstance(v,str) and v.strip():
            return v.strip()
    y,m,d = row.get('year'), row.get('month'), row.get('day')
    try: return f"{int(y)}-{int(m):02d}-{int(d):02d}"
    except Exception: return str(y)

# ---------------- TASK B ----------------
TERROR = re.compile(r'terroris', re.I)

def actor_token_idx(tokens, case):
    """Return token indices matching the actor for each case."""
    idx = []
    lt = [t.lower() for t in tokens]
    if case == 'klan':
        for i,t in enumerate(lt):
            if 'klux' in t or 'kuklux' in t or re.match(r'k+lans?\w*', t) or t.startswith('klan'):
                idx.append(i)
    elif case == 'malaya':
        for i,t in enumerate(lt):
            if 'malaya' in t:
                idx.append(i)
    elif case == 'maumau':
        for i,t in enumerate(lt):
            if t.startswith('mau'):
                if 'mau-mau' in t or 'maumau' in t:
                    idx.append(i)
                elif i+1 < len(lt) and lt[i+1].startswith('mau'):
                    idx.append(i)
    return idx

ACTOR_RE = {
    # broad net: OCR renders Ku Klux as 'U Klux', 'Kukluxism', 'Kuklux_', etc.
    'klan':   re.compile(r'klux|klan', re.I),
    'malaya': re.compile(r'malaya', re.I),
    'maumau': re.compile(r'mau[\s\-]mau', re.I),
}

CASES = [
    ("Klan 1851-1900",      'klan',   df[df.label=='1851-1900']),
    ("Klan 1931-1950",      'klan',   df[df.label=='1931-1950']),
    ("Malaya 1931-1950",    'malaya', df[df.label=='1931-1950']),
    ("Malaya 1948-1950",    'malaya', df[(df.label=='1931-1950') & (df.year>=1948)]),
    ("Mau Mau 1951-1960",   'maumau', df[df.label=='1951-1960']),
]

rows = []
snippet_out = []
for name, case, sub in CASES:
    pat = ACTOR_RE[case]
    mention = sub[sub.text.str.contains(pat, na=False)]
    n_mention = len(mention)
    n_para = 0; n_para_fallback_used = 0; n_w15 = 0
    has_breaks = 0
    snippets = []
    for _, row in mention.iterrows():
        text = row['text']
        tokens = text.split()
        aidx = actor_token_idx(tokens, case)
        tidx = [i for i,t in enumerate(tokens) if TERROR.search(t)]
        # (2) same paragraph, fallback +/-50 words
        para_hit = False
        if '\n\n' in text:
            has_breaks += 1
            for para in text.split('\n\n'):
                if pat.search(para) and TERROR.search(para):
                    para_hit = True; break
        else:
            n_para_fallback_used += 1
            for a in aidx:
                if any(abs(a-t) <= 50 for t in tidx):
                    para_hit = True; break
        if para_hit: n_para += 1
        # (3) +/-15 words
        w15 = any(abs(a-t) <= 15 for a in aidx for t in tidx)
        if w15:
            n_w15 += 1
            if len(snippets) < 5:
                a, t = min(((a,t) for a in aidx for t in tidx if abs(a-t)<=15),
                           key=lambda p: abs(p[0]-p[1]))
                lo = max(0, min(a,t)-22); hi = min(len(tokens), max(a,t)+23)
                snip = ' '.join(tokens[lo:hi]).replace('\n',' ')
                snippets.append(f"[{datestr(row)}] ...{snip}...")
    rows.append(dict(Case=name, Articles_mentioning_actor=n_mention,
                     Cooccur_paragraph_or_50w=n_para,
                     Cooccur_within_15_words=n_w15,
                     Articles_with_para_breaks=has_breaks,
                     Fallback_50w_window_used=n_para_fallback_used,
                     Corpus_articles_in_period=len(sub)))
    snippet_out.append(f"\n{'='*90}\n{name}: {n_mention} actor articles; "
                       f"{n_para} same-paragraph(/50w) co-occurrence; {n_w15} within 15 words\n{'='*90}")
    snippet_out.extend(snippets if snippets else ["  (no within-15-word snippets)"])

pd.DataFrame(rows).to_csv(OUT+"predicate_cooccurrence.csv", index=False)
with open(OUT+"predicate_cooccurrence_snippets.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(snippet_out))
print(pd.DataFrame(rows).to_string(index=False))

# ---------------- TASK E1: KWIC 'bulgar' ----------------
kwic_lines = []
sub = df[df.label=='1901-1930']
b_re = re.compile(r'\bbulgar\w*', re.I)
n_art = 0
for _, row in sub.iterrows():
    text = row['text']
    if not b_re.search(text): continue
    n_art += 1
    tokens = text.split()
    for i,t in enumerate(tokens):
        if b_re.match(t):
            lo, hi = max(0,i-12), min(len(tokens), i+13)
            kwic_lines.append(f"[{datestr(row)}] ...{' '.join(tokens[lo:hi])}...")
            break   # one line per article for diversity
    if len(kwic_lines) >= 10: break
with open(OUT+"kwic_bulgar_1901_1930.txt", "w", encoding="utf-8") as f:
    f.write(f"'bulgar*' KWIC, 1901-1930 analysis corpus (articles containing bulgar*: shown {len(kwic_lines)})\n\n")
    f.write("\n\n".join(kwic_lines))
print(f"\nKWIC bulgar: wrote {len(kwic_lines)} lines")

# total articles containing bulgar* in period
n_bulgar_articles = sub.text.str.contains(b_re, na=False).sum()
print("1901-1930 articles containing bulgar*:", n_bulgar_articles)

# ---------------- TASK E2: 'terrorist' token counts ----------------
# raw counts pre-1980
e2 = []
for lab in values:
    s = df[df.label==lab]
    t_sing = s.text.str.count(r'(?i)\bterrorist\b').sum()
    t_plur = s.text.str.count(r'(?i)\bterrorists\b').sum()
    t_ism  = s.text.str.count(r'(?i)\bterrorism\b').sum()
    e2.append(dict(Period=lab, Source='raw_text', terrorist=int(t_sing), terrorists=int(t_plur),
                   terrorism=int(t_ism), n_articles=len(s)))
# processed counts (what the models actually see) — all 7 periods
pq = pd.read_parquet(PARQUET, columns=['label','clean_lemmatized_phrased'])
for lab in ['1851-1900','1901-1930','1931-1950','1951-1960','1961-1980','1981-20010910','20010911-2019']:
    s = pq[pq.label==lab]['clean_lemmatized_phrased']
    tok_t = s.str.count(r'\bterrorist\b').sum()
    tok_ism = s.str.count(r'\bterrorism\b').sum()
    docs_t = (s.str.contains(r'\bterrorist\b')).sum()
    e2.append(dict(Period=lab, Source='processed_tokens', terrorist=int(tok_t), terrorists=np.nan,
                   terrorism=int(tok_ism), n_articles=len(s), docs_containing_terrorist=int(docs_t)))
e2df = pd.DataFrame(e2)
e2df.to_csv(OUT+"terrorist_token_counts.csv", index=False)
print(e2df.to_string(index=False))
print("\nDone. Outputs written to reviewer_analyses/")
