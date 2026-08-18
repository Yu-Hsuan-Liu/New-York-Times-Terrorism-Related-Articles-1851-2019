# Correction record (August 2026)

During preparation of this repository, we found that `nyt_main_training.py` labeled every row of the web-scrape text file by index position (`index >= 56985` → 2001–2019, else 1981–2001) without first dropping the pre-1981 rows that the scrape had also collected (master-list index < 22855). As a result 11,549 pre-1981 articles (9,220 from the 1970s) were placed in the 1981–2001 period, 9,480 of them duplicating articles already present under their true period via the OCR corpus. The fix is the `index >= 22855` filter now in the script. Everything below compares the accepted-manuscript values (v95) with the corrected rerun; the six other periods are unchanged.


## Table 2 — articles and tokens per period

| Period | Articles v95 | Articles new | Tokens v95 | Tokens new |
|---|---|---|---|---|
| 1851-1900 | 597 | 597 | 331,382 | 331,382 |
| 1901-1930 | 1,133 | 1,133 | 370,159 | 370,159 |
| 1931-1950 | 2,831 | 2,831 | 600,685 | 600,685 |
| 1951-1960 | 1,505 | 1,505 | 239,942 | 239,942 |
| 1961-1980 | 9,880 | 9,880 | 2,161,011 | 2,161,011 |
| 1981-20010910 | 28,470 | 16,921 | 7,765,606 | 4,716,898 |
| 20010911-2019 | 34,444 | 34,444 | 10,521,591 | 10,521,591 |
| TOTAL | 78,860 | 67,311 | 21,990,376 | 18,941,668 |

## Table 3 — Word2Vec CBOW neighbors (only 1981–2001 changes; other periods identical or within run-to-run noise)

**terrorism, tokens named in Table 3:** international_terrorism 1→1; sponsor_terrorism 5→4; global_terrorism 22→21

| Rank | terrorism v95 | terrorism new |
|---|---|---|
| 1 | international_terrorism | international_terrorism |
| 2 | terrorist_activity | terror |
| 3 | terror | terrorist_activity |
| 4 | act_terrorism | sponsor_terrorism |
| 5 | sponsor_terrorism | act_terrorism |
| 6 | subversion | combat_terrorism |
| 7 | subversion_terrorism | terrorist_attack |
| 8 | urban_terrorism | fight_terrorism |
| 9 | violence | subversion |
| 10 | combat_terrorism | terrorist |
| 11 | act_violence | terrorist_organization |
| 12 | terroristic | act_violence |
| 13 | fight_terrorism | specifically |
| 14 | violent_act | aggression |
| 15 | aggression | violent_act |
| 16 | terrorist_attack | mass_murder |
| 17 | terroristic_act | violence |
| 18 | airplane_hijacking | terror_attack |
| 19 | terrorist | drug_traffic |
| 20 | indiscriminate_violence | create_atmosphere |
| 21 | terrorist_organization | global_terrorism |
| 22 | global_terrorism | indiscriminate_violence |
| 23 | lawlessness | blackmail |
| 24 | aircraft_hijacking | violent_activity |
| 25 | extremism | international_cooperation |

Top-25 overlap: 17/25

**terrorist, tokens named in Table 3:** terrorist_organization 1→1; islamic_militant 11→9; islamic_extremist 12→10

| Rank | terrorist v95 | terrorist new |
|---|---|---|
| 1 | terrorist_organization | terrorist_organization |
| 2 | extremist | terrorist_activity |
| 3 | extremist_group | terrorist_attack |
| 4 | terrorist_attack | terror |
| 5 | terrorist_activity | extremist |
| 6 | terrorists | extremist_group |
| 7 | terror | terrorist_cell |
| 8 | urban_guerrilla | act_terrorism |
| 9 | terrorist_cell | islamic_militant |
| 10 | urban_guerrillas | islamic_extremist |
| 11 | islamic_militant | international_terrorism |
| 12 | islamic_extremist | terrorists |
| 13 | underground_organization | mass_murder |
| 14 | guerrillas | terrorist_network |
| 15 | act_terrorism | terrorism |
| 16 | killers | deadly_attack |
| 17 | terrorist_network | purportedly |
| 18 | palestinian_guerrilla | saudi_exile |
| 19 | plane_hijacking | weapon_explosive |
| 20 | commando | terrorism_suspect |
| 21 | terroristic_act | blackmail |
| 22 | muslim_militant | militant_group |
| 23 | militant | urban_guerrilla |
| 24 | gunman | tie_bin_laden |
| 25 | guerrilla | west_berlin_discotheque |

Top-25 overlap: 13/25

## Table 5 — per-period LDA K* and coherence

| Period | K v95 | K new | c_v v95 | c_v new |
|---|---|---|---|---|
| 1851-1900 | 4 | 4 | 0.475 | 0.475 |
| 1901-1930 | 4 | 4 | 0.339 | 0.339 |
| 1931-1950 | 4 | 4 | 0.466 | 0.466 |
| 1951-1960 | 3 | 3 | 0.445 | 0.445 |
| 1961-1980 | 4 | 4 | 0.419 | 0.419 |
| 1981-20010910 | 4 | 4 | 0.408 | 0.456 |
| 20010911-2019 | 4 | 4 | 0.448 | 0.448 |

## Table 6 — 1981–2001 topic keywords (other periods identical)

| Topic | v95 | new |
|---|---|---|
| T0 | israel, israeli, arab, attack, palestinian, force, military, lebanon, israelis, palestinians | bomb, security, hostage, officer, home, bombing, car, embassy, building, plane |
| T1 | political, military, terrorism, army, british, kill, violence, force, communist, black | soviet, iran, military, reagan, policy, soviet_union, political, congress, force, plan |
| T2 | war, right, military, washington, force, policy, south, congress, mean, interest | military, political, wilson, investigation, trial, judge, army, organization, suspect, guerrilla |
| T3 | kill, bomb, night, army, officer, home, street, fire, attack, building | israel, israeli, arab, lebanon, palestinian, libya, palestinians, force, military, syria |

## Procrustes displacement (Section 4.1 Temporal Stability Check; Table S8, Figure S5)

| Transition | Word | v95 | new |
|---|---|---|---|
| 1851-1900 -> 1901-1930 | terrorism | 0.465 | 0.465 |
| 1851-1900 -> 1901-1930 | terrorist | 0.072 | 0.072 |
| 1901-1930 -> 1931-1950 | terrorism | 0.367 | 0.367 |
| 1901-1930 -> 1931-1950 | terrorist | 0.163 | 0.163 |
| 1931-1950 -> 1951-1960 | terrorism | 0.475 | 0.475 |
| 1931-1950 -> 1951-1960 | terrorist | 0.138 | 0.138 |
| 1951-1960 -> 1961-1980 | terrorism | 0.495 | 0.495 |
| 1951-1960 -> 1961-1980 | terrorist | 0.250 | 0.250 |
| 1961-1980 -> 1981-20010910 | terrorism | 0.619 | 0.628 |
| 1961-1980 -> 1981-20010910 | terrorist | 0.156 | 0.208 |
| 1981-20010910 -> 20010911-2019 | terrorism | 0.322 | 0.218 |
| 1981-20010910 -> 20010911-2019 | terrorist | 0.139 | 0.155 |
| mean | terrorism | 0.457 | 0.441 |
| mean | terrorist | 0.153 | 0.164 |

Text numbers: peak 0.619→0.628 (same transition), minimum 0.322→0.218 (same transition), terrorism mean 0.457→0.441, terrorist mean 0.153→0.164.

## Random Forest (Supplementary S2, Tables S4–S5)

| Metric | v95 | new |
|---|---|---|
| Weighted F1 (leak-free) | 0.666 [0.635, 0.699] | 0.684 [0.652, 0.715] |
| Weighted precision / recall | 0.671 / 0.671 | 0.698 / 0.687 |
| Macro F1 | 0.666 | 0.684 |
| CountVectorizer F1 | 0.663 [0.629, 0.694] | 0.692 [0.658, 0.723] |
| TF-IDF F1 | 0.687 [0.652, 0.722] | 0.696 [0.666, 0.729] |

| Period | P v95 | P new | R v95 | R new | F1 v95 | F1 new |
|---|---|---|---|---|---|---|
| 1851-1900 | 0.712 | 0.715 | 0.832 | 0.824 | 0.767 | 0.766 |
| 1901-1930 | 0.612 | 0.636 | 0.597 | 0.571 | 0.604 | 0.602 |
| 1931-1950 | 0.697 | 0.661 | 0.633 | 0.617 | 0.664 | 0.638 |
| 1951-1960 | 0.592 | 0.564 | 0.731 | 0.782 | 0.654 | 0.655 |
| 1961-1980 | 0.667 | 0.753 | 0.538 | 0.588 | 0.595 | 0.660 |
| 1981-20010910 | 0.670 | 0.812 | 0.508 | 0.575 | 0.578 | 0.673 |
| 20010911-2019 | 0.746 | 0.745 | 0.858 | 0.850 | 0.798 | 0.794 |

## W2V robustness, uniform min_count=5 (Table S7)

Overall: mean overlap 13.6/25→13.5/25, mean Jaccard 0.40→0.40, min 2/25→3/25. Only the four 1981–2001 rows change: terrorism CBOW 21→19, SkipGram 11→9; terrorist CBOW 21→20, SkipGram 2→3. Sentence in Section 5.3 (al_qaeda in 2001–2019; klan theme in 1851–1900) unaffected.

## Global LDA (Supplementary S1)

Coherence sweep K=2..15 (balanced subsample): highest K=10 (0.482) → K=13 (0.474); K=12: 0.474→0.451. Full-corpus K=12 run: 0.498→0.531.

| K | v95 | new |
|---|---|---|
| 2 | 0.387 | 0.377 |
| 3 | 0.368 | 0.375 |
| 4 | 0.384 | 0.365 |
| 5 | 0.429 | 0.400 |
| 6 | 0.420 | 0.407 |
| 7 | 0.443 | 0.408 |
| 8 | 0.454 | 0.442 |
| 9 | 0.436 | 0.431 |
| 10 | 0.482 | 0.417 |
| 11 | 0.465 | 0.420 |
| 12 | 0.473 | 0.451 |
| 13 | 0.452 | 0.474 |
| 14 | 0.467 | 0.472 |
| 15 | 0.461 | 0.423 |

K=12 topics on the corrected corpus (Tables S2–S3 need re-labeling; same themes recur):

- Topic 0: political, british, kill, army, guerrilla, military, violence, force, communist, organization
- Topic 1: school, child, home, student, business, mayor, job, street, run, help
- Topic 2: israel, iran, arab, meet, meeting, libya, peace, washington, syria, palestinian
- Topic 3: iraq, military, iraqi, force, kill, attack, troop, soldier, baghdad, army
- Topic 4: kill, attack, bomb, building, bombing, airport, fire, plane, car, officer
- Topic 5: french, russia, soviet, france, russian, german, europe, moscow, european, germany
- Topic 6: judge, lawyer, trial, prosecutor, rule, investigation, accuse, terrorism, prison, defendant
- Topic 7: agency, plan, security, intelligence, program, federal, threat, attack, provide, effort
- Topic 8: israel, israeli, palestinian, palestinians, hamas, attack, lebanon, kill, israelis, gaza
- Topic 9: pakistan, attack, afghanistan, taliban, india, kill, muslim, military, pakistani, al_qaeda
- Topic 10: bush, iraq, war, campaign, vote, election, republican, white_house, democrats, president_bush
- Topic 11: war, right, mean, problem, policy, force, political, america, peace, economic
