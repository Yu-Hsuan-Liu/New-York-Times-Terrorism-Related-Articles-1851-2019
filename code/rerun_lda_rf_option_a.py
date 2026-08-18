# -*- coding: utf-8 -*-
"""
Option A: Retrain LDA and RF with updated check_additional_sw2 stopwords.
Re-applies the current check_additional_sw2 (from nyt_main_training.py) to
clean_lemmatized from the saved parquet, then saves a new parquet and retrains
both RF (balanced, seed=123) and LDA (K=12, seed=123).

Outputs saved to C:/Users/tosea/claude_test_project/nlp_results/:
  RF_Final_PerClass.csv       — per-period precision/recall/F1/support
  RF_Final_Bootstrap_CI.txt   — weighted F1 + 95% bootstrap CI
  RF_Feature_Importance.csv   — top-50 feature importances
  RF_Confusion_Matrix.csv     — 7x7 confusion matrix
  LDA_Topic_Keywords.csv      — LDA topic keywords for all 12 topics
  LDA_Topic_Proportions.csv   — topic proportions by period (winner-take-all)
  LDA_Normalized_Keywords.txt — summary of labelled topics
  LDA_Coherence_Scores.csv    — K=12 coherence score
  lda_coherence_log.txt       — coherence log
"""
import re
import sys
import pickle
import warnings
import random
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (precision_recall_fscore_support, f1_score,
                             classification_report, confusion_matrix)
import gensim.corpora as corpora
from gensim.models import LdaMulticore, CoherenceModel
from gensim.corpora import Dictionary

warnings.filterwarnings("ignore")
random.seed(123)
np.random.seed(123)

# --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
_os.makedirs(RESULTS_ROOT, exist_ok=True)
PARQUET_PATH  = _os.path.join(DATA_ROOT, 'combined_processed_df.parquet')
NEW_PARQUET   = _os.path.join(DATA_ROOT, 'combined_processed_df_updated.parquet')
OUT_DIR       = RESULTS_ROOT
SAMPLE_DIR    = _os.path.join(DATA_ROOT, 'sample_updated')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Copy the FULL updated check_additional_sw2 from nyt_main_training.py
# ─────────────────────────────────────────────────────────────────────────────
_ADDITIONAL_SW2_LIST = [
        'act_letters', 'act_proquest', 'act_proquest_historical', 'act_proquest_historical_time',
        'act_proquest_historical_times', 'associated_dress_proquest_historical',
        'associated_dress_time_proquest', 'associate_dress_dug_proquest', 'berlin_sov',
        'bueno_air_dig', 'cable_dispatch', 'callender_time', 'class_hit', 'class_hit_act_proquest',
        'class_hit_correspondence', 'class_hit_dec_proquest', 'class_hit_dug_proquest',
        'class_hit_henry_tanner', 'class_hit_hugh', 'class_hit_otto_tireless',
        'class_hit_proquest_historical', 'class_hit_run_proquest', 'class_hit_sov_proquest',
        'class_hit_special_table', 'class_hit_special_york', 'class_hit_time_dec',
        'class_hit_time_dug', 'class_hit_time_proquest', 'class_hit_time_sov',
        'class_hit_time_web', 'class_hit_times_rep', 'class_hit_tireless_york',
        'class_hit_web_proquest', 'class_htt', 'company_special_table', 'company_tireless',
        'correspondence_york', 'correspondent_time_proquest', 'correspondent_times',
        'dec_proquest_historical', 'dec_proquest_historical_time', 'dec_proquest_historical_times',
        'dev_sol', 'dig_historical', 'dig_letter', 'dig_proquest_historical',
        'dig_proquest_historical_times', 'dispatch_time_time_rep', 'dispatch_times',
        'dispatch_times_london', 'dispatch_times_times_rep', 'dublin_web', 'dug_proquest_historical',
        'dug_proquest_historical_time', 'dug_proquest_historical_times', 'est_history',
        'full_page_image_microfilm', 'hague_oct', 'historical', 'historical_block_due',
        'historical_oo', 'historical_red', 'historical_summary', 'historical_times',
        'historical_times_brazil', 'historical_times_cyprus', 'historical_times_mau_mau',
        'historical_xx', 'hot', 'howe_dig_proquest', 'johannesburg_rev', 'juan_special_dug_proquest',
        'keep_upi', 'london_dec', 'london_sov', 'malcolm_times_proquest', 'metropolitan_dig_proquest',
        'metropolitan_times', 'metropolitan_times_proquest', 'metropolitan_times_rep',
        'miami_dec', 'nations_sov', 'netherlands_dec', 'ny', 'paris_dig', 'pes_historical',
        'proquest_historical', 'proquest_historical_block_due', 'proquest_historical_british',
        'proquest_historical_french', 'proquest_historical_guatemala', 'proquest_historical_japanese',
        'proquest_historical_red', 'proquest_historical_summary_international',
        'proquest_historical_terrorist', 'proquest_historical_time', 'proquest_historical_times',
        'proquest_historical_times_mau', 'proquest_historical_world_brief', 'reich_class_hit',
        'rep_proquest_historical', 'rep_proquest_historical_time', 'rep_proquest_historical_times',
        'robert_sov_proquest', 'roe_historical', 'run_proquest', 'run_proquest_historical',
        'run_proquest_historical_times', 'sam_pope_brewerspecial_york', 'shanghai_dig',
        'singapore_dig', 'slain_class_hit', 'sm', 'sov_kellogg', 'sov_proquest',
        'sov_proquest_historical', 'sov_proquest_historical_times', 'sov_upi', 'spa',
        'spa_time', 'spad', 'spad_time', 'spai', 'special_act_proquest', 'special_aux',
        'special_cablegram', 'special_correspondence', 'special_dispatch',
        'special_dug_proquest_historical', 'special_fo', 'special_hor', 'special_hors',
        'special_jerk', 'special_lo', 'special_proquest_historical', 'special_run_proquest',
        'special_table', 'special_table_york_times', 'special_time_time', 'special_time_time_dec',
        'special_time_time_dug', 'special_time_time_proquest', 'special_time_time_rep',
        'special_time_time_run', 'special_time_time_sov', 'special_time_time_web',
        'special_timees', 'special_times', 'special_times_act_proquest', 'special_times_algiers',
        'special_times_dec_proquest', 'special_times_proquest_historical',
        'special_times_rep_proquest', 'special_times_run_proquest', 'special_times_sov_proquest',
        'special_times_times', 'special_times_times_proquest', 'special_times_times_rep',
        'special_times_times_sov', 'special_times_times_web', 'special_times_web_proquest',
        'special_timesby', 'special_toe', 'special_yore', 'special_york', 'special_york_time_rep',
        'special_york_times', 'special_york_times_proquest', 'special_york_times_times',
        'specialto_time', 'summary_proquest_historical', 'table_york', 'tad_times_proquest_historical',
        'terrorist_act_proquest', 'terrorist_dec_proquest', 'terrorist_dig_proquest',
        'terrorist_rep_proquest', 'terrorist_run_proquest', 'thh', 'tim_time', 'tim_times',
        'tim_times_proquest_historical', 'time_act_proquest', 'time_br', 'time_company_tireless',
        'time_dec', 'time_dec_proquest_historical', 'time_dug', 'time_dug_proquest_historical',
        'time_pal', 'time_proquest_historical_time', 'time_rep', 'time_rep_proquest_historical',
        'time_run', 'time_run_proquest_historical', 'time_sm', 'time_sov',
        'time_sov_proquest_historical', 'time_time', 'time_time_act_proquest',
        'time_time_dec_proquest', 'time_time_dug_proquest', 'time_time_proquest_historical',
        'time_time_rep_proquest', 'time_time_run_proquest', 'time_time_sov_proquest',
        'time_time_war_proquest', 'time_time_web_proquest', 'time_war_proquest',
        'time_web_proquest_historical', 'timees', 'times_act_proquest', 'times_company_tireless',
        'times_dec', 'times_dec_proquest_historical', 'times_dig', 'times_dig_proquest_historical',
        'times_dug_proquest_historical', 'times_johannesburg', 'times_lisbon', 'times_london',
        'times_madrid', 'times_montreal', 'times_nations', 'times_paris', 'times_proquest_historical',
        'times_proquest_historical_times', 'times_rep', 'times_rep_proquest_historical',
        'times_rome', 'times_run_proquest', 'times_sov', 'times_sov_proquest_historical',
        'times_tel_aviv', 'times_times', 'times_times_act_proquest', 'times_times_dec_proquest',
        'times_times_proquest_historical', 'times_times_rep_proquest', 'times_times_sov_proquest',
        'times_times_web_proquest', 'times_war', 'times_web', 'times_web_proquest_historical',
        'times_world_brief', 'tireless_dispatch', 'tireless_sum', 'tireless_times',
        'tireless_times_london', 'tireless_york', 'tireless_york_tim_time',
        'tireless_york_tim_times', 'tireless_york_time', 'tireless_york_time_proquest',
        'tireless_york_time_time', 'tireless_york_times', 'tireless_york_times_dug',
        'tireless_york_times_proquest', 'tireless_york_times_rep', 'tireless_york_times_times',
        'tireless_yorrk_time', 'uruguay_dig', 'war_proquest', 'war_proquest_historical',
        'web_letter', 'web_proquest_historical', 'web_proquest_historical_times', 'web_upi',
        'world_apa', 'world_brief', 'world_briefs', 'xx', 'yore', 'yore_time', 'yore_times',
        'york', 'york_time', 'york_time_harold_denny', 'york_time_time', 'york_time_time_dec',
        'york_time_time_dug', 'york_time_time_proquest', 'york_time_time_rep', 'york_time_time_run',
        'york_time_time_sov', 'york_time_time_web', 'york_times', 'york_times_dug_proquest',
        'york_times_harold_denny', 'york_times_proquest_historical', 'york_times_run_proquest',
        'york_times_times', 'york_times_times_dec', 'york_times_times_proquest',
        'york_times_times_rep', 'york_times_times_sov', 'york_times_times_web', 'yorre_time',
        'yorrk_time', 'times_dublin', 'roundup', 'charle_times_proquest', 'special_times_proquest',
        'abend_special_table', 'alvin_special', 'alvin_times', 'apple_special',
        'apple_special_time_time', 'apple_special_times', 'apple_special_times_london',
        'ben_franklin', 'bernard', 'bernard_special', 'bernard_special_times',
        'bernard_special_times_times', 'bernard_times', 'bernard_times_times',
        'bernard_weinraubspecial_time_time', 'camille_york_times', 'carey',
        'christopher_wren', 'clarence_streit_tireless_york', 'clarence_tireless',
        'clarence_tireless_york_times', 'clifton_daniel_special_york', 'clifton_daniel_tireless_york',
        'clifton_danielspecial_york_time', 'clifton_york_times', 'clifton_york_times_times',
        'clyde_special', 'dana_adams', 'dana_adams_schmidt', 'david_binder', 'david_special_times',
        'david_vidal', 'draw_middleton', 'drew_middletonspecial', 'edward_special_times',
        'edwin_james', 'eric_pace_special_times', 'flora_lewis', 'flora_lewis_special_times',
        'frederick_special', 'frederick_special_table', 'frederick_times_dig',
        'frederick_tireless_york_times', 'garcon_transatlantic_tireless_telegraph', 'gen_sheridan',
        'gene_currivanspecial_york_time', 'gene_special', 'gene_tireless', 'gene_york',
        'gene_york_times', 'granger', 'granger_blair_special', 'guido_special_table',
        'hanson_baldwin_special_times', 'hanson_baldwin_times_proquest', 'harold_denny',
        'harold_denny_special_table', 'henry_dug', 'henry_special', 'henry_special_times',
        'henry_special_times_proquest', 'henry_special_times_times', 'henry_special_york_times',
        'henry_tanner', 'henry_tanner_special', 'henry_tanner_special_times',
        'henry_times_proquest', 'henry_times_times', 'homer', 'homer_special',
        'homer_special_times', 'howe_special', 'howe_special_ankara_turkey', 'howe_special_times',
        'howe_special_times_proquest', 'irving_special', 'irving_times_rep', 'james_markham',
        'james_markham_special_times', 'james_special_times', 'jay_special_times_times',
        'jefferson_davis', 'john_burn', 'john_burns', 'john_lee', 'john_special_times',
        'jon_special_times', 'jon_special_times_times', 'jonathan_special_times_times',
        'joseph_novitskispecial', 'joseph_special', 'juan_onisspecial_time_time', 'juan_special',
        'juan_special_buenos_air', 'juan_special_times', 'juan_special_times_times', 'juan_times',
        'juan_times_times', 'julian_louis', 'julian_louis_special_times', 'julian_louis_york_times',
        'kathleen_special', 'kathleen_special_times', 'kathleen_special_times_times',
        'kathleen_teltschspecial_time_time', 'kathleen_times_times', 'kenneth_briggs',
        'lawrence_fellow_special_times', 'lawrence_fellows_special_times', 'lawrence_tireless',
        'leonard', 'leonard_ingallsspecial', 'malcolm_browne_special',
        'malcolm_browne_special_times', 'marjorie_hunter_special_times', 'michael',
        'michael_clark', 'michael_kaufman', 'milton', 'nairobi_sonya', 'nairobi_sonya_web',
        'occasional_correspondent', 'otto_tireless_york_times', 'paul_hofmann_special_time',
        'paul_hofmannspecial', 'paul_special', 'paul_special_times', 'paul_special_times_rome',
        'paul_special_times_proquest', 'paul_special_times_times', 'paul_times_times',
        'peter_special_times', 'professor', 'raymond', 'raymond_anderson_special_times',
        'richard_johnston', 'richard_special', 'robert_aldenspecial', 'robert_special',
        'robert_special_times', 'robert_times_proquest', 'robert_trumbull', 'roy_reed',
        'roy_reed_special_times', 'roy_times_times', 'russell_porter', 'russell_porter_special_york',
        'sam_pope_brewer_special', 'sam_pope_brewer_tireless', 'sam_pope_york_times', 'seth',
        'seth_king', 'seth_kingspecial_time_time', 'seth_times_times', 'sidney', 'smith_special',
        'special_times_addis_ababa', 'special_times_salisbury', 'steven', 'steven_roberts',
        'sydney_tireless', 'sydney_tireless_york_times', 'tad_special', 'tad_special_washington',
        'thomas', 'thomas_brady_special_time', 'thomas_brady_special_times', 'thomas_bradyspecial',
        'thomas_johnson', 'tom_wicker', 'ulster_bernard', 'walter', 'william_borders',
        'william_carney', 'thomas_times_proquest', 'pal_proquest_historical_times',
        'proquest_historical_nazi', 'proquest', 'continued_age_column', 'age_column', 'continued',
        'times_dublin', 'times_jerusalem', 'times_algiers', 'times_caracas',
        'hart_times_times', 'seth_king_special_times', 'irving_special_times',
        'special_times', 'upi', 'correspondence', 'dispatch', 'subscription_list',
        'abominable', 'abrogate', 'acceptance', 'accompany', 'admiral_worthy', 'affairs',
        'aide', 'aides', 'alarming', 'allegation', 'alleged', 'allude', 'amid',
        'amid_indication', 'amongst', 'anniversary', 'apparent', 'appropriately',
        'associate_dress', 'associated_dress', 'assumption', 'astounding', 'attribute',
        'authentic', 'authenticity', 'autumn', 'become_apparent', 'become_frequent',
        'behavior', 'branches', 'briefly', 'candid', 'candor', 'carry_weight', 'catalogue',
        'coming', 'committed', 'companion', 'confirm', 'conscientious', 'cope', 'correction',
        'correctly', 'culminate', 'customer', 'dark', 'dawn', 'daylight', 'dec',
        'deep_impression', 'deeply_divide', 'definite', 'deformity', 'deplorable',
        'deplore', 'difference_opinion', 'dignified', 'disagree', 'disapproval',
        'disavow', 'disclaim', 'disclosure', 'dispassionate', 'dramatic_rescue',
        'editorial', 'effectual', 'energetic', 'envoy', 'except_perhaps', 'explicit',
        'exposition', 'expressly', 'fall_short', 'faraway', 'favorite', 'fifteen_person',
        'flagrant', 'forwards', 'fragment', 'fresh', 'gene', 'governorship', 'gov',
        'gratified', 'gross', 'hair', 'headline', 'honors', 'humiliate', 'impute',
        'inclusive', 'incomplete', 'increasingly', 'indignant', 'indorse', 'inference',
        'inauguration', 'initial', 'intelligible', 'intervene', 'introduce_bill',
        'justifiable', 'late', 'latitude', 'leading', 'letters', 'lightest', 'loop',
        'main_street', 'mat', 'memorial_service', 'misrepresentation', 'narrate',
        'necessitate', 'obnoxious', 'omission', 'originally', 'outline', 'papers',
        'participation', 'perseverance', 'persistent', 'persistently', 'physically',
        'plain', 'politics', 'preposterous', 'publishing', 'pursuance', 'quell', 'racket',
        'reappearance', 'recapitulate', 'recall', 'refutation', 'relatively_quiet',
        'reliable', 'repeatedly', 'repetition', 'reportedly_confess', 'resolutions',
        'rebuke', 'rites', 'rosy', 'scarcely_mention', 'sequel', 'sensational', 'sensible',
        'seventeen', 'shortly', 'shun', 'side_border', 'side_side', 'simplicity',
        'solicitation', 'sole_object', 'stepping_effort', 'stirring', 'striking',
        'strong_language', 'strongly', 'subsequent', 'substantially', 'tacit', 'temperate',
        'therefrom', 'tolerably', 'totally', 'ultimate_success', 'uncertain_future',
        'understood', 'unqualified', 'unrealistic_impractical_obsession', 'unsuccessful',
        'unwillingness_inability', 'universally', 'underlying_condition', 'vague_directive',
        'verbs', 'wellknown_fact', 'wherein', 'wide_scale', 'widely_separate',
        'yearold_boy', 'yearold_daughter', 'correspondence', 'dispatch', 'special_times', 'page', 'editor', 'unconfirme',
        'reportedly', 'officially', 'broadcast_tonight', 'announce_tonight',
        'view_full_article', 'available_nytime_com', 'web_nytime_com', 'upcoming',
        'meanwhile', 'latter', 'former', 'earlier', 'stated', 'cited', 'sources',
        'ace', 'ai', 'astoria', 'aug', 'ave', 'aye', 'bar', 'batter', 'ben', 'bicycle',
        'biter', 'blonde', 'blot', 'blue', 'blush', 'bonfire', 'brand', 'bulldozer',
        'calmer', 'cap', 'carol', 'cattle', 'cemetery', 'ch', 'channe', 'charge_clot',
        'charge_clotting', 'clot', 'colombo_ri', 'comb_clot', 'comb_hills', 'combed',
        'comings', 'company_nana', 'cotton_crop', 'counsels', 'count_von', 'cur', 'curb',
        'cutis', 'cycle', 'days', 'detective', 'diagonal_shading', 'dictation',
        'disintegration', 'dissolve_yuma', 'drink', 'drum', 'eee', 'ees', 'eg', 'ein',
        'eleven_person', 'err', 'evening', 'facial', 'fig_murphy', 'fill', 'fire_marshal',
        'fireman', 'flame', 'flames', 'fling', 'flog', 'ford', 'fortnight_hour',
        'forty_person', 'funeral', 'furnishe', 'furrow', 'genital', 'gin', 'gleam',
        'grady', 'greatly_exaggerated', 'grip', 'guido', 'harvest', 'haunt', 'herr',
        'hills', 'hit', 'hor', 'horsey', 'huge_veil', 'hull', 'icon', 'idol',
        'imperille', 'incident_occur', 'irk', 'iss', 'ix', 'jacket', 'le', 'letters_au_au',
        'lull', 'luman', 'lumen', 'marche', 'mau', 'med', 'meme', 'meridian', 'mess',
        'mmm', 'modify', 'nairobi_sonya_letters', 'nbsp_behind_veil', 'nbsp_graphic_response',
        'nbsp_howell_raine_series', 'nbsp_wound', 'negation', 'newsman', 'northeast',
        'nos', 'ont', 'oris', 'painless', 'parts', 'peu_tears', 'phone_call', 'pink',
        'potter', 'push', 'qui_ana', 'raft', 'razor', 'rectal', 'rectal_times', 'renew',
        'reunion', 'robe', 'rouse', 'rowe', 'rusk', 'scent', 'sec', 'sedan', 'semi_stern',
        'semi_unionist', 'sera', 'shoulder', 'sift', 'signal', 'sixty_siege', 'soda',
        'sort_pillow', 'sos', 'special_algier', 'special_caracas', 'special_fo',
        'special_holmes', 'special_hor', 'special_lines', 'special_rome', 'special_tres',
        'spoil_system', 'spur', 'st_inst', 'suis_arc', 'ter', 'terr', 'terror_ism',
        'thr', 'time_le', 'tion', 'tive', 'toll', 'tome', 'tre', 'tremen', 'tres',
        'twenty_mile', 'typhus_dig', 'typhus_sov', 'typhus_typhus', 'vi_hence',
        'vi_semi', 'village_near', 'watt', 'widow', 'widen', 'wolf',
        'oftener', 'therewith', 'therein', 'therefrom', 'thereafter', 'enact',
        'ix', 'ra', 'hor', 'ont', 'ave', 'err',
        'william_borders', 'roy_reed', 'seth_king', 'henry_tanner', 'flora_lewis',
        'malcolm_times', 'jon_special', 'robert_sov', 'james_markham', 'tom_wicker',
        'hanson_baldwin', 'charle_egan', 'joseph_levy', 'henry_dug',
        'singer', 'breed', 'angry', 'faraway', 'inclusive', 'truthful',
        'appropriately', 'oftener', 'frankly_admit', 'fester_lloyd', 'marche',
        'fortnight_hour', 'twenty_mile', 'stepping_effort', 'daily_herald',
        'special_havana', 'special_times_algiers', 'proquest_historical_guatemala',
        'summary_international_national', 'hanson_baldwin_times_proquest',
        'granger_times_proquest_historical', 'tad_times_proquest_historical',
        'paul_special_times_algiers', 'terrorist_henry_tanner', 'lloyd_garrison',
        'william_farrell', 'jay', 'brow', 'special_times_algiers',
        'times', 'urea', 'lash', 'paul_special_times_algiers', 'terrorist_henry_tanner', 'granger_blair_special_times',
        'bernard_run_proquest', 'tad_times', 'pain_james_markham',
        'shortlive', 'urea', 'lash', 'jerry', 'phil', 'madrid_web', 'former',
        'editor', 'think', 'page', 'talk', 'give', 'come', 'find', 'call', 'live',
        'support', 'issue', 'fact', 'report', 'percent', 'million', 'billion',
        'nbsp', 'vol', 'say', 'said', 'know', 'get', 'give', 'make',
        'go', 'look', 'want', 'tell', 'use', 'issue',
        'thing', 'something', 'way', 'day', 'year', 'time', 'official', 'member', 'nbsp', 'sov', 'thh', 'iii', 'vi', 'au', 'dec', 'rep', 'vol', 'irve', 'ja', 'na',
        'rib', 'pari', 'las', 'ff', 'und', 'nd', 'om', 'rete', 'vy', 'rrs', 'tes', 'meg',
        'bu_loud', 'sup', 'acr', 're', 'co', 'con_lift', 'con_tint', 'tohave', 'andother',
        'ofthese',
        'backward', 'expect', 'thus', 'reflect', 'remind', 'echo',
        'suggest', 'conclusion', 'addition', 'part', 'probability', 'possibility',
        'significance', 'untrue', 'unfair', 'unimportant', 'hereby', 'plainly', 'grossly',
        'truly', 'soon_possible',
        'pronounced', 'wisely', 'meantime', 'evident', 'mildly',
        'exceptionally', 'honestly', 'strict', 'prominently',
        'temporary', 'sincere', 'grateful', 'loyal', 'entire', 'actual',
        'successful', 'possible', 'perfect', 'single', 'common', 'far', 'sporadic',
        'sufficient', 'absolute', 'extreme', 'essential', 'effective', 'complete',
        'startling', 'significant', 'ambiguous', 'illogical', 'overwhelmed', 'mostly',
        'slightly', 'slight', 'able', 'somewhat', 'document', 'equally', 'successfully',
        'al', 'non', 'seem', 'list', 'fit',
        'cased', 'assess', 'falsely', 'history', 'multiple', 'repeat', 'huge', 'contrary', 'los',
        'information', 'government', 'administration', 'official', 'national', 'international',
        'power', 'party', 'member', 'group', 'leader', 'committee',
        'law', 'act', 'bill', 'case', 'court', 'order', 'charge', 'police', 'arrest',
        'country', 'nation', 'state', 'union', 'public', 'world', 'present', 'president', 'area',
        'place', 'part', 'present', 'general', 'man', 'men', 'people', 'hold', 'charge', 'case',
        'house', 'department', 'held', 'situation', 'edit', 'cities', 'fourteen', 'eighth',
        'triple', 'tap', 'wrap', 'expected', 'sov_cela_turner',
        'run_vi', 'sov', 'thh', 'iii', 'und', 'der', 'canadian_dress', 'maid', 'air_maid',
        'dig', 'hill', 'beta', 'nought', 'wit', 'herein', 'therein', 'null_void', 'yield_demand',
        'company', 'sept', 'end', 'life', 'become', 'feel', 'question', 'person', 'believe', 'change',
        'excepted', 'american', 'book', 'write', 'play', 'age', 'show', 'near', 'film', 'try',
        'young', 'turn', 'movie', 'family', 'woman', 'dear', 'willfully', 'overwhelming_majority',
        'partial', 'als', 'din', 'qui', 'fax', 'org',
        # Added 2026-02: new artifact variants
        'abend_tireless_times', 'abend_tireless_york_times', 'agents_york',
        'algeria_times_proquest', 'alvin_special_times_dec', 'alvin_special_times_london',
        'alvin_times_proquest_historical', 'apple_special_dig_proquest',
        'associate_dress_act_proquest', 'associate_dress_dig_proquest',
        'attacks_york_washington', 'bernard_dig_proquest', 'bernard_proquest_historical',
        'bernard_special_times_london', 'charle_proquest_historical',
        'class_hit_dig_proquest', 'class_hit_hanson_baldwin', 'class_hit_thomas_brady',
        'class_hit_times_proquest', 'clifton_york_times_dig', 'clifton_york_times_jerusalem',
        'correspondent_dig_proquest', 'cyprus_class_hit', 'dig_proquest',
        'dig_proquest_historical_summary', 'frederick_tireless',
        'gene_special_york_times', 'henry_dig_proquest', 'historical_caracas',
        'historical_comb', 'historical_excerpt', 'historical_guiana',
        'historical_headliner', 'historical_mau_mau', 'historical_nazi',
        'historical_slain', 'historical_ulster', 'howe_special_dig_proquest',
        'howe_special_proquest_historical', 'hugh_tireless_york_times',
        'james_times_proquest', 'john_times_proquest', 'jon_special_times_proquest',
        'jonathan_special_dig_proquest', 'jonathan_special_proquest_historical',
        'jonathan_special_times_proquest', 'jonathan_times_proquest',
        'joseph_special_times_proquest', 'juan_special_run_proquest',
        'juan_special_times_dig', 'juan_special_times_proquest', 'juan_times_proquest',
        'kathleen_special_times_proquest', 'lanse_york_times_times',
        'malaya_class_hit', 'michael_times_proquest', 'national_metropolitan_dec_proquest',
        'ne_historical_times', 'pal_proquest_historical', 'paul_times_proquest',
        'pe_historical', 'peter_times_proquest', 'potent_historical',
        'prone_historical', 'proquest_historical_arabs', 'proquest_historical_associate_dress',
        'proquest_historical_britain', 'proquest_historical_chief',
        'proquest_historical_crime', 'proquest_historical_dress_international',
        'proquest_historical_garter', 'proquest_historical_interesting',
        'proquest_historical_irish', 'proquest_historical_israel',
        'proquest_historical_kill', 'proquest_historical_palestine',
        'proquest_historical_police', 'proquest_historical_political',
        'proquest_historical_run', 'proquest_historical_soviet',
        'proquest_historical_talk', 'proquest_historical_terror',
        'proquest_historical_venezuela', 'proquest_historical_wagon',
        'proquest_historical_war', 'rep_historical', 'report_dig_proquest',
        'robert_act_proquest', 'robert_dig_proquest', 'robert_times_rep',
        'roe_historical_times', 'roy_reed_special_york', 'sheila_rule_special_york',
        'smith_times_proquest', 'special_dig_proquest', 'special_table_tre',
        'special_times_dig_proquest', 'special_york_times_algiers',
        'steven_special_york', 'table_york_times', 'tad_proquest_historical',
        'tad_times_rep_proquest', 'terry_times_rep', 'times_proquest',
        'tireless', 'tireless_belgrade', 'tireless_bucharest', 'tireless_due',
        'tireless_jerusalem', 'tireless_jerusalem_dig', 'tireless_times_jerusalem',
        'tireless_times_paris', 'tireless_toe', 'tireless_tre',
        'tireless_tre_jerusalem', 'tireless_vienna', 'tireless_wear',
        'tireless_yore', 'tireless_yore_jerusalem', 'tireless_york_times_dig',
        'tireless_york_times_jerusalem', 'tireless_york_times_london',
        'walter_tireless', 'wireless_york_times', 'york_landmarks',
        'york_times_dig_proquest',
        # Added 2026-02 (second pass): post-retrain cleanup
        'special_tre', 'buenos_air_dig', 'georgetown_british_qui_ana',
        'colombo_melon', 'thomas_brady', 'vagina', 'penis', 'domo',
        'sub_jest', 'warsaw_dispatch',
]
_ADDITIONAL_SW2_SET = set(_ADDITIONAL_SW2_LIST)  # computed once at module level


def check_additional_sw2(var):
    fin_txt = [word for word in var.split() if word not in _ADDITIONAL_SW2_SET]
    return ' '.join(fin_txt)


def handle_vio_lence_merger(var):
    fin_txt = []
    for word in var.split():
        if word.lower() == 'vio_lence':
            fin_txt.append('violence')
        else:
            fin_txt.append(word.lower())
    return ' '.join(fin_txt)


def clean_garbage_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\b(\w+ \w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\b(\w+ \w+ \w+)( \1\b)+', r'\1', text)
    return text


def format_topics_sentences(ldamodel, corpus):
    rows_list = []
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: x[1], reverse=True)
        dominant_topic = row[0][0]
        topic_perc_contrib = round(row[0][1], 4)
        wp = ldamodel.show_topic(dominant_topic, topn=10)
        topic_keywords = ', '.join([w for w, _ in wp])
        rows_list.append([dominant_topic, topic_perc_contrib, topic_keywords])
    df = pd.DataFrame(rows_list, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load parquet and re-apply updated preprocessing
# ─────────────────────────────────────────────────────────────────────────────
print("Loading parquet...")
df = pd.read_parquet(PARQUET_PATH)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

print("Re-applying updated check_additional_sw2 to clean_lemmatized...")
df['clean_lemmatized_phrased'] = df['clean_lemmatized'].apply(check_additional_sw2)
df['clean_lemmatized_phrased'] = df['clean_lemmatized_phrased'].apply(handle_vio_lence_merger)
df['clean_lemmatized_phrased'] = df['clean_lemmatized_phrased'].apply(clean_garbage_text)

# Remove empty documents
before = len(df)
df = df[df['clean_lemmatized_phrased'].str.strip() != ''].copy()
after = len(df)
print(f"  Removed {before - after} empty documents after re-preprocessing.")
print(f"  Period counts after update:")
for p, cnt in sorted(df.groupby('label').size().items()):
    print(f"    {p}: {cnt}")

print(f"\nSaving updated parquet to {NEW_PARQUET}...")
df.to_parquet(NEW_PARQUET, index=False)
print("  Saved.")

ANALYSIS_COLUMN = 'clean_lemmatized_phrased'

# ─────────────────────────────────────────────────────────────────────────────
# 3. Random Forest — balanced sampling, seed=123
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("RANDOM FOREST CLASSIFICATION")
print("="*70)

n_samples = df[df['label'] == '1851-1900'].shape[0]
print(f"Balanced sample size (n per period, based on 1851-1900 count): {n_samples}")

sampled = pd.concat([
    df[df['label'] == '1851-1900'].sample(n=n_samples, replace=True, random_state=123),
    df[df['label'] == '1901-1930'].sample(n=n_samples, replace=True, random_state=123),
    df[df['label'] == '1931-1950'].sample(n=n_samples, replace=True, random_state=123),
    df[df['label'] == '1951-1960'].sample(n=n_samples, replace=True, random_state=123),
    df[df['label'] == '1961-1980'].sample(n=n_samples, replace=True, random_state=123),
    df[df['label'] == '1981-20010910'].sample(n=n_samples, replace=True, random_state=123),
    df[df['label'] == '20010911-2019'].sample(n=n_samples, replace=True, random_state=123),
], ignore_index=True)

print(f"Total balanced sample: {len(sampled)} documents ({n_samples}/period x 7 periods)")

X = sampled[ANALYSIS_COLUMN].astype(str)
y = sampled['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=123, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

print("Vectorizing (CountVectorizer, sparse)...")
vec = CountVectorizer()
X_train_v = vec.fit_transform(X_train)
X_test_v  = vec.transform(X_test)
print(f"  Vocabulary size: {len(vec.vocabulary_)}")

# Save vectorizer
with open(os.path.join(SAMPLE_DIR, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vec, f)

param_grid = {
    'n_estimators': [10, 100],
    'max_depth':    [None, 1, 10],
    'criterion':    ['gini', 'entropy'],
}

print("Running GridSearchCV (12 combinations, cv=5)...")
grid = GridSearchCV(
    RandomForestClassifier(random_state=123),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid.fit(X_train_v, y_train)
best_params = grid.best_params_
print(f"  Best params: {best_params}")
print(f"  Best CV score: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_v)

# Save model
with open(os.path.join(SAMPLE_DIR, 'my_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

# Overall metrics
f1_w = f1_score(y_test, y_pred, average='weighted')
f1_m = f1_score(y_test, y_pred, average='macro')
print(f"\nWeighted F1: {f1_w:.4f}  Macro F1: {f1_m:.4f}")
print(classification_report(y_test, y_pred))

# Per-class metrics (per period)
period_labels_sorted = sorted(df['label'].unique())
# Map internal labels to display labels
label_map = {
    '1851-1900': '1851-1900',
    '1901-1930': '1901-1930',
    '1931-1950': '1931-1950',
    '1951-1960': '1951-1960',
    '1961-1980': '1961-1980',
    '1981-20010910': '1981-2001 (pre-9/11)',
    '20010911-2019': '2001-2019 (post-9/11)',
}

prec, rec, f1, sup = precision_recall_fscore_support(
    y_test, y_pred, labels=period_labels_sorted, zero_division=0
)

per_class_rows = []
for i, lbl in enumerate(period_labels_sorted):
    display_lbl = label_map.get(lbl, lbl)
    per_class_rows.append({
        'Period':    display_lbl,
        'Precision': round(prec[i], 3),
        'Recall':    round(rec[i], 3),
        'F1':        round(f1[i], 3),
        'Support':   int(sup[i]),
    })

per_class_df = pd.DataFrame(per_class_rows)
per_class_path = os.path.join(OUT_DIR, 'RF_Final_PerClass.csv')
per_class_df.to_csv(per_class_path, index=False)
print(f"\nPer-class metrics saved to: {per_class_path}")
print(per_class_df.to_string(index=False))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=period_labels_sorted)
cm_df = pd.DataFrame(cm, index=period_labels_sorted, columns=period_labels_sorted)
cm_path = os.path.join(OUT_DIR, 'RF_Confusion_Matrix.csv')
cm_df.to_csv(cm_path)
print(f"\nConfusion matrix saved to: {cm_path}")
print(cm_df)

# Feature importance (top 50)
feat_names = vec.get_feature_names_out()
importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=feat_names).nlargest(50)
feat_imp_path = os.path.join(OUT_DIR, 'RF_Feature_Importance.csv')
feat_imp.to_csv(feat_imp_path, header=['Importance'])
print(f"\nTop 20 features:")
print(feat_imp.head(20).to_string())
print(f"Feature importances saved to: {feat_imp_path}")

# Bootstrap CI for weighted F1
print("\nRunning 1000-iteration bootstrap CI...")
rng = np.random.default_rng(123)
y_test_arr = np.array(y_test)
y_pred_arr = np.array(y_pred)
boot_f1s = []
for _ in range(1000):
    idx = rng.integers(0, len(y_test_arr), size=len(y_test_arr))
    boot_f1s.append(f1_score(
        y_test_arr[idx], y_pred_arr[idx],
        average='weighted', zero_division=0
    ))
ci_lo = np.percentile(boot_f1s, 2.5)
ci_hi = np.percentile(boot_f1s, 97.5)
print(f"  Weighted F1: {f1_w:.4f}  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

ci_path = os.path.join(OUT_DIR, 'RF_Final_Bootstrap_CI.txt')
with open(ci_path, 'w') as fp:
    fp.write(f"Weighted F1: {f1_w:.4f}\n")
    fp.write(f"95% Bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]\n")
print(f"Bootstrap CI saved to: {ci_path}")

# Overall metrics file
metrics_path = os.path.join(OUT_DIR, 'RF_Metrics.txt')
with open(metrics_path, 'w') as fp:
    fp.write(f"Weighted F1: {f1_w:.4f}\n")
    fp.write(f"Macro F1:    {f1_m:.4f}\n")
    fp.write(f"Best params: {best_params}\n")
print(f"RF overall metrics saved to: {metrics_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LDA — K=12, seed=123
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("LDA TOPIC MODELING (K=12)")
print("="*70)

data_tokens = [str(sent).split() for sent in df[ANALYSIS_COLUMN].tolist()]
id2word = Dictionary(data_tokens)
id2word.filter_extremes(no_below=5, no_above=0.5)
corpus = [id2word.doc2bow(text) for text in data_tokens]
print(f"  Dictionary size after filter: {len(id2word)}")
print(f"  Corpus docs: {len(corpus)}")

n_topic = 12
print(f"Training LDA with K={n_topic}, random_state=123, passes=10, workers=4...")
lda_model = LdaMulticore(
    corpus=corpus, id2word=id2word, num_topics=n_topic,
    random_state=123, passes=10, workers=4
)

# Save topic keywords
topics_list = []
print("\nLDA Topics:")
for idx, topic in lda_model.print_topics(-1):
    print(f"  Topic {idx}: {topic}")
    topics_list.append({'Topic_ID': idx, 'Keywords': topic})
topics_df = pd.DataFrame(topics_list)
kw_path = os.path.join(OUT_DIR, 'LDA_Topic_Keywords.csv')
topics_df.to_csv(kw_path, index=False)
print(f"\nLDA keywords saved to: {kw_path}")

# Coherence score
try:
    coh_model = CoherenceModel(model=lda_model, texts=data_tokens,
                                dictionary=id2word, coherence='c_v')
    coherence = coh_model.get_coherence()
    print(f"  Coherence (c_v): {coherence:.4f}")
    coh_path = os.path.join(OUT_DIR, 'LDA_Coherence_Scores.csv')
    pd.DataFrame([{'n_topics': n_topic, 'coherence': round(coherence, 4)}]).to_csv(coh_path, index=False)
    log_path = os.path.join(OUT_DIR, 'lda_coherence_log.txt')
    with open(log_path, 'w') as fp:
        fp.write(f"K={n_topic}: coherence={coherence:.4f}\n")
except Exception as e:
    print(f"  Coherence computation failed: {e}")

# Topic proportions by period (dominant topic assignment)
print("Computing dominant topic per document...")
df2 = df.reset_index(drop=True)
topic_sentences = format_topics_sentences(lda_model, corpus)
df_topic = pd.concat([topic_sentences, df2[['label']]], axis=1)

topic_labels = {}
for i in range(n_topic):
    words = lda_model.show_topic(i, topn=5)
    topic_labels[i] = f"T{i}: {'-'.join([w for w, _ in words])}"

PERIOD_ORDER = [
    '1851-1900', '1901-1930', '1931-1950', '1951-1960',
    '1961-1980', '1981-20010910', '20010911-2019',
]

topic_counts = df_topic.groupby(['label', 'Dominant_Topic']).size().unstack(fill_value=0)
topic_counts_norm = topic_counts.div(topic_counts.sum(axis=1), axis=0)
topic_counts_norm.columns = [topic_labels.get(c, c) for c in topic_counts_norm.columns]
topic_counts_norm = topic_counts_norm.reindex(PERIOD_ORDER)

prop_path = os.path.join(OUT_DIR, 'LDA_Topic_Proportions.csv')
topic_counts_norm.to_csv(prop_path)
print(f"LDA topic proportions saved to: {prop_path}")

# Normalized keywords summary
norm_path = os.path.join(OUT_DIR, 'LDA_Normalized_Keywords.txt')
with open(norm_path, 'w', encoding='utf-8') as fp:
    fp.write("LDA K=12 Topic Keywords (Option A rerun, seed=123)\n")
    fp.write("="*60 + "\n")
    for idx in range(n_topic):
        words = lda_model.show_topic(idx, topn=10)
        wp_str = ', '.join([f"{w} ({p:.3f})" for w, p in words])
        fp.write(f"Topic {idx}: {wp_str}\n")
print(f"LDA normalized keywords saved to: {norm_path}")

print("\n" + "="*70)
print("OPTION A COMPLETE")
print(f"  RF Weighted F1: {f1_w:.4f}  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Best RF params: {best_params}")
print(f"  Vocab size: {len(vec.vocabulary_)}")
print("  All outputs saved to:", OUT_DIR)
print("="*70)
