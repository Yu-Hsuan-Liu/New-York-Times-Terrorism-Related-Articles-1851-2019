# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle
import re
import warnings
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import spacy
import datetime

# NLTK Imports
from nltk import word_tokenize
from nltk.text import Text
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Sklearn Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA

# Gensim Imports
from gensim.models import Word2Vec, Phrases, LdaMulticore, CoherenceModel, LdaModel
from gensim.models.phrases import Phraser
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from kneed import KneeLocator

# ==========================================
# 1. GLOBAL SETUP & FUNCTIONS
# ==========================================
# Suppress warnings
warnings.filterwarnings("ignore")

# Load Spacy Model globally
try:
    nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])
except OSError:
    print("WARNING: Spacy model 'en_core_web_md' not found. Please run: python -m spacy download en_core_web_md")

def clean_text(txt_in):
    if not isinstance(txt_in, str): return ""
    clean = re.sub('[^A-Za-z]+', " ", txt_in).strip()
    return clean

def rem_sw(var):
    """Removes standard NLTK English stopwords."""
    sw = set(stopwords.words('english'))
    my_test = [word for word in var.split() if word not in sw]
    my_test = ' '.join(my_test)
    return my_test.lower()

def check_additional_sw(var):
    additional_sw = ["new", "de","work", "time", "permiss", "one", "would", "the", "The", 'who',
                     "two", "three", "year", "work", "span", "man", "men", "By", 'what', 'why', 'when', 'how', 'where',
                     "r", "e", "p", "copyright", "index", "n", "a", "b", "c",
                     "d", "f", "g", "h", "i", "j","k","l","m","n","o","q",
                     "s","t","u","v","w",'x','y','z', "said", "could", "upon", "ist",
                     "may", "without", "made", "make", "through", "saying",
                     "even","New", "York", "Times", "copyright", "said", "would", "one" ,"two", "year", "mr", "added", "including", "years",
                      "monday", "see", "make", "three", "since", "say", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                     "made", "want", "many", "without", "need", "get", "way", "through", "whether", "next", "used", "saying", "going",
                     "even", "month", "week", "led", "also",  "go",  "still", "told", "may",
                  "back", "like", "news", "people", "take", "never", "often", "every", "though", "later", "recent", "use", "could", "much",
                 "last", "around", "put", "away", "began", "left", "four", "five", "six", "seven", "eight", "nine", "ten", "good", "months",
                 "accross", "others", "first", "second", "third", "ms", "already", "long", "known", "wrote", "among", "asked", "accross",
                 "according", "might", "high", "yesterday", "today", "tomorrow",  "state",
                 "along", "must", "went", "little", "nan", "published", "oe", "mrs",
                 "e", "ay", "te", "ta", "et", "ad", "however", "other", "another", "en", "os",
                 "ar", "shall", "less", "more", "great", "well", "large", "old", "yet", "nothing",
                 "work", "upon", "several", "whose", "matter", "es", "ever", "almost",
                 "know",  "ago", "cannot", "thing", "cause", "no", "yes", "er",
                 "span", "index", "newspapers", "1923", "current", "reproduction", "reproduced", "1923current", "newyork", "prohibited",
                 "city", "country", "county", "day", "time", "either", "pg", "owner", "states", "united", "us", "reproduc", "reproduct", "word",
                   "office", "official", "name", "unit", "file", "permission", "january",
                   "february", "march", "april", "may", "june", "july", "august", "september",
                   "october", "november", "december", "historical time", 'Historical Time']

    fin_txt = [word for word in var.split() if word not in additional_sw]
    fin_txt = ' '.join(fin_txt)
    return fin_txt

def handle_klan_merger(var):
    fin_txt = []
    for word in var.split():
        word_lower = word.lower()
        if word_lower in ["klux", "kuklux","kukluxism", "kuklu", "kukiu", "kutlux", "klansman"]:
             fin_txt.append("klan")
        else:
             fin_txt.append(word_lower)
    return ' '.join(fin_txt)

def check_dictionary(text):
    if not isinstance(text, str): return ""
    doc = nlp(text)
    valid_words = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_oov and token.is_alpha
    ]
    return " ".join(valid_words)

def lem_fun(var):
    lemmatizer = WordNetLemmatizer()
    tmp_txt = [lemmatizer.lemmatize(word) for word in var.split()]
    return ' '.join(tmp_txt)

def give_year_labels(df):
    conditions = [
        ((df["year"] >= 1851) & (df["year"] <= 1900)),
        ((df["year"] >= 1901) & (df["year"] <= 1930)),
        ((df["year"] >= 1931) & (df["year"] <= 1950)),
        ((df["year"] >= 1951) & (df["year"] <= 1960)),
        ((df["year"] >= 1961) & (df["year"] <= 1980))]
    values = ['1851-1900', '1901-1930', '1931-1950', '1951-1960', '1961-1980']
    df['label'] = np.select(conditions, values, default='Unknown')
    return df

def fetch_bi_grams(var):
    sentence_stream = np.array(var, dtype=object)
    bigram = Phrases(sentence_stream, min_count=3, threshold=10, delimiter="_")
    trigram = Phrases(bigram[sentence_stream], min_count=3, threshold=10)
    bigram_phraser = Phraser(bigram)
    trigram_phraser = Phraser(trigram)
    bi_grams = list()
    tri_grams = list()
    for sent in sentence_stream:
        bi_sent = bigram_phraser[sent]
        bi_grams.append(bi_sent)
        tri_grams.append(trigram_phraser[bi_sent])
    return bi_grams,  tri_grams

def check_additional_sw2(var):
    additional_sw2 = [
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
        'proquest_historical_nazi', 'proquest',     'continued_age_column', 'age_column', 'continued',
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
        'times',  'urea', 'lash', 'paul_special_times_algiers', 'terrorist_henry_tanner', 'granger_blair_special_times',
        'bernard_run_proquest', 'tad_times', 'pain_james_markham',
        'shortlive', 'urea', 'lash', 'jerry', 'phil', 'madrid_web', 'former',
        'editor', 'think', 'page', 'talk', 'give', 'come', 'find', 'call', 'live', 
        'support', 'issue', 'fact', 'report', 'percent', 'million', 'billion', 
        'nbsp', 'vol', 'say', 'said', 'know', 'get', 'give', 'make', 
        'go', 'look', 'want', 'tell', 'use',  'issue', 
        'thing', 'something', 'way', 'day', 'year', 'time', 'official', 'member','nbsp', 'sov', 'thh', 'iii', 'vi', 'au', 'dec', 'rep', 'vol', 'irve', 'ja', 'na', 
        'rib', 'pari', 'las', 'ff', 'und', 'nd', 'om', 'rete', 'vy', 'rrs', 'tes', 'meg', 
        'bu_loud', 'sup', 'acr', 're', 'co', 'con_lift', 'con_tint', 'tohave', 'andother', 
        'ofthese',
        'backward',  'expect','thus', 'reflect', 'remind', 'echo',
        'suggest', 'conclusion', 'addition', 'part', 'probability', 'possibility', 
        'significance', 'untrue', 'unfair', 'unimportant', 'hereby', 'plainly', 'grossly', 
        'truly', 'soon_possible', 
        'pronounced', 'wisely',  'meantime',  'evident', 'mildly', 
        'exceptionally', 'honestly', 'strict',  'prominently', 
        'temporary', 'sincere', 'grateful', 'loyal', 'entire', 'actual', 
        'successful', 'possible',  'perfect', 'single', 'common', 'far', 'sporadic',
        'sufficient', 'absolute', 'extreme', 'essential', 'effective', 'complete', 
        'startling', 'significant','ambiguous', 'illogical', 'overwhelmed', 'mostly',
        'slightly', 'slight', 'able', 'somewhat',  'document', 'equally', 'successfully', 
        'al',  'non', 'seem',  'list', 'fit', 
        'cased', 'assess',  'falsely',  'history', 'multiple', 'repeat',  'huge', 'contrary', 'los',
        'information', 'government', 'administration', 'official', 'national', 'international',
        'power', 'party', 'member', 'group', 'leader', 'committee',
        'law', 'act', 'bill', 'case', 'court', 'order', 'charge', 'police', 'arrest',
        'country', 'nation', 'state', 'union', 'public', 'world','present', 'president', 'area', 
        'place', 'part', 'present', 'general', 'man', 'men', 'people', 'hold', 'charge', 'case', 
        'house', 'department', 'held', 'situation', 'edit', 'cities', 'fourteen', 'eighth',
        'triple', 'tap', 'wrap', 'expected', 'sov_cela_turner',
        'run_vi', 'sov', 'thh', 'iii', 'und', 'der', 'canadian_dress', 'maid', 'air_maid', 
        'dig', 'hill', 'beta', 'nought', 'wit', 'herein', 'therein', 'null_void', 'yield_demand',
        'company', 'sept', 'end', 'life', 'become', 'feel', 'question', 'person', 'believe', 'change',
        'excepted', 'american', 'book', 'write', 'play', 'age', 'show', 'near', 'film', 'try',
        'young', 'turn', 'movie', 'family', 'woman', 'dear', 'willfully', 'overwhelming_majority',
        'partial', 'als', 'din', 'qui', 'fax', 'org'
    ]

    fin_txt = [word for word in var.split() if word not in additional_sw2]
    fin_txt = ' '.join(fin_txt)
    return fin_txt

def handle_vio_lence_merger(var):
    fin_txt = []
    for word in var.split():
        word_lower = word.lower()
        if word_lower in ["vio_lence"]:
             fin_txt.append("violence")
        else:
             fin_txt.append(word_lower)
    return ' '.join(fin_txt)

def clean_garbage_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\b(\w+ \w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\b(\w+ \w+ \w+)( \1\b)+', r'\1', text)
    return text

def train_dynamic_word2vec(df, output_path, analysis_col='clean_lemmatized_phrased'):
    periods = sorted(df['label'].unique())
    target_keywords = ['terrorism', 'terrorist'] 
    all_extracted_data = []

    print("\n" + "="*80)
    print("STARTING WORD2VEC TRAINING & MULTI-KEYWORD CHECK")
    print("="*80)

    for period in periods:
        print(f"\nProcessing Period: {period}")
        subset = df[df['label'] == period]
        if subset.empty: continue
            
        num_docs = len(subset)
        print(f"  > Data Size: {num_docs} documents")

        if num_docs < 1000:
            my_vec_size = 50; my_window = 3; my_epochs = 50; my_min_count = 3
            strategy = "Small (Low Dim, High Epochs)"
        elif num_docs < 10000:
            my_vec_size = 100; my_window = 5; my_epochs = 20; my_min_count = 5
            strategy = "Medium (Standard)"
        else:
            my_vec_size = 200; my_window = 5; my_epochs = 10; my_min_count = 20
            strategy = "Large (High Dim, Clean Noise)"
            
        print(f"  > Strategy: {strategy}")

        sentences = subset[analysis_col].str.split().tolist()

        for sg_mode in [0, 1]:
            mode_name = 'SkipGram' if sg_mode == 1 else 'CBOW'
            print(f"\n  > Training {mode_name}...", end=" ")
            
            model = Word2Vec(
                sentences=sentences, vector_size=my_vec_size, window=my_window,
                min_count=my_min_count, workers=4, sg=sg_mode, epochs=my_epochs, seed=123
            )
            
            safe_period = str(period).replace("/", "-")
            filename = f"{output_path}/w2v_{safe_period}_{mode_name}.model"
            model.save(filename)
            print("Done.")
            
            for target in target_keywords:
                if target in model.wv:
                    top_words = model.wv.most_similar(target, topn=500)
                    temp_df = pd.DataFrame(top_words, columns=['Neighbor_Word', 'Similarity'])
                    temp_df['Target_Word'] = target
                    temp_df['Period'] = period
                    temp_df['Model_Type'] = mode_name
                    temp_df['Rank'] = range(1, 501)
                    all_extracted_data.append(temp_df)
                else:
                    print(f"    [Target: '{target}'] NOT FOUND.")
    
    return all_extracted_data

def check_keyword_proximity(df, text_col, period, target_word, check_word, save_path, max_distance=50, max_examples=10):
    print(f"\nScanning Period: {period} for '{target_word}' near '{check_word}'...")
    subset = df[df['label'] == period]
    found_count = 0
    results_list = []


    output_file = f"{save_path}/Context_{period}_{target_word}_{check_word}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"--- Analysis Report: {target_word} near {check_word} ({period}) ---\n\n")

        for index, row in subset.iterrows():
            text = row[text_col]
            if not isinstance(text, str): continue
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            words = clean_text.split()
            
            target_indices = [i for i, w in enumerate(words) if target_word in w]
            check_indices = [i for i, w in enumerate(words) if check_word in w]
            
            if target_indices and check_indices:
                for t_idx in target_indices:
                    match_found = False
                    for c_idx in check_indices:
                        distance = abs(t_idx - c_idx)
                        if distance <= max_distance:
                            start_pos = max(0, min(t_idx, c_idx) - 25)
                            end_pos = min(len(words), max(t_idx, c_idx) + 25)
                            snippet_list = words[start_pos:end_pos]
                            highlighted = []
                            for w in snippet_list:
                                if target_word in w or check_word in w:
                                    highlighted.append(f"**{w.upper()}**")
                                else:
                                    highlighted.append(w)
                            
                            context_str = ' '.join(highlighted)
                            
                            
                            print(f"--- Match #{found_count + 1} (Row {index}) ---")
                            print(f"Distance: {distance} words")
                            print(f"Context: ... {context_str} ...\n")
                            
                            
                            f.write(f"Match #{found_count + 1} (Row {index})\n")
                            f.write(f"Distance: {distance} words\n")
                            f.write(f"Context: ... {context_str} ...\n")
                            f.write("-" * 50 + "\n")
                            
                            found_count += 1
                            match_found = True
                            break 
                    if match_found: break 
            if found_count >= max_examples: break
        
        if found_count == 0:
            msg = "No close context found."
            print(msg)
            f.write(msg)
    
    print(f"Context results saved to: {output_file}")
    
    
def coherence_score(data, colname):
    num_workers = 5
    the_data = data[colname].str.split()
    dictionary = Dictionary(the_data)
    id2word = corpora.Dictionary(the_data)
    corpus = [id2word.doc2bow(text) for text in the_data]
    c_scores = list()
    print(f"Starting LDA processing with {num_workers} workers...")
    for n_topics in range(1, 16):
        ldamodel = LdaMulticore(
            corpus, num_topics=n_topics, id2word=id2word, iterations=10, 
            passes=5, workers=num_workers, random_state=123
        )
        coherence_model_lda = CoherenceModel(
            model=ldamodel, texts=the_data, dictionary=dictionary, coherence='c_v', processes=1
        )
        score = coherence_model_lda.get_coherence()
        c_scores.append(score)
        print(f"Num Topics: {n_topics}, Coherence Score: {score:.4f}") 
    x = range(1, len(c_scores) + 1)
    kn = KneeLocator(x, c_scores, curve='concave', direction='increasing')
    opt_topics = kn.knee
    print("Optimal topics is", opt_topics)
    plt.plot(x, c_scores)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    return opt_topics

def run_dynamic_lda_optimization(df, analysis_col, save_dir, additional_sw2):
    periods = sorted(df['label'].unique())
    results_summary = {}

    for period in periods:
        print("="*80)
        print(f"Processing Period: {period}")
        
        subset = df[df['label'] == period]
        docs = subset[analysis_col].tolist()
        num_docs = len(docs)
        
        if num_docs < 10: 
            print("  > Skip: Not enough docs.")
            continue

        if num_docs < 1000:
            my_no_below = 3; my_k_range = range(1, 5); my_passes = 50; my_chunksize = num_docs
            dataset_type = "Small (High Precision Mode)"
        elif num_docs < 10000:
            my_no_below = 5; my_k_range = range(1, 5); my_passes = 20; my_chunksize = 2000
            dataset_type = "Medium (Balanced Mode)"
        else:
            my_no_below = 20; my_k_range = range(1, 5); my_passes = 5; my_chunksize = 4000
            dataset_type = "Large (High Efficiency Mode)"

        print(f"  > Profile: {dataset_type} | Data: {num_docs} docs")

        data_tokens = [str(sent).split() for sent in docs]
        data_tokens = [[w for w in doc if w not in additional_sw2] for doc in data_tokens]
        
        id2word = corpora.Dictionary(data_tokens)
        id2word.filter_extremes(no_below=my_no_below, no_above=0.4)
        corpus = [id2word.doc2bow(text) for text in data_tokens]
        
        best_score = -1.0
        best_model = None
        best_k = -1
        
        print(f"  > Searching Best K in {my_k_range}...", end=" ")
        
        for k in my_k_range:
            try:
                # Use LdaModel (Single Core) inside loops to avoid nested multiprocessing issues on Windows
                model = LdaModel(
                    corpus=corpus, id2word=id2word, num_topics=k, random_state=123,
                    passes=my_passes, chunksize=my_chunksize, alpha='auto', eta='auto', eval_every=None
                )
                cm = CoherenceModel(model=model, texts=data_tokens, dictionary=id2word, coherence='c_v', processes=1)
                score = cm.get_coherence()
                
                if score > best_score:
                    best_score = score; best_model = model; best_k = k
                    print(f"K={k}({score:.3f})↑", end=" ")
                else:
                    print(f"K={k}({score:.3f})↓ STOP")
                    break 
            except Exception as e:
                print(f"[Error K={k}]", end=" ")
                continue
                
        if best_model:
            safe_period = str(period).replace("/", "-")
            best_model.save(f"{save_dir}/lda_{safe_period}_k{best_k}.pkl")
            with open(f"{save_dir}/dictionary_{safe_period}.pkl", 'wb') as f:
                pickle.dump(id2word, f)
            print(f"\n  >>> Winner for {period}: K={best_k} (Coherence: {best_score:.4f})")
            topics_data = []
            for idx, topic in best_model.print_topics(-1):
                print(f"  T{idx}: {topic}")
                topics_data.append(topic)
            results_summary[period] = topics_data
        print("\n")

    return results_summary

def format_topics_sentences(ldamodel, corpus):
    rows_list = []
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                rows_list.append([int(topic_num), round(prop_topic, 4)])
            else:
                break
    df = pd.DataFrame(rows_list, columns=['Dominant_Topic', 'Perc_Contribution'])
    return df

# ==========================================
# 3. MAIN EXECUTION BLOCK (THE FIX)
# ==========================================

if __name__ == '__main__':
    # --- ALL EXECUTION CODE MUST BE INDENTED HERE ---
    
    start_time = time.time()
    print("Initializing Script...")
    
    # 1. Setup Paths
    # --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
    import os as _os
    REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
    DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
    RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
    _os.makedirs(RESULTS_ROOT, exist_ok=True)
    module_path = _os.path.join(DATA_ROOT, '')
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    os.chdir(module_path)
    print(f"Working Directory: {os.getcwd()}")
    
    # Custom Import (Essential for Random Forest)
    # Ensure my_functions_NYT_gpu.py is in the same folder!
    from my_functions_NYT_gpu import * 

    # 2. Load Data
    print("Loading Data...")
    online_df = pd.read_csv('nyt_online_texts.csv', sep=',', encoding="ISO-8859-1", on_bad_lines='skip')
    print(online_df.head()) # Replaced display()

    all_list = pd.read_csv(r"total_nyt_terrorism_news1850-1980_raw.csv", encoding="ISO-8859-1")
    my_pd = pd.read_csv(r"NYT_text_1851_1980.csv", encoding="ISO-8859-1")

    # 3. Filtering & Preprocessing
    print("Preprocessing Data...")
    
    not_contain = ["The Screen:","Book Review","The Kong and I","TV:","TV ",
                   "THE SCREEN","Screen","STAGE VIEW","Publishing:",
                   "Public and Private Games","Paperback","ON TELEVISION",
                   "News of the Theater","IN AND OUT OF BOOKS","Editors' Choice",
                   "Books: ","Books--Authors","Books of","Books and Authors",
                   "Books -- Authors","BOOK NOTES","BOOK ENDS","Best Sellers",
                   "BEHIND THE BEST SELLERS","At the Movies",
                   "Arts and Leisure Guide","About Real Estate","Advertising",
                   "Answers to Weekly Quiz","Answers to Quiz","A Listing of","TV",
                   "BOOKS OF THE TIMES","THE PLAY","MUSIC VIEW", "Books of The",
                   "GOING OUT Guide", "Movies", "Paper backs",
                   "Paperback Best Sellers", "Paperbacks",
                   "Westchester/This Week","Long Island/ This Week",
                   "Television This Week","New Jersey/This Week", "Weekly News Quiz",
                   "The Screen;", "What's in a Book Name?", "When Is a Movie So Bad",
                   "Sports Editor's Mailbox: The Animals at Shea/Scoreboard at Yankee Stadium",
                   "Sports News Briefs","Sports World Special","Sports of The Times",
                   "Sports of the Times",
                   "Stage: ", "A preview OF FALL BOOKS; ", "Books Authors; ",
                   "NEW PUBLICATONS", "NEW PUBLICATIONS", "New Books","OF MANY THINGS: ",
                   "PREVIEW of FALL BOOKS", "Movie Mailbag", "NEW BOOKS",
                   "Passionate Story of a Bandit:'Augusto Matraga' Is at 5th Avenue Cinema Movie From Brazil by Santos Arrives",
                   "Action Movie Set in Latin America", "M-G-M FILLS ROLES IN TWO NEW FILMS;",
                   "His Latest Volume of Collected Plays", "Musical Cartoon to Play At the Brooklyn Academy",
                   "STUDENTS TO GIVE PLAYS; 4 Municipal Colleges to Present Drama Festival May 12",
                   "The Theater: A Musical","'Minnie' on Music -- and Rain",
                   "Amid Trials of Croatian Nationalists, a Satirical Musical Comedy in Zagreb Evokes Laughter Through Tears",
                   "THE STADIUM SEASON; Management Overcomes Many Difficulties -- Need of Music in Wartime",
                   "Drama: Adapting Wiesel's", "HITCHCOCK: MASTER MELODRAMATIST", "Terrorism Is Drama",
                   "They Weren't Riotous Comedies", "Sports In America",
                   "'Mary Burns, Fugitive,' the Melodrama of a Girl Who Loved a Murderer, at the Paramount.",
                   "'Thunderbolt,' Entebbe Raid In Israeli Film",
                   "3 1/2-Hour Film Based on Uris' Novel Opens", "Screen: ",
                   "6 Film Studio's Vie Over Entebbe Raid", "STAGE VIEW",
                   "CHARGES TERRORISM IN FILM INDUSTRY; Head of New Jersey",
                   "FILM CONTROVERSY", "FILM VIEW", "Film Festival: ", "Film Fete:",
                   "Film:", "Kennedy Center Drops Disputed Film", "SHOP TALK",
                   "Shouldn't Suspense Films Do More Than Just Kill Time", "STAMPS",
                   "'Viva Italia!,' Starring Vittorio Gassman, Returns to Day of Separate-Sketch Film:The Cast"]

    excluding_index_list = [15424, 11867, 11874, 13516, 14336, 18744, 19544, 19603, 20110,
                            20346, 20670, 21318, 21508, 21610, 21649, 21659, 21927, 22122, 22216]

    # Filter my_pd
    my_pd = my_pd[~(my_pd.title.str.contains('|'.join(not_contain), na=False))]
    my_pd = my_pd.drop(excluding_index_list, errors='ignore')
    
    # Filter online_df
    online_df = online_df[~online_df.texts.duplicated()]
    online_df = online_df[(online_df.texts.str.contains("terrorist", na=False)|online_df.texts.str.contains("terrorism", na=False))]
    # The web-scrape file also holds digital texts for pre-1981 articles (master-list index < 22855).
    # Those articles are covered by the OCR corpus (my_pd) under their true period; keep only the
    # 1981-2019 rows here so no article enters the corpus twice or under the wrong period.
    online_df['index'] = pd.to_numeric(online_df['index'], errors='coerce')
    online_df = online_df[online_df['index'] >= 22855]
    
    # Filter keywords in my_pd
    my_pd = my_pd[(my_pd.text.str.contains("terrorist", na=False)|my_pd.text.str.contains("terrorism", na=False))]
    # Apply Cleaning Chain (my_pd)
    print("Cleaning Historical Data...")
    my_pd['clean'] = my_pd.text.apply(clean_text)
    my_pd['clean_klan'] = my_pd.clean.apply(handle_klan_merger)
    my_pd['clean_sw'] = my_pd.clean_klan.apply(rem_sw)
    my_pd['clean_adsw'] = my_pd.clean_sw.apply(check_additional_sw)
    my_pd['clean_checked'] = my_pd.clean_adsw.apply(check_dictionary)
    my_pd['clean_lemmatized'] = my_pd.clean_checked.apply(check_additional_sw)
    my_pd['clean_lemmatized'] = my_pd.clean_lemmatized.apply(rem_sw)
    my_pd = my_pd.dropna(subset=['clean_lemmatized'])
    my_pd = give_year_labels(my_pd)

    # Apply Cleaning Chain (online_df)
    print("Cleaning Online Data...")
    online_df = online_df[online_df['texts'].apply(lambda x: isinstance(x, str))]
    online_df['clean'] = online_df.texts.apply(clean_text)
    online_df['clean_klan'] = online_df.clean.apply(handle_klan_merger)
    online_df['clean_sw'] = online_df.clean_klan.apply(rem_sw)
    online_df['clean_adsw'] = online_df.clean_sw.apply(check_additional_sw)
    online_df['clean_checked'] = online_df.clean_adsw.apply(check_dictionary)
    online_df['clean_lemmatized'] = online_df.clean_checked.apply(check_additional_sw)
    online_df['clean_lemmatized'] = online_df.clean_lemmatized.apply(rem_sw)
    online_df = online_df.dropna(subset=['clean_lemmatized'])

    # Combine Data
    split_index = 56985
    online_df['label'] = np.where(online_df['index'] >= split_index, '20010911-2019', '1981-20010910')
    
    my_pd_short = my_pd[['label', 'clean_lemmatized']]
    online_df_short = online_df[['label','clean_lemmatized']]
    combined_df = pd.concat([my_pd_short, online_df_short], ignore_index=True)
    
    # 4. Phrase Detection
    print("Detecting collocations (Bigrams/Trigrams)...")
    tokenized_sentences = [text.split() for text in combined_df['clean_lemmatized']]
    bi_grams_tokens, tri_grams_tokens = fetch_bi_grams(tokenized_sentences)
    
    combined_df['clean_bi'] = [' '.join(tokens) for tokens in bi_grams_tokens]
    combined_df['clean_tri'] = [' '.join(tokens) for tokens in tri_grams_tokens]
    combined_df['clean_lemmatized'] = combined_df['clean_tri']
    
    # Final Cleaning Pass (using local functions)
    combined_df['clean_lemmatized_phrased'] = combined_df.clean_lemmatized.apply(check_additional_sw2)
    combined_df['clean_lemmatized_phrased'] = combined_df.clean_lemmatized_phrased.apply(handle_vio_lence_merger)
    combined_df['clean_lemmatized_phrased'] = combined_df.clean_lemmatized_phrased.apply(clean_garbage_text)

    ANALYSIS_COLUMN = 'clean_lemmatized_phrased'
    
    # Save/Load Parquet
    parquet_filename = 'combined_processed_df.parquet'
    combined_df.to_parquet(parquet_filename, index=False)
    print(f"Data saved to {parquet_filename}")
    loaded_parquet_df = pd.read_parquet(parquet_filename)

    # 5. Run Word2Vec
    print("\n--- Starting Word2Vec ---")
    w2v_results = train_dynamic_word2vec(loaded_parquet_df, module_path, ANALYSIS_COLUMN)
    
    if w2v_results:
        final_df = pd.concat(w2v_results, ignore_index=True)
        csv_filename = module_path + "FINAL_ALL_MODELS_RESULTS.csv"
        final_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"Results saved to: {csv_filename}")
        print(final_df.head(20)) # Replaced display()

   # 6. Run Keyword Proximity Check (With Saving)
    print("\n--- Starting Keyword Check ---")
    
    check_keyword_proximity(loaded_parquet_df, ANALYSIS_COLUMN, '1851-1900', 'terroris', 'klan', module_path)
    check_keyword_proximity(loaded_parquet_df, ANALYSIS_COLUMN, '1901-1930', 'terroris', 'night_rider', module_path)

    # 7. Random Forest Classification (With Saving)
    print("\n--- Starting Random Forest Classification ---")
    random_forest_start_time = time.time()

    # (Sampling Logic)
    n_samples = loaded_parquet_df[loaded_parquet_df.label == "1851-1900"].shape[0]
    sampled_pd = pd.concat([
        loaded_parquet_df[loaded_parquet_df.label == "1851-1900"].sample(n=n_samples, replace=True),
        loaded_parquet_df[loaded_parquet_df.label == "1901-1930"].sample(n=n_samples, replace=True),
        loaded_parquet_df[loaded_parquet_df.label == "1931-1950"].sample(n=n_samples, replace=True),
        loaded_parquet_df[loaded_parquet_df.label == "1951-1960"].sample(n=n_samples, replace=True),
        loaded_parquet_df[loaded_parquet_df.label == "1961-1980"].sample(n=n_samples, replace=True),
        loaded_parquet_df[loaded_parquet_df.label == '1981-20010910'].sample(n=n_samples, replace=True),
        loaded_parquet_df[loaded_parquet_df.label == '20010911-2019'].sample(n=n_samples, replace=True)
    ])

    sample_path = module_path + "sample/" # Ensure correct path
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    try:
        print("Vectorizing for RF...")
        my_vec_data = my_vec_fun(sampled_pd[ANALYSIS_COLUMN], 1, 1, sample_path)

        print("Training Random Forest Model...")
        model, metrics, preds, X_test, y_test, y_pred = my_model_fun_grid(
            my_vec_data, sampled_pd.label, sample_path)

        print("\n--- Model Metrics ---")
        print(metrics)
        # [NEW] Save Metrics to CSV
    
        if isinstance(metrics, dict):
            pd.DataFrame([metrics]).to_csv(module_path + "RF_Metrics.csv", index=False)
        else:
            with open(module_path + "RF_Metrics.txt", "w") as f: f.write(str(metrics))
        print(f"Random Forest metrics saved to {module_path}RF_Metrics.csv/txt")

        print("\n--- Top Predictive Words ---")
        feature_names = my_vec_data.columns
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).nlargest(50) # Save Top 50
        print(feat_imp.head(20))
        # [NEW] Save Feature Importance
        feat_imp.to_csv(module_path + "RF_Feature_Importance.csv", header=["Importance"])
        print(f"Feature Importance saved to {module_path}RF_Feature_Importance.csv")

    except NameError:
        print("Error: 'my_vec_fun' or 'my_model_fun_grid' not defined.")

    # 8. Run LDA (With Saving Plots & Data)
    print("\n--- Starting LDA ---")
    # ... (Coherence check logic omitted for brevity, stick to your n_topic=12) ...
    
    print("Preparing LDA Topic Modeling...")
    data_tokens = [str(sent).split() for sent in loaded_parquet_df[ANALYSIS_COLUMN].tolist()]
    id2word = corpora.Dictionary(data_tokens)
    id2word.filter_extremes(no_below=5, no_above=0.5)
    corpus = [id2word.doc2bow(text) for text in data_tokens]
    
    n_topic = 12
    print(f"Training LDA with K={n_topic}...")
    
    lda_model = LdaMulticore(
        corpus=corpus, id2word=id2word, num_topics=n_topic, random_state=123,
        passes=10, workers=4
    )
    
    # [NEW] Save Topic Keywords to CSV
    topics_list = []
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")
        topics_list.append({"Topic_ID": idx, "Keywords": topic})
    pd.DataFrame(topics_list).to_csv(module_path + "LDA_Topic_Keywords.csv", index=False)
    print(f"LDA Keywords saved to {module_path}LDA_Topic_Keywords.csv")

    # Plotting & Saving Data
    print("Generating Plots...")
    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus)
    my_pd_reset = loaded_parquet_df.reset_index(drop=True)
    df_dominant_topic = pd.concat([df_topic_sents_keywords, my_pd_reset[['label']]], axis=1)

    topic_labels = {}
    for i in range(n_topic):
        words = lda_model.show_topic(i, topn=5)
        topic_words = "-".join([w[0] for w in words])
        topic_labels[i] = f"T{i}: {topic_words}"

    topic_counts = df_dominant_topic.groupby(['label', 'Dominant_Topic']).size().unstack(fill_value=0)
    topic_counts_normalized = topic_counts.div(topic_counts.sum(axis=1), axis=0)
    topic_counts_normalized.columns = [topic_labels.get(c, c) for c in topic_counts_normalized.columns]

    # [NEW] Save the Plot Data (CSV)
    topic_counts_normalized.to_csv(module_path + "LDA_Topic_Proportions.csv")
    print(f"LDA Chart Data saved to {module_path}LDA_Topic_Proportions.csv")

    # [NEW] Save the Plot Image (PNG)
    ax = topic_counts_normalized.plot(kind='bar', stacked=True, figsize=(18, 9), colormap='tab20', width=0.85)
    plt.title(f'LDA Topic Prominence (K={n_topic})')
    plt.xlabel('Period')
    plt.ylabel('Proportion')
    
    handles, labels = ax.get_legend_handles_labels()
    handles_reversed = list(reversed(handles))
    labels_reversed = list(reversed(labels))
    plt.legend(handles_reversed, labels_reversed, title='Topic ID & Top 5 Words', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(module_path + "LDA_Topic_Chart.png", dpi=300, bbox_inches='tight') # SAVE BEFORE SHOW
    print(f"LDA Chart Image saved to {module_path}LDA_Topic_Chart.png")
    plt.show() 

    # 9. Dynamic LDA Optimization (With Saving)
    print("\n--- Starting Dynamic LDA Optimization ---")
    SAVE_DIR = "lda_models_final_dynamic"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    stop_list_for_lda = [] 
    
    final_dynamic_results = run_dynamic_lda_optimization(loaded_parquet_df, ANALYSIS_COLUMN, SAVE_DIR, stop_list_for_lda)
    
    # [NEW] Convert Dynamic Results Dictionary to CSV
    # format: Period | Topic_ID | Keywords
    dynamic_rows = []
    for period, topics in final_dynamic_results.items():
        for idx, topic_str in enumerate(topics):
            dynamic_rows.append({"Period": period, "Topic_ID": idx, "Keywords": topic_str})
    
    if dynamic_rows:
        pd.DataFrame(dynamic_rows).to_csv(module_path + "Dynamic_LDA_Results.csv", index=False)
        print(f"Dynamic LDA results saved to {module_path}Dynamic_LDA_Results.csv")

    end_time = time.time()
    
    elapsed_time_all = end_time - start_time
    print("-" * 30)
    print(f"Time Now: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Execution Time: {elapsed_time_all:.2f} seconds ({elapsed_time_all/60:.2f} mins)")
    print("-" * 30)
    print("DONE! All results have been saved to CSV and PNG.")