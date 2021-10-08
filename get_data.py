# -*- coding: utf-8 -*-


# IMPORTS

import numpy as np
from collections import Counter

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#!python -m spacy download en_core_web_lg     # if this model has not been installed
import en_core_web_lg

import csv

from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile


# GLOBALS

sid = SentimentIntensityAnalyzer()
NLP = en_core_web_lg.load()
path = get_tmpfile("word2vec_X.model")
data_vectors={}


# ------------------------------------------------
# Semantic vectorizer based on Roget's Thesaurus.

# Unpack and cash necessary files.
with open('Roget_thesaurus.txt', 'r') as index_file:
    raw_index = index_file.read().splitlines()

with open('Roget_pos_tags.txt', 'r') as one_file:
  index_pos_tags=one_file.read().splitlines()

unique_combinations=Counter(index_pos_tags)
frequent_pos_combinations=[]
for k, v in unique_combinations.items():
  if len(k.split())>1 and v>100:
    frequent_pos_combinations.append(k)

with open('Roget_lemmas.txt', 'r') as two_file:
  index_lemmas=two_file.read().lower().splitlines()


# Roget's vectorizer support functions.
def check_ngram_in_roget(ngram):
  pos = index_lemmas.index(ngram)
  item = raw_index[pos]
  nums = item[item.find(' % ')+3:].split(',')[:-1]
  nums = [n[n.find('::')+2:] for n in nums]
  return nums

def get_ngram_indices(ngram):
  list_of_indices = []
  if ngram in index_lemmas:
    list_of_indices = check_ngram_in_roget(ngram)
  else:
    sn = wn.synsets(ngram)
    if len(sn)>0:
      for syns in wn.synsets(ngram):
        hyps = []
        for h in syns.hypernyms():
          extra = str(h)
          extra = extra[extra.find("'")+1:extra.find('.')].replace('_', ' ')
          hyps.append(extra)
        if len(hyps)>0:
          for h in hyps:
            if h in index_lemmas:
              list_of_indices.extend(check_ngram_in_roget(h))

  return list_of_indices


# Roget's vectorizer main function.
# Input: a text of any length
# Output: a semantic vector of length 39
def process_text(text):

  text_split = text.splitlines()
  list_of_sentences = []
  for ts in text_split:
    list_of_sentences.extend(sent_tokenize(ts))

  list_of_collocations=[]
  list_of_indices=[]

  for los in list_of_sentences:

    sentence_spacy=NLP(los)

    for token in sentence_spacy:
      if token.is_alpha == True:
        list_of_indices.extend(get_ngram_indices(token.lemma_.lower()))

    list_of_pos=[ss.pos_ for ss in sentence_spacy]
    string_of_pos=' '.join(list_of_pos)
    for fpc in frequent_pos_combinations:
      if fpc in string_of_pos:
        len_fpc=fpc.count(' ')
        start=string_of_pos.index(fpc)
        index_combination=string_of_pos[:start].count(' ')
        list_of_collocations.append(' '.join([s.lemma_.lower() for s in sentence_spacy[index_combination:index_combination+len_fpc+1]]))

  for loc in list_of_collocations:
    list_of_indices.extend(get_ngram_indices(loc))
  
  semantic_groups=[0]*39
  for loi in list_of_indices:
    pos=int(loi)
    semantic_groups[pos]+=1
  
  return semantic_groups


# ------------------------------------------------
# Word2vec preprocessor.

def w2v(all_texts_as_a_string):

  sentences=[]
  text_spacy = NLP(all_texts_as_a_string)
  for sentence in text_spacy.sents:
    sentences.append([token.lemma_.lower() for token in sentence])

  model=Word2Vec(sentences, min_count=1, size=200, workers=4)
  model.save(path)


# ------------------------------------------------
# Function get_vecs
# Aim: process articles from the Propaganda data set based on their format: number of a text and files with labels.
# It concatenates vectors of word2vec model trained on the texts of articles, a vector of a token and 
# context window made with SpaCy, a sentiment vector of the context window made with NLTK Sentiment Vader processor and
# a semantic vector of the context window made with our Roget's processor.
# Input: cw - length of the context window; name of the file with an article
# Output: a numpy array (type: float) of length 200+300+300+4+39=843

def get_vecs(cw, article):

  num=article[-13:-4]
  if num not in data_vectors:
    data_vectors[num]={'prop':[], 'none':[]}

  with open(article) as text_file:
    text_to_analyze=text_file.read()
  text_spacy=NLP(text_to_analyze)

  labelled_frags=[]
  with open('article'+num+'.task2-TC.labels', 'r') as labels:
    labels=labels.read().splitlines()
  for l in labels:
    l=l.split()
    labelled_frags.append((int(l[2]), int(l[3]), l[1]))

  propaganda_vectors=[]
  not_propaganda_vectors=[]

  length_text=len(text_spacy)

  for i in range(0, length_text):

    start_pos=i-cw
    end_pos=i+cw

    if start_pos<0:
      start_pos=0
    if end_pos>length_text-1:
      end_pos=length_text-1

    token=text_spacy[i]
    tl=token.lower_
    if tl in model.wv.vocab:
      token_w2v=model.wv[token.lower_]
    else:
      token_w2v=np.array([0.0]*200)
    context_vector=text_spacy[start_pos:end_pos].vector
    context_text=text_spacy[start_pos:end_pos].text
    token_start=token.idx
    token_end=token_start+len(token.text)
    sent_vector=list(sid.polarity_scores(context_text).values())
    roget_vector=[float(x) for x in process_text(context_text)]
    concat_vec=np.hstack((token_w2v, token.vector, context_vector, sent_vector, roget_vector))

    prop=0
    for frag in labelled_frags:
      if token_start>=frag[0] and token_end<=frag[1]:
        prop=1
        break
    if prop==0:
      data_vectors[num]['none'].append(([token_start, token_end, token.text], concat_vec))
    else:
      data_vectors[num]['prop'].append(([token_start, token_end, token.text], concat_vec))
    
  return data_vectors


cw=7
article='article111111111.txt'

# Before getting vectors of tokens, unpack and cash as one variable all texts of articles to be processed
# to train a Word2vec model. E.g. here on the example of one article:
with open(article) as text_file:
  text_to_analyze=text_file.read()
w2v(text_to_analyze)
model=Word2Vec.load(path)
# Get vectors of article111111111.txt. Vectors are stored in a dictionary.
art_data_vectors=get_vecs(cw, article)
art_data_vectors['111111111']['none'][30]

