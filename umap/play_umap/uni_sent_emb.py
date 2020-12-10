#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:57:36 2020

@author: conroy
"""

from sklearn.decomposition import PCA


import tensorflow_hub as hub
import numpy as np
#import tensorflow_text
from utils_umap import run_umap_example
import pickle
from scipy.linalg import svd

NUM_SENTENCES = 2000
en_sentences_file = '/Users/conroy/Downloads/6way/UNv1.0.6way.en'
es_sentences_file = '/Users/conroy/Downloads/6way/UNv1.0.6way.es'

# # Some texts of different lengths.
# english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
# italian_sentences = ["cane", "I cuccioli sono carini.", "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane."]
# japanese_sentences = ["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"]

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
#english_vectors = embed(english_sentences)
ORTHO='pca8'
n_neighbors=[2,4,8]
if NUM_SENTENCES>2000:
    with open(en_sentences_file) as f:
        en_sentences = f.read().splitlines()
    with open(es_sentences_file) as f:
        es_sentences = f.read().splitlines()
    en_vectors = np.array(embed(en_sentences[:NUM_SENTENCES]))
    es_vectors = np.array(embed(es_sentences[:NUM_SENTENCES]))
    vectors = np.vstack((en_vectors,es_vectors))
    labels=NUM_SENTENCES*['en']+NUM_SENTENCES*['es']
else:
    vectors=pickle.load(open('/tmp/vectors.pkl','rb'))
    labels=pickle.load(open('/tmp/labels.pkl','rb'))
if ORTHO=='sign':
    SE=np.sign(vectors[:NUM_SENTENCES])
    SS=np.sign(vectors[NUM_SENTENCES:])
    vectors[:NUM_SENTENCES]=vectors[:NUM_SENTENCES]*SE
    vectors[NUM_SENTENCES:]=vectors[NUM_SENTENCES:]*SS*-1.0
elif ORTHO[:3]=='svd':
    k=int(ORTHO[3:])
    U,s,Vt=svd(vectors[:NUM_SENTENCES])
    vectors[:NUM_SENTENCES,:k]=U[:,:k]*s[:k]
    U,s,Vt=svd(vectors[NUM_SENTENCES:])
    vectors[NUM_SENTENCES:,:k]=U[:,:k]*s[:k]
    vectors=vectors[:,:k]
elif ORTHO[:3]=='pca':
    k=int(ORTHO[3:])
    pca = PCA(n_components=k, svd_solver='full')
    vectors=pca.fit_transform(vectors)

for n_neighbor in n_neighbors:
    feats,labs=run_umap_example(vectors,n_neighbors=n_neighbor,category_labels=labels)