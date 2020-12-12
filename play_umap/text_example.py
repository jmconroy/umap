#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:58:57 2020

@author: conroy

"""


import pandas as pd
import umap
import umap.plot
import time

# Used to get the data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans




def run_umap_text_example(dataset,n_components=2,init='spectral'):
    tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words='english')
    tfidf_word_doc_matrix = tfidf_vectorizer.fit_transform(dataset.data)
    t0 = time.process_time()
    tfidf_embedding = umap.UMAP(metric='hellinger',n_components=n_components,
                                init=init).fit(tfidf_word_doc_matrix)
    t1 = time.process_time()
    print('Elapsed time for umap %d embedding = %f'%(n_components,t1-t0))
    category_labels = [dataset.target_names[x] for x in dataset.target]
    hover_df = pd.DataFrame(category_labels, columns=['category'])
    # For interactive plotting use
    # fig = umap.plot.interactive(tfidf_embedding, labels=dataset.target, hover_data=hover_df, point_size=1)
    # show(fig)
    if n_components==2:
        umap.plot.points(tfidf_embedding, labels=hover_df['category'])
    return tfidf_embedding,category_labels
dataset = fetch_20newsgroups(subset='all',
                             shuffle=True, random_state=42)



inits =['spectral_scaled_adjacency', 'spectral_normalized_Laplacian','spectral_adjacency']

for init in inits:
    for n in [2,10,100,200]:
        tfidf_embedding,cats=run_umap_text_example(dataset,n_components=n,
                                                   init=init)
        clusterer = KMeans(n_clusters=20)
        E=tfidf_embedding.embedding_
        cats_hat = clusterer.fit(E).labels_
        ari=adjusted_rand_score(cats_hat,cats)
        print('ARI=%f'%(ari))


# OLD
# ORG init time: 1.498785
# Elapsed time for umap 2 embedding = 221.436308
# ARI=0.407215
# OLD
# ORG init time: 2.917165
# Elapsed time for umap 10 embedding = 227.861244
# ARI=0.432704
# OLD
# ORG init time: 25.970898
# Elapsed time for umap 100 embedding = 312.192934
# ARI=0.427276
# OLD
# ORG init time: 75.456692
# Elapsed time for umap 200 embedding = 442.809990
# ARI=0.416964

# NEW
# NEW init time: 0.888342
# Elapsed time for umap 2 embedding = 219.953054
# ARI=0.431387
# NEW
# NEW init time: 1.687691
# Elapsed time for umap 10 embedding = 226.890066
# ARI=0.433861
# NEW
# NEW init time: 17.669248
# Elapsed time for umap 100 embedding = 294.917300
# ARI=0.427587
# NEW
# NEW init time: 45.480090
# Elapsed time for umap 200 embedding = 381.690496
# ARI=0.398898
    
# 2    1.498785  0.888342
# 10   2.917165  1.687691
# 100 25.970898 17.669248
# 200 75.456692 45.480090