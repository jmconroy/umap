#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:58:44 2020

@author: conroy
"""

import pandas as pd
import umap
import umap.plot
import time

# Some plotting libraries
#import matplotlib.pyplot as plt
from bokeh.plotting import  output_notebook
from bokeh.resources import INLINE
output_notebook(resources=INLINE)


def run_umap_example(feature_matrix,n_components=2,init="spectral",n_neighbors=2,category_labels=None):
    t0 = time.process_time()
    feature_embedding = umap.UMAP(n_components=n_components,init=init,n_neighbors=n_neighbors).fit(feature_matrix)
    t1 = time.process_time()
    print('Elapsed time for umap %d embedding = %f'%(n_components,t1-t0))
    if category_labels is not None:
        hover_df = pd.DataFrame(category_labels, columns=['category'])
    # For interactive plotting use
    # fig = umap.plot.interactive(tfidf_embedding, labels=dataset.target, hover_data=hover_df, point_size=1)
    # show(fig)
    if n_components==2:
        if category_labels is not None:
            umap.plot.points(feature_embedding, labels=hover_df['category'])
        else:
            umap.plot.points(feature_embedding)
    return feature_embedding,category_labels
