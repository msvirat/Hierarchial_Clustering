# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:19:28 2021

@author: Sathiya vigraman M
"""

#---------Clustering-------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.chdir('D:/Dataset')

main_df = pd.read_excel('University_Clustering.xlsx')

def norm_func(i):
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

scaling_df = norm_func(main_df.iloc[:, 2:9])

#-----hierarchial clustering------

from scipy.cluster.hierarchy import linkage
#import scipy.cluster.hierarchy as sch # for creating dendrogram 

distance_matrix = linkage(scaling_df, method = 'complete', metric = 'euclidean') #creating distance_matrix

from scipy.cluster.hierarchy import dendrogram

dendrogram(distance_matrix, leaf_rotation = 0)#plt all arguments apply for this


from sklearn.cluster import AgglomerativeClustering 

modal_df = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity = 'euclidean')

modal_df.fit(scaling_df)

modal_df.labels_

main_df['Cluster'] = modal_df.labels_

main_df_reorder = main_df.iloc[:, [0, 1, 8, 2, 3, 4, 5, 6, 7]]#inter change colunms

main_df_reorder = main_df_reorder.sort_values(by = ['Cluster']) #sorting by a colunm value

main_df_reorder.to_csv('University_ordered_hieray.csv', index = False) # Save a dataframe to a csv



#-----K means Clustering--------

from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters = 3)

kmeans_model.fit(scaling_df)

kmeans_model.labels_


main_df['Cluster'] = kmeans_model.labels_

main_df_reorder = main_df.iloc[:, [0, 1, 8, 2, 3, 4, 5, 6, 7]]#inter change colunms

main_df_reorder = main_df_reorder.sort_values(by = ['Cluster']) #sorting by a colunm value

main_df_reorder.to_csv('University_ordered_kmeans.csv', index = False) # Save a dataframe to a csv



















