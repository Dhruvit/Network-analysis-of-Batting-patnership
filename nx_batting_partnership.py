# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:22:35 2020

@author: dpatel
"""

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
#import warnings
from statistics import median

import matplotlib.pyplot as plt

df = pd.read_csv("ipl_batting_partnerships.csv")

#data preparetion
#prepare dataframe for Delhi Capitals
df_dc = df[df['team'] == 'Delhi Capitals']

df_dc['partners'] = [sorted([i,j]) for i, j in zip(df_dc['player_1'], df_dc['player_2'])]
df_dc['partnership'] = ["".join(i) for i in df_dc['partners']]

#empty list to store players name
p1 = []
p2 = []

#empty lists to store median of runs scored
r1 = []
r2 = []

for p in df_dc['partnership'].unique():
    
    temp = df_dc[df_dc['partnership'] == p]
    p1.append(temp.iloc[0]['player_1'])
    p2.append(temp.iloc[0]['player_2'])
    
    a = []
    b = []
    
    #extract individual score for both the players
    for index, row in temp.iterrows():
        # scores of player 1
        a.append(row['score_1'])
        
        # scores of player 2
        b.append(row['score_2'])
        
    # append median of scores
    r1.append(median(a))
    r2.append(median(b))
    
# aggregated batting-partnership data
temp_df = pd.DataFrame({'p1':p1, 'p2':p2, 'r1':r1, 'r2':r2})

#Now that we have the median runs scored by each and every batsman, we can compute the performance metric (overall contribution)
# find the leading batsman
temp_df['lead'] = np.where(temp_df['r1'] >= temp_df['r2'], temp_df['p1'], temp_df['p2'])
temp_df['follower'] = np.where(temp_df['lead'] == temp_df['p1'], temp_df['p2'], temp_df['p1'])
temp_df['large_score'] = np.where(temp_df['r1'] >= temp_df['r2'], temp_df['r1'], temp_df['r2'])
temp_df['total_score'] = temp_df['r1'] + temp_df['r2']

#performance ratio
temp_df['performance'] = temp_df['large_score']/(temp_df['total_score']+0.01)

#construct network
#construct graph
G = nx.from_pandas_edgelist(temp_df, 'follower', 'lead', ['performance'], create_using = nx.MultiDiGraph())

#get edge weights
_, wt = zip(*nx.get_edge_attributes(G, 'performance').items())

#plot graph
plt.figure(figsize=(9,9))
pos = nx.spring_layout(G, k = 20, seed = 21) # k regulates the distance b/w nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=4000, pos=pos, edgelist=G.edges(), edge_color='g', arrowsize=15)
plt.show()

print('The count of all the edges')
print(list(G.degree))

print('The count of all incoming edges')
print(list(G.in_degree))