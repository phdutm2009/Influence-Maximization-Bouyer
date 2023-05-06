#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#All defined libraries
import networkx as nx
import operator
import datetime as t
from copy import deepcopy
from random import random
import os
import numpy as np
import pandas as pd
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep


# In[2]:


#ReadFilesToGraph
def ReadFilesToGraph(filePath, NumberOfNodes, start):
    G = nx.Graph()
    for i in range(1, NumberOfNodes + 1):
        G.add_node(i)

    lines = [line.rstrip('\n') for line in open(filePath)]

    for i in range(start, len(lines)):
        edgesNode = lines[i].split(' ')
        G.add_edge(int(edgesNode[0]), int(edgesNode[1]))

    return G


# In[3]:


#SIoT dataset
file = 'NewGraphEdges_Dataset.txt'


# In[4]:


G = ReadFilesToGraph(file, 16216, 0)


# In[ ]:


#ProbDegree algorithm

import networkx as nx
import datetime as rt
import random
import operator

print("start run:")
time1 = rt.datetime.now()
p = 0.03
iteration = 1000
k = 50
# read graph with multi edge.
file = 'NewGraphEdges_Dataset.txt'
print("Data set:", file)
G = nx.MultiGraph()
with open(file) as f:
    for line in f:
        u, v = map(int, line.split())
        try:
            G[u][v]['weight'] += 1
        except:
            G.add_edge(u, v, weight=1)
dp_u_dict = {}
for u in nx.nodes(G):
    d_u = G.degree(u)
    ne = [n for n in G[u]]
    p_d_list = []
    for j in range(len(ne)):
        d_N = G.degree(ne[j])
        p_d = d_N*random.uniform(0, 1)
        p_d_list.append(p_d)
    sigma_p_d = sum(p_d_list)
    dp_u = d_u + sigma_p_d
    dp_u_dict[u] = dp_u
dp_u_dict_sorted = dict(sorted(dp_u_dict.items(), key=operator.itemgetter(1)))
m = list(dp_u_dict_sorted.keys())
seed = []
runnig = True
while runnig:

    max_dp_u = m.pop()
    ne = [n for n in G[max_dp_u]]
    if len(seed) == 0:
        seed.append(max_dp_u)
    else:
        for w in range(len(seed)):
            if seed[w] in ne:
                pass
            else:
                if max_dp_u in seed:
                    pass
                else:
                    seed.append(max_dp_u)
        if len(seed) == k:
            break
print(seed)
S = []
for q in range(len(seed)):
    S.append(seed[q])


# In[ ]:


#Selection of top 50 nodes
top_50 = seed
top_50


# In[10]:


#IC model
def runIC(G, S, p):
    count = 0
    count_each_round = [0,0,0,0,0,0]
    allCount = 0
    allCount_each_round = [0,0,0,0,0,0]
    T = deepcopy(S)  # copy already selected nodes

    # ugly C++ version
    i = 0
    seed_index = 0
    Ts = [T.copy()]
    while seed_index < len(Ts):
        Ts.append([])
        i = 0
        while i < len(Ts[seed_index]):
            for v in G[Ts[seed_index][i]]:  # for neighbors of a selected node
                allCount += 1
                if seed_index < 6:
                    allCount_each_round[seed_index]+=1
                if any(v in sublist for sublist in Ts):  # if it was selected.
                    count += 1
                    if seed_index < 6:
                        count_each_round[seed_index]+=1
                else:
                    w = 1  # count the number of edges between two nodes
                    if random() <= 1 - (1 - p) ** w:  # if at least one of edges propagate influence
                        T[i], 'influences', v
                        T.append(v)
                        Ts[seed_index + 1].append(v)
            i += 1
        if len(Ts[seed_index]) == 0:
            break
        else:
            seed_index += 1
    Ts_Len = []
    Ts_Len = [len(item) for item in Ts[:-2]]

    return T, count / allCount, Ts[:-2], Ts_Len, np.array(count_each_round)/np.array(allCount_each_round)



#Our IC Model!
def avgSize(G, S, p, iterations):
    avg = 0
    c = []
    Ts = []
    Ts_len = []
    failures=[]
    for i in range(iterations):
        f = runIC(G, S, p)
        avg += float(len(f[0])) / iterations
        c.append(f[1])
        Ts.append(f[2])
        Ts_len.append(f[3])
        failures.append(f[4])
    m1 = sum(c) / len(c)
    lens = [len(item) for item in Ts_len]
    m = max(lens)
    for i in range(len(Ts)):
        if len(Ts[i]) < m:
            while len(Ts[i]) < m:
                Ts[i].append([])

    rounds_avg = [0] * m
    for i in range(m):
        s = 0
        for j in range(len(Ts)):
            s += len(Ts[j][i])
        rounds_avg[i] = s / iterations
    return avg, m1, Ts, Ts_len, rounds_avg,failures


# In[11]:


#The influence spread of the first 6 rounds
def filterRound(L, num):
    return sum(L[:num])


# In[12]:


#Find the number of infected nodes with degrees 1, 2 and 3 in the first six rounds of execution
import numpy as np
def find_NIDS(G,l,round=6):
    """ run for each iteration"""
    NIDS1 = [i*0 for i in range(round-1)]            # for each round find a number
    NIDS2 = [i*0 for i in range(round-1)]
    NIDS3 = [i*0 for i in range(round-1)]
    for i in range(1,round):
        for node in l[i]:
            if G.degree(node)==1:
                NIDS1[i-1] += 1
            if G.degree(node)==2:
                NIDS2[i-1] += 1
            if G.degree(node)==3:
                NIDS3[i-1] += 1
    return NIDS1,NIDS2,NIDS3


# In[ ]:


#propagation 
IC0_1 = {}

for i in range (5,51,5):
    print("-------------------------------------------------------")
    #print(alg2_results_id_from_0[0:i])
    IC0_1[i] = avgSize(G, top_50[0:i], 0.03, 1000)
    NIDS1_all = [x*0 for x in range(5)]            # for each round
    NIDS2_all = [x*0 for x in range(5)]
    NIDS3_all = [x*0 for x in range(5)]
#     failures = [i*0 for i in range(round)]            # for each round find a number

    for z in range(1000):
        NIDS1,NIDS2,NIDS3 = find_NIDS(G,IC0_1[i][2][z])
        for k in range(5):
            NIDS1_all[k] += NIDS1[k]
            NIDS2_all[k] += NIDS2[k]
            NIDS3_all[k] += NIDS3[k]
            
    failures =np.sum(np.array(IC0_1[i][5]),axis=0)/1000

#Output:total  influence spread :overall failure rate: influence spread of each round 
#influence spread in the first six rounds
#the number of infected nodes with degree 1 in the first six rounds: the number of infected nodes with degree 2 in the first six rounds: the number of infected nodes with degree 3 in the first six rounds
#failure rate in the first six rounds    
    print(i, ':', IC0_1[i][0], ":", IC0_1[i][1], ":", IC0_1[i][4], "\n", filterRound(IC0_1[i][4],5))
    print("NIDS_1 : ",np.array(NIDS1_all)/1000,"NIDS_2 : ",np.array(NIDS2_all)/1000,"NIDS_3 : ",np.array(NIDS3_all)/1000)
    
    print(failures)

