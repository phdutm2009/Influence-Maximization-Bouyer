import networkx as nx
import datetime as rt
import operator
import copy
import math
import numpy as np


def runIC (G_base_base, S, p):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    import random
    T = deepcopy(S)  # copy already selected nodes

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G_base[T[i]]:  # for neiG_basehbors of a selected node
            if v not in T:  # if it wasn't selected yet
                w = G_base.number_of_edges(T[i], v)  # count the number of edges between two nodes
                if random.uniform(0, 1) < 1-(1-p)**w:  # if at least one of edges propagate influence
                    # print (T[i], 'influences', v)
                    T.append(v)
        i += 1
    return T


def avgSize(G_base, S, p, iterations):
    avg = 0
    # Influenc_arr = []
    for i in range(iterations):
        # ddd = float(len(runIC(G_base, S, p)))
        avg += float(len(runIC(G_base, S, p)))/iterations
        # Influenc_arr.append(ddd)
    # print(Influenc_arr)

    # print("min", avg - min(Influenc_arr))
    # print("max",  max(Influenc_arr)- avg)
    return avg


print("start run:")
p = 0.01
iteration = 1000
k = 30
l = 3
# read graph with multi edge.
file = "dblp.txt"
print("Data set:", file)
G = nx.Graph()
with open(file) as f:
    for line in f:
        u, v = map(int, line.split())
        try:
            G[u][v]['weight'] += 1
        except:
            G.add_edge(u, v, weight=1)
G_base = nx.MultiGraph()
with open(file) as f:
    for line in f:
        u, v = map(int, line.split())
        try:
            G_base[u][v]['weight'] += 1
        except:
            G_base.add_edge(u, v, weight=1)
print("ass", nx.density(G_base))
############################################################################
time1 = rt.datetime.now()
# Calculate imp for each node in the graph.
degree_all_node = dict(nx.degree(G))
degree_all_node_sorted = dict(sorted(degree_all_node.items(), key=operator.itemgetter(1)))
CC_all_node = dict(nx.clustering(G))
Imp_dic = {}
for v in nx.nodes(G):
    ne0 = [n for n in G[v]]
    degree_nodes_neighbor = {}
    for b in range(len(ne0)):
        degree_nodes_neighbor[ne0[b]] = G.degree(ne0[b])
    nodemax = max(degree_nodes_neighbor.items(), key=operator.itemgetter(1))[0]
    ne1 = [n for n in G_base[nodemax]]
    common_neighbour = list(set(ne0).intersection(ne1))
    Imp_v = degree_all_node[v]-(len(common_neighbour)*(1-CC_all_node[v]))
    Imp_dic[v] = Imp_v

############################################################################
oz = list(Imp_dic.keys())
############################################################################
# Travel for each node
Flag0 = list(nx.nodes(G))
Flag1 = []
Flag2 = []
for r in range(len(oz)):
    path = []
    if len(Flag0) == 0:
        break
    if oz[r] in Flag0:
        Flag0.remove(oz[r])
        path.append(oz[r])
    if oz[r] in Flag1:
        if oz[r] in Flag0:
            Flag0.remove(oz[r])
        continue
    if oz[r] in Flag2:
        if oz[r] in Flag0:
            Flag0.remove(oz[r])
        continue

    Runnig = True
    while Runnig:
        if len(path)== 3:
            Flag2.append(path[2])
            if path[2] in Flag0:
                Flag0.remove(path[2])
            break
        if len(Flag0) == 0:
            break
        ne = [n for n in G[path[len(path) - 1]]]

        Imp_nodes_neighbor = {}
        for b in range(len(ne)):
            Imp_nodes_neighbor[ne[b]] = Imp_dic[ne[b]]

        nodes_max_Imp = []
        for key, value in Imp_nodes_neighbor.items():
            if value > Imp_dic[path[len(path) - 1]]:
                nodes_max_Imp.append(key)
        if len(nodes_max_Imp) == 0:
            if path[len(path) - 1] in Flag0:
                Flag0.remove(path[len(path) - 1])
            Flag2.append(path[len(path) - 1])
            break
        if len(nodes_max_Imp)==1:
            pass
        if len(nodes_max_Imp) > 1:
            za = {}
            for eb in range(len(nodes_max_Imp)):
                za[nodes_max_Imp[eb]] = Imp_dic[nodes_max_Imp[eb]]
                T = list(za.values())
                m = max(T)
                qq = []
                for key, value in za.items():
                    if value < m:
                        qq.append(key)

            for bb in range(len(qq)):
                nodes_max_Imp.remove(qq[bb])
        nodes_less_maxImp = list(set(ne) - set(nodes_max_Imp))

        for az in range(len(nodes_less_maxImp)):
            Flag1.append(nodes_less_maxImp[az])

            if nodes_less_maxImp[az] in Flag0:
                Flag0.remove(nodes_less_maxImp[az])

        if len(nodes_max_Imp) == 0:
            if path[len(path) - 1] in Flag1:
                Flag1.remove(path[len(path) - 1])
            if path[len(path) - 1] in Flag0:
                Flag0.remove(path[len(path) - 1])
            Flag2.append(path[len(path) - 1])

            break

        if len(nodes_max_Imp) == 1:
            if nodes_max_Imp[0] in Flag2:
                if nodes_max_Imp[0] in Flag1:
                    Flag1.remove(nodes_max_Imp[0])
                if nodes_max_Imp[0] in Flag0:
                    Flag0.remove(nodes_max_Imp[0])
                Flag1.append(path[len(path)-1])
                if path[len(path)-1] in Flag0:
                    Flag0.remove(path[len(path)-1])
                break


            if nodes_max_Imp[0] in Flag1:
                if nodes_max_Imp[0] in Flag0:
                    Flag0.remove(nodes_max_Imp[0])
                if len(path)==1:
                    Flag1.append(path[len(path)-1])
                    if path[len(path)-1] in Flag0:
                        Flag0.remove(path[len(path)-1])
                    break
                if len(path)==2:
                    Flag2.append(path[len(path)-1])
                    if path[len(path)-1] in Flag0:
                        Flag0.remove(path[len(path)-1])
                    break
                if nodes_max_Imp[0] in Flag0:
                    Flag0.remove(nodes_max_Imp[0])
            if nodes_max_Imp[0] in Flag0:
                Flag0.remove(nodes_max_Imp[0])
                if path[len(path) - 1] in Flag1:
                    pass
                if path[len(path) - 1] is not Flag1:
                    Flag1.append(path[len(path) - 1])
                    if path[len(path) - 1] in Flag0:
                        Flag0.remove(path[len(path) - 1])
                path.append(nodes_max_Imp[0])
        if len(nodes_max_Imp) > 1:
            node_max_With_Flag1 = []
            node_max_With_Flag0 = []
            node_max_With_Flag2 = []
            for z in range(len(nodes_max_Imp)):
                if nodes_max_Imp[z] in Flag2:
                    node_max_With_Flag2.append(nodes_max_Imp[z])
                if nodes_max_Imp[z] in Flag1:
                    node_max_With_Flag1.append(nodes_max_Imp[z])
                if nodes_max_Imp[z] in Flag0:
                    node_max_With_Flag0.append(nodes_max_Imp[z])
                    node_max_With_Flag0.remove(nodes_max_Imp[z])



            cc_max_Imp_node = {}
            for g in range(len(nodes_max_Imp)):
                cc = nx.clustering(G, nodes=nodes_max_Imp[g])
                cc_max_Imp_node[nodes_max_Imp[g]] = cc
            cc_max_Imp_node_sorted = dict(sorted(cc_max_Imp_node.items(), key=operator.itemgetter(1)))
            RR = list(cc_max_Imp_node_sorted.keys())
            node_with_maxImp_minclustring = RR[0]

            if len(node_max_With_Flag0) > 1:
                cc_max_Imp_node = {}
                for g in range(len(node_max_With_Flag0)):
                    cc = nx.clustering(G, nodes=nodes_max_Imp[g])
                    cc_max_Imp_node[nodes_max_Imp[g]] = cc
                cc_max_Imp_node_sorted = dict(sorted(cc_max_Imp_node.items(), key=operator.itemgetter(1)))
                RR = list(cc_max_Imp_node_sorted.keys())
                node_with_maxImp_minclustring = RR[0]
                if path[len(path) - 1] in Flag1:
                    pass
                if path[len(path) - 1] is not Flag1:
                    Flag1.append(path[len(path) - 1])
                if path[len(path) - 1] in Flag0:
                    Flag0.remove(path[len(path) - 1])
                if path[len(path) - 1] is not Flag0:
                    pass
                path.append(node_with_maxImp_minclustring)
            if len(node_max_With_Flag0) == 0:
                if len(node_max_With_Flag1) > 0:
                    cc_max_Imp_node = {}
                    for g in range(len(node_max_With_Flag1)):
                        cc = nx.clustering(G, nodes=nodes_max_Imp[g])
                        cc_max_Imp_node[nodes_max_Imp[g]] = cc
                    cc_max_Imp_node_sorted = dict(
                        sorted(cc_max_Imp_node.items(), key=operator.itemgetter(1)))
                    RR = list(cc_max_Imp_node_sorted.keys())
                    node_with_maxImp_minclustring = RR[0]
                    if path[len(path) - 1] in Flag1:
                        pass
                    if path[len(path) - 1] is not Flag1:
                        Flag1.append(path[len(path) - 1])
                    if path[len(path) - 1] in Flag0:
                        Flag0.remove(path[len(path) - 1])
                    if path[len(path) - 1] is not Flag0:
                        pass
                    path.append(node_with_maxImp_minclustring)
                if len(node_max_With_Flag1) == 0:
                    if path[len(path) - 1] in Flag1:
                        pass
                    if path[len(path) - 1] is not Flag1:
                        Flag1.append(path[len(path) - 1])
                    if path[len(path) - 1] in Flag0:
                        Flag0.remove(path[len(path) - 1])
                    if path[len(path) - 1] is not Flag0:
                        pass
                    break

############################################################################
print("step 1")
# Add intermediate nodes.
Flag1 = list(dict.fromkeys(Flag1))

Flag1_with_Imp = {}
for se in range(len(Flag1)):
    Flag1_with_Imp[Flag1[se]]=Imp_dic[Flag1[se]]
# print("Flag1_with_Imp", Flag1_with_Imp)
Flag1_with_Imp_dic_sorted = dict(sorted(Flag1_with_Imp.items(), key=operator.itemgetter(1)))
seaz = list(Flag1_with_Imp_dic_sorted.keys())
max_Imp_Flag1_2k = []
for xx in range(2*k):
    max_Imp_Flag1_2k.append(seaz[(len(Flag1_with_Imp)-1)-xx])
clustring_max_Imp_Flag1_2k = {}
for rr in range(len(max_Imp_Flag1_2k)):
    clustring_max_Imp_Flag1_2k[max_Imp_Flag1_2k[rr]] = CC_all_node[max_Imp_Flag1_2k[rr]]
clustring_max_Imp_Flag1_2k_sorted = dict(sorted(clustring_max_Imp_Flag1_2k.items(), key=operator.itemgetter(1)))
key_clustring_max_Imp_Flag1_2k_sorted = list(clustring_max_Imp_Flag1_2k_sorted.keys())
node_Flag1_with_minclustring = []
for dd in range(k):
    node_Flag1_with_minclustring.append(key_clustring_max_Imp_Flag1_2k_sorted[dd])
Flag2 = list(dict.fromkeys(Flag2))
Flag2_with_Imp = {}
for sx in range(len(Flag2)):
    Flag2_with_Imp[Flag2[sx]] = Imp_dic[Flag2[sx]]
Flag2_with_Imp_dic_sorted = dict(sorted(Flag2_with_Imp.items(), key=operator.itemgetter(1)))
kc = list(Flag2_with_Imp_dic_sorted.keys())
print(len(kc))
max_Imp_Flag2_k = []
print(len(Flag2_with_Imp))
for xx in range(k):
    if xx > len(kc):
        break
    else:
        max_Imp_Flag2_k.append(kc[(len(Flag2_with_Imp)-1)-xx])
Flag2 = node_Flag1_with_minclustring+max_Imp_Flag2_k
# print("step2")
nodes_intermediate = copy.deepcopy(Flag2)
for tt in range(len(Flag2)):
    ne = [n for n in G[Flag2[tt]]]
    for jjj in range(len(ne)):
        nne = [n for n in G[ne[jjj]]]
        if Flag2[tt] in nne:
            nne.remove(Flag2[tt])
        for iii in range(len(nne)):
            if nne[iii] in Flag2:
                # print("...",nne[iii])
                if ne[jjj] in Flag2:
                    pass
                else:
                    for fg in range(len(Flag2)):
                        if G.degree(ne[jjj]) > G.degree(Flag2[fg]):
                            nodes_intermediate.append(ne[jjj])

influential_candide_nodes = Flag2 + nodes_intermediate
influential_candide_nodes = list(dict.fromkeys(influential_candide_nodes))
# print("step3")
t1_dic = {}
nne_all = []
Average_Imp_neighbor_dic = {}
###############################################

for i in range(len(influential_candide_nodes)):
    count = 0
    ne = [n for n in G_base[influential_candide_nodes[i]]]
    for j in range(len(ne)):
        for f in range(len(ne)):
            if G_base.has_edge(ne[j], ne[f]):
               count = count + 1
    t1_dic[influential_candide_nodes[i]] = count
# print("rp")

###############################################
for ty in range(len(influential_candide_nodes)):
    ne = [n for n in G_base[influential_candide_nodes[ty]]]
    cne = []
    Imp_neighbor = []
    for bv in range(len(ne)):
        Imp_neighbor.append(Imp_dic[ne[bv]])
        nne = [n for n in G_base[ne[bv]]]
        nne.remove(influential_candide_nodes[ty])
    Average_Imp_neighbor_dic[influential_candide_nodes[ty]]= ((sum(Imp_neighbor)/len(Imp_neighbor))+Imp_dic[influential_candide_nodes[ty]])
# print("1")
pow_dict = {}
for ar in range(len(influential_candide_nodes)):
     pow = (degree_all_node[influential_candide_nodes[ar]]) - ((math.exp(nx.square_clustering(G, nodes=influential_candide_nodes[ar]))*t1_dic[influential_candide_nodes[ar]])/Average_Imp_neighbor_dic[influential_candide_nodes[ar]])
     pow_dict[influential_candide_nodes[ar]] = pow
pow_dic_sorted = dict(sorted(pow_dict.items(), key=operator.itemgetter(1), reverse=True))
time2 = rt.datetime.now()
totaltime = time2-time1
print("Run time:", totaltime.total_seconds()*1000)
max_Inf = list(pow_dic_sorted.keys())
S = []
for i in range(0, 30):
    D = max_Inf[i]
    S.append(D)
    print("Influence is ", avgSize(G_base, S, 0.01, 1000), "for k=", i+1, ".")
print("Run finished!!!")
print("Seed=", S)

