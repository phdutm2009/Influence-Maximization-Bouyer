import datetime as t
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import asyn_fluidc
from functools import reduce
import itertools
import random
import operator
from copy import deepcopy
import community
import itertools


def runIC (G_IC, S, p=.01):
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
        for v in G_IC[T[i]]:  # for neighbors of a selected node
            if v not in T:  # if it wasn't selected yet
                w = G_IC.number_of_edges(T[i], v)  # count the number of edges between two nodes

                if random.uniform(0, 1) < 1-(1-p)**w:  # if at least one of edges propagate influence
                    # print (T[i], 'influences', v)
                    T.append(v)
        i += 1

    return T

def avgSize(G_IC, S, p, iterations):
    avg = 0
    for i in range(iterations):
        avg += float(len(runIC(G_IC, S, p)))/iterations
    return avg


# Insert Datasets
G_IC = nx.MultiGraph()
file = "oregon1_010331.txt"
with open(file) as f:
    for line in f:
        e0, e1 = map(int, line.split())
        try:
            G_IC[e0][e1]["weight"] += 1
        except KeyError:
            G_IC.add_edge(e0, e1)

iteration = 100
print("Dataset:", file)
print(nx.info(G_IC, n= None))
G = nx.read_edgelist(file)
G = G.to_undirected()
# Print information Graph
number_of_nodess = G.number_of_nodes
m = G.size
print("Number Nodes: ", G_IC.number_of_nodes())
print("Number Edges: ", G_IC.size())
time1 = t.datetime.now()
# Community detection algorithm
partition = community.best_partition(G_IC)
count = 0
c = []
for com in set(partition.values()):
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
    c.append(list_nodes)
# print("Number of  community: ", len(c), count)
i = 0

max_cores = []
running = True
while running:
    if i == len(c):
        break
    # n is number of node in the community.
    n = len(c[i])

    # create Graph in community
    results = [int(m) for m in c[i]]
    # print("community",i,":",results)
    Com_of_graph = nx.Graph()
    Com_of_graph.add_nodes_from(results)
    number_of_com_node = Com_of_graph.number_of_nodes()
    print("Node of community: ", number_of_com_node)
    for al in range(len(results)):
        list_of_edge = []
        edges = G_IC.edges(results[al])
        second_element = list(list(zip(*edges))[1])
        for se in range(len(second_element)):
            if second_element[se] in results:
                list_of_edge.append((results[al], second_element[se]))
        Com_of_graph.add_edges_from(list_of_edge)
    print("Node of c: ", Com_of_graph.size())

    # u1 = []
    # v1 = []
    # for j in range(0, number_of_com_node-1):
    #     for k in range(0, number_of_com_node-1):
    #         u = results[j]
    #         v = results[k]
    #         if G.number_of_edges(str(u), str(v)) > 0:
    #             u1.append(u)
    #             v1.append(v)
    # '''
    #     for list_u1 in range(0, len(u1)):
    #         for list_v1 in range(0,len(v1)):
    #             if u1[list_u1]==v1[list_v1]and u1[list_v1]==v1[list_u1]:
    #                 del u1[list_v1]
    #                 del v1[list_v1]
    #                 len(u1)-1
    #                 len(v1)-1
    #                 #print("list_v1:",list_v1)
    #                 #print("list_u1:",list_u1)
    #
    # '''
    # # print("u1:",u1)
    # # print("v1:",v1)
    # edge_of_from = list(zip(u1, v1))
    # # print(edge_of_from)
    # Com_of_graph.add_edges_from(edge_of_from)
    Com_of_graph.remove_edges_from(nx.selfloop_edges(Com_of_graph))
    core_number = nx.core_number(Com_of_graph)
    # print("core_number in community ",i, ":",core_number)
    # Find Max core in community
    max_core = [key for m in [max(core_number.values())] for key, val in core_number.items() if val == m]
    # print("max_core in community ", i, ":", max_core)
    max_cores.append(max_core)
    i = i+1
# create simple list of list of list
max_cores_isolate = reduce(operator.concat, max_cores)

# Detecte neighbor between Max_cores in community
new_c = []
for zz in range(len(c)):
    ccc = [int(m) for m in c[zz]]
    new_c.append(ccc)
max_core_is_neghibor1 = []
max_core_is_neghibor2 = []
com_linked1 = []
com_linked2 = []
count1 = 0
count2 = 0
e = 0
while running:
    if e == len(max_cores):
        break
    mc = [int(w) for w in max_cores[e]]
    for nnn in range(0, len(mc)-1):
        for mmm in range(0, len(max_cores)-1):
            node_maxcore1 = mc[nnn]
            node_maxcore2 = max_cores_isolate[mmm]
            if G.number_of_edges(str(node_maxcore1), str(node_maxcore2)) > 0:
                for ll in range(len(new_c)):
                    item = new_c[ll]
                    for j in range(len(item)):
                        if node_maxcore1 == item[j]:
                            # print("one node with max core is in ", "(", ll, j, ").")
                            # count1=count1+1
                            com_linked1.append(ll)
                for i in range(len(new_c)):
                    item = new_c[i]
                    for j in range(len(item)):
                        if node_maxcore2 == item[j]:
                            # print("Other node with max core is in ", "(", i, j, ").")
                            com_linked2.append(i)

                break
    # print(count1)
    # print(count2)
    e = e+1
# print(com_linked)
# print(len(com_linked1), len(com_linked2))
# print(type(max_cores_isolate))
com_linked11 = []
com_linked22 = []
for j in range(len(com_linked1)):
    # print(j)
    if com_linked1[j] != com_linked2[j]:
        com_linked11.append(com_linked1[j])
        com_linked22.append(com_linked2[j])
# print(len(com_linked11), len(com_linked22))
# print(com_linked11)
# print(com_linked22)
com_linked11_without_duplicate1 = []
com_linked11_without_duplicate2 = []
for ii in range(len(com_linked11)):
    if com_linked11[ii] in com_linked11_without_duplicate1 and com_linked22[ii]in com_linked11_without_duplicate2:
        pass
    else:
        com_linked11_without_duplicate1.append(com_linked11[ii])
        com_linked11_without_duplicate2.append(com_linked22[ii])
# print(com_linked11_without_duplicate1)
# print(com_linked11_without_duplicate2)

# merge communities that have neighbor's max_cores
d = []
merged = []
new_com_merge = []
x = [None]*174161
seed = []
list_of_score = []
teta_com = []  # teta is list of teta for each community.
# teta is[number of node for each community/(number of edge for each community +1)]
for i in range(len(com_linked11_without_duplicate1)):
    if com_linked11_without_duplicate1[i] in merged and com_linked11_without_duplicate2 in merged:
        pass
    else:
        x[com_linked11_without_duplicate1[i]]=com_linked11_without_duplicate2[i]
        merged.append(com_linked11_without_duplicate1[i])
        merged.append(com_linked11_without_duplicate2[i])
    if com_linked11_without_duplicate1[i] in merged or com_linked11_without_duplicate2 in merged:
        if com_linked11_without_duplicate2[i] in merged:
            if x[com_linked11_without_duplicate2[i]] == None:
                x[com_linked11_without_duplicate1[i]] = com_linked11_without_duplicate2[i]
                merged.append(com_linked11_without_duplicate1[i])
            else:
                x[com_linked11_without_duplicate1[i]] = x[com_linked11_without_duplicate2[i]]
                merged.append(com_linked11_without_duplicate1[i])
# print(x)
f = list(set(com_linked11_without_duplicate2))
for j in range(len(f)):
    d = []
    if f[j] in x:
        if new_c[f[j]] in d:
            pass
        else:
            d.append(new_c[f[j]])
        a = f[j]
        idx = [i for i in range(len(x)) if x[i] == a]
        for kk in range(len(idx)):
            d.append(new_c[idx[kk]])
    if len(d) == 0:
        pass
    else:
        ff = sum(d, [])
        new_com_merge.append(ff)
new_com_merge.sort()
new_com_merge = list(new_com_merge for new_com_merge, _ in itertools.groupby(new_com_merge))
# print("merged", merged)

for i in range(len(new_c)):
    if i in merged:
        pass
    else:
        new_com_merge.append(new_c[i])
# create Graph in new_com_merge
i = 0
Influence_of_node = {}
Degree_of_node = {}
Neighbor = {}
NeighborNeighbor = {}
while running:
    S = []
    if i == len(new_com_merge):
        break
    results = [int(m) for m in new_com_merge[i]]
    # print("----------------------------------------------------------------------")
    # print("community", i, ":")
    Com_of_graph_max_core = nx.Graph()
    Com_of_graph_max_core.add_nodes_from(results)
    number_of_com_node = Com_of_graph_max_core.number_of_nodes()
    # print("Node of community: ", number_of_com_node)
    for al in range(len(results)):
        list_of_edge = []
        edges = G_IC.edges(results[al])
        second_element = list(list(zip(*edges))[1])
        for se in range(len(second_element)):
            if second_element[se] in results:
                list_of_edge.append((results[al], second_element[se]))
        Com_of_graph_max_core.add_edges_from(list_of_edge)
    # u1 = []
    # v1 = []
    # for j in range(number_of_com_node-1):
    #     for k in range(number_of_com_node-1):
    #         u = results[j]
    #         v = results[k]
    #         if G.number_of_edges(str(u), str(v)) > 0:
    #             u1.append(u)
    #             v1.append(v)
    # edge_of_from = list(zip(u1, v1))
    # Com_of_graph_max_core.add_edges_from(edge_of_from)
    teta = number_of_com_node/(Com_of_graph_max_core.size()+1)
    teta_com.append(teta)
    # print("number of node", nx.number_of_nodes(Com_of_graph_max_core))
    print("teta", teta)
    # rrrrr=[1251, 1251, 31, 1533, 960, 880, 1540, 1408, 338, 1312, 225, 1937]
    # for ggg in range(len(rrrrr)):
    #       if Com_of_graph_max_core.has_node(rrrrr[ggg]):
    #           print(rrrrr[ggg], "is community", i)
    #
    # print("node:", Com_of_graph_max_core.number_of_nodes())
    # print(Com_of_graph_max_core.size())
    p_score = random.uniform(0, 1)
    score_dic = {}
    for l in range(Com_of_graph_max_core.number_of_nodes()):
        mg = []
        mmg = []
        ne = [n for n in Com_of_graph_max_core[results[l]]]
        Neighbor[results[l]] = ne
        # print("Node:", results[l], "--", "Neigbhor:", ne)
        NeighborNeighbor_of_node = []
        for jjj in range(len(ne)):
            nne = [n for n in Com_of_graph_max_core[ne[jjj]]]
            NeighborNeighbor_of_node.append(nne)
            # print("nne", nne)
        NeighborNeighbor_of_node = sum(NeighborNeighbor_of_node, [])
        NeighborNeighbor[results[l]] = NeighborNeighbor_of_node
        # Degree_for_node calculate degree for each node in community for add to Degree_of_node dictionary.
        Degree_for_node = Com_of_graph_max_core.degree(results[l])
        Degree_of_node[results[l]] = Degree_for_node
        # print("NeighborNeighbor_of_node", NeighborNeighbor_of_node)
        h = 0
        while running:
            if h == len(ne):
                break
            Neigbhor_degree = Com_of_graph_max_core.degree(ne[h])
            mg.append(Neigbhor_degree)
            # print(Neigbhor_degree)
            h = h+1
        sigma_Neigbhor_degree = sum(mg)
        # print("sigma_Neigbhor_degree node", results[l], "--->", sigma_Neigbhor_degree)
        g = 0
        while running:
            if g == len(NeighborNeighbor_of_node):
                break
            NeigbhorNeigbhor_degree = Com_of_graph_max_core.degree(NeighborNeighbor_of_node[g])
            mmg.append(NeigbhorNeigbhor_degree)
            g = g+1
        sigma_NeigbhorNeigbhor_degree = sum(mmg)
        # print("sigma_NeigbhorNeigbhor_degree node", results[l], "--->", sigma_NeigbhorNeigbhor_degree)
        score = p_score*(sigma_Neigbhor_degree*sigma_NeigbhorNeigbhor_degree)
        # print("Score", results[l], "--->", score)
        score_dic[results[l]] = score
    list_of_score.append(score_dic)
    max_score = max(score_dic.items(), key=operator.itemgetter(1))[0]
    # print(max_score)
    S.append(max_score)
    Influence = avgSize(G_IC, S, 0.01, iteration)
    Influence_of_node[max_score] = Influence
    # print("Influence", max_score, ":", Influence)
    i = i+1
print("number of community after finding max core's neighbor: ", len(new_com_merge))

max_Influence = max(Influence_of_node.items(), key=operator.itemgetter(1))[0]
print("node's Influence:", Influence_of_node[max_Influence], "for k=1")
seed.append(max_Influence)
# Influence_of_node is dictionary of influence for each node.
del Influence_of_node[max_Influence]
k = 0
Influence_of_node = {}
while running:
    if len(seed) == 30:
        break
    # find neighbor of seed.
    NOS = []
    # calculate neighbor for each seed node.
    for i in range(len(seed)):
        NO = Neighbor[seed[i]]
        NOS.append(NOS)
        Neighbor_of_seed = sum(NOS, [])
    S = []
    # update scores.
    for j in range(len(list_of_score)):
        if teta_com[j] > 1.5:
            pass
        else:
            ha = list_of_score[j]
            N = []
            # keys put in list N. keys is node in community.
            for key in ha:
                N.append(key)
            # print(N)
            # score for each seed is zero.
            for s in range(len(seed)):
                if seed[s] in N:
                    ha.update({seed[s]: 0})
            # If node was seed, score for seed node is zero.
            for ui in range(len(N)):
                if N[ui] in seed:
                    ha.update({N[ui]: 0})
                else:
                    # ne is nighbor of list N[ui].
                    ne = Neighbor[N[ui]]
                    de = []
                    dde = []
                    # If nodes within ne was in Neighbor_of_seed, degree must update.
                    # de is list of neighbor degree update.
                    # dde is list of neighborneighbor degree update.
                    for xx in range(len(ne)):
                        if ne[xx] in Neighbor_of_seed:
                            df = NeighborNeighbor[ne[xx]]
                            for hh in range(len(df)):
                                dde = Degree_of_node[df[hh]]
                            sigma_dde = sum(dde)
                            # UD is update for degree.
                            UD = Degree_of_node[ne[xx]]-(1/(Degree_of_node[ne[xx]] + sigma_dde))
                            de.append(UD)
                        # if nodes within ne was in seed degree must update zero.
                        elif ne[xx] in seed:
                            Degree_of_node.update({ne[xx]: 0})
                            UD = Degree_of_node[ne[xx]]
                            de.append(UD)
                        # if nodes within ne wasnt is seed, degree shouldnt update.
                        elif ne[xx] not in seed:
                            UD = Degree_of_node[ne[xx]]
                            de.append(UD)
                    # nne is neighbor of neighbor N[ui] node.
                    nne = NeighborNeighbor[N[ui]]
                    dde = []
                    for ss in range(len(nne)):
                        # if nodes within nne was in seed degree must update zero.
                        if nne[ss] in seed:
                            Degree_of_node.update({nne[ss]: 0})
                            # UDD is update neighbor of neighbor.
                            UDD = Degree_of_node[nne[ss]]
                            dde.append(UDD)
                        else:
                            UDD = Degree_of_node[nne[ss]]
                            dde.append(UDD)
                    sigma_Neigbhor_degree = sum(de)
                    sigma_NeigbhorNeigbhor_degree = sum(dde)
                    # score updates.
                    update_score = p_score * (sigma_Neigbhor_degree * sigma_NeigbhorNeigbhor_degree)
                    ha.update({N[ui]: update_score})
            # find a node with maximimum score in each community.
            max_score = max(ha.items(), key=operator.itemgetter(1))[0]
            S = []
            S = deepcopy(seed)
            S.append(max_score)
            # print("S",S)
            # calculate influence for max_score in each community.
            Influence = avgSize(G_IC, S, 0.01, iteration)
            Influence_of_node[max_score] = Influence
    # find a node with max influence between all communities.
    max_Influence = max(Influence_of_node.items(), key=operator.itemgetter(1))[0]
    seed.append(max_Influence)
    S = deepcopy(seed)
    # print("S", S)
    print("nodes influence", Influence_of_node[max_Influence], "for k=", len(seed))
    del Influence_of_node[max_Influence]
    k = k+1
time2 = t.datetime.now()
print("seed", seed)
totaltime = time2 - time1
print("Run Time:", int(totaltime.total_seconds()*1000))
