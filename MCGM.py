import datetime as dt
import networkx as nx
import random
import operator
import community
import math
from copy import deepcopy
import statistics



def runIC (G, S, p=.01):
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
        for v in G[T[i]]:  # for neighbors of a selected node
            if v not in T:  # if it wasn't selected yet
                w = G.number_of_edges(T[i], v)  # count the number of edges between two nodes

                if random.uniform(0, 1) < 1-(1-p)**w:  # if at least one of edges propagate influence
                    # print (T[i], 'influences', v)
                    T.append(v)
        i += 1

    # neat pythonic version
    # legitimate version with dynamically changing list: http://stackoverflow.com/a/15725492/2069858
    # for u in T: # T may increase size during iterations
    #     for v in G[u]: # check whether new node v is influenced by chosen node u
    #         w = G[u][v]['weight']
    #         if v not in T and random() < 1 - (1-p)**w:
    #             T.append(v)
    return T

def avgSize(G, S, p, iterations):
    avg = 0
    for i in range(iterations):
        avg += float(len(runIC(G, S, p)))/iterations
    return avg


# Insert Datasets

G = nx.MultiGraph()
file ="oregon1_010331.txt"
with open(file) as f:
    for line in f:
        e0, e1 = map(int, line.split())
        try:
            G[e0][e1]["weight"] += 1
        except KeyError:
            G.add_edge(e0, e1)
k = 30
iteration = 1000
print("Dataset:", file)
# Print information Graph
number_of_nodess = G.number_of_nodes
m = G.size
print("Number Nodes: ", G.number_of_nodes())
print("Number Edges: ", G.size())
time1 = dt.datetime.now()
degree = dict(G.degree())
k_max = max(list(degree.values()))
median_degree = statistics.median(list(degree.values()))

eigenvector_centrality = dict(nx.eigenvector_centrality_numpy(G))
X_max = max(list(eigenvector_centrality.values()))
X_median = statistics.median(list(eigenvector_centrality.values()))

G_S = nx.Graph(G)
G_S.remove_edges_from(nx.selfloop_edges(G_S))
Core_number = dict(nx.core_number(G_S))
K_S_max = max(list(Core_number.values()))
K_S_median = statistics.median(list(Core_number.values()))

Alpha = (max(median_degree/k_max, X_median/X_max))/(K_S_max/K_S_median)
MCGM = {}
for u in G.nodes():
    Score_u = (degree[u]/k_max)+(Alpha*Core_number[u]/K_S_max)+(eigenvector_centrality[u]/X_max)
    ne = [n for n in G[u]]  # ne is neighbour of u.
    NeighborNeighbor_of_node = []
    for i in range(len(ne)):
        nne = [n for n in G[ne[i]]]
        NeighborNeighbor_of_node.append(nne)
    NeighborNeighbor_of_node = sum(NeighborNeighbor_of_node, [])
    node_in_R2 = ne + NeighborNeighbor_of_node
    # node_in_R2 = sum(node_in_R2, [])
    M = []
    for j in range(len(node_in_R2)):
        d = len(nx.shortest_path(G, source = u, target= node_in_R2[j]))
        Y = (Score_u * ((degree[node_in_R2[j]]/k_max)+(Alpha*Core_number[node_in_R2[j]]/K_S_max)+(eigenvector_centrality[node_in_R2[j]]/X_max)))/d**2
        M.append(Y)
    SUM_MCGM = sum(M)
    MCGM[u] = SUM_MCGM
MCGM_dic = sorted(MCGM.items(), key=operator.itemgetter(1))
MCGM_dic = dict(MCGM_dic)
m = list(MCGM_dic.keys())
max_Inf = []
for k in range(30):
    max = m.pop()
    max_Inf.append(max)
S = []
Result = []
n = [1, 5, 10, 15, 20, 25, 30]
for i in range(0, 30):

    D = max_Inf[i]
    S.append(D)
    print("Influence is ", avgSize(G, S, 0.01, 1000), "for k=", i+1, ".")
    if i+1 in n:
        Result.append(avgSize(G,S, 0.01, 1000))
print("MCGM = ", Result)
print("Run finished!!!")
time2 = dt.datetime.now()
Totaltime = time2 - time1
print("Run Time:", int(Totaltime.total_seconds())*1000)
