import networkx as nx
from operator import itemgetter
import datetime as dt


def runIC (G, S, p):
    '''
     Runs independent cascade model.
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


k = 30  # K is number of seed.
LI = 0
LI_dic = {}  # LI_dic is dictionary of node as key and LI as value.
iteration = 1000  # iteration is number of repeat for monte-carlo.
p = 0.01
# creat graph.
Time1 = dt.datetime.now()
G = nx.MultiGraph()
file = "out.as20000102.txt"
print("Data set:", file)
with open(file) as f:
    for line in f:
        e0, e1 = map(int, line.split())
        try:
            G[e0][e1]["weight"] += 1
        except KeyError:
            G.add_edge(e0, e1)

# Calculate LI for each node.
for u in G.nodes():
    ne = [n for n in G[u]]  # ne is neighbour of u.
    LI = 0
    for i in range(len(ne)):
        # if neighbour degree was greater than node u,LI increases by one unit.
        if G.degree(u) < G.degree(ne[i]):
            LI = LI+1
    LI_dic[u] = LI
LI_0_dic = {k: v for k, v in LI_dic.items() if v == 0}
node_LI_0 = list(LI_0_dic)
Degree_LI_0 = {}
for j in range(len(node_LI_0)):
    degrees = G.degree(node_LI_0[j])
    Degree_LI_0[node_LI_0[j]] = degrees
d = []
for key, value in sorted(Degree_LI_0.items(), key=itemgetter(1), reverse=True):
    d.append(key)
print(d)
S = []
for h in range(k):
    S.append(d[h])
    print("Node influence:", avgSize(G, S, p, iteration), "for k=", len(S))
Time2 = dt.datetime.now()
Totaltime = Time2 - Time1
print("Run Time:", int(Totaltime.total_seconds()*1000))
print(S)