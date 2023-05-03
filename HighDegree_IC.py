import networkx as nx
import random
from copy import deepcopy
import operator
import datetime as t


def runIC (G, S, p):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''

    T = deepcopy(S)# copy already selected nodes

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]:# for neighbors of a selected node
            if v not in T:# if it wasn't selected yet
                w = G.number_of_edges(T[i], v)# count the number of edges between two nodes

                if random.uniform(0, 1) < 1-(1-p)**w:# if at least one of edges propagate influence
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


time1 = t.datetime.now()
print("Start Run:")
p = 0.01
# Read Graph
G = nx.MultiGraph()

with open("hep.txt") as f:
    for line in f:
        e0, e1 = map(int, line.split())
        try:
            G[e0][e1]["weight"] += 1
        except KeyError:
            G.add_edge(e0, e1)
G = G.to_undirected()
print(nx.info(G))
# print(nx.average_clustering(G))

# Print information Graph
n = G.number_of_nodes
m = G.size
print(type(n))
print("Number Nodes: ", G.number_of_nodes())
print("Number Edges: ", G.size())

Degree = G.degree()
Degree = sorted(Degree, key=lambda x: x[1])
Degree = dict(Degree)
print(Degree)
m = list(Degree.keys())
max_Degree = []
for k in range(50):
    max = m.pop()
    max_Degree.append(max)
S = []
for i in range(0, 30):
    D = max_Degree[i]
    S.append(D)
    print("Influence is ", avgSize(G, S, 0.01, 1000), "for k=", i+1, ".")
print("Run finished!!!")
print(S)
time2 = t.datetime.now()
totaltime = time2-time1
print("Run time :", int(totaltime.total_seconds()*1000))