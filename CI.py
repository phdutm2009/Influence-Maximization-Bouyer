import heapq
import networkx as nx
import datetime as dt


def runIC (G, S, p=.01):
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


# l is radius of ball.
# k is number of seed.
l = 4
k = 30
p = 0.01
iteration = 1000
# Insert Data sets
print("start Run")
Time1 = dt.datetime.now()
G = nx.MultiGraph()
file = "p120.txt"
print("Data set:", file)
with open(file) as f:
    for line in f:
        e0, e1 = map(int, line.split())
        try:
            G[e0][e1]["weight"] += 1
        except KeyError:
            G.add_edge(e0, e1)
# CI is list of CI for each nodes.
CI = []
# CI_dic is dictionary of nodes and CI.
CI_dic = {}
ball = {}
# S is list of seed.
S = []
# create ball of radius l centered on each node u.
for u in nx.nodes(G):
    edges = nx.bfs_edges(G, source=u, depth_limit=l)
    nodes = [v for u, v in edges]
    ball[u] = nodes
    d = []
    # calculate degree for each node in ball.
    for i in range(len(nodes)):
        degree_in_ball = G.degree(nodes[i])
        d.append(degree_in_ball)
    sigma_degree_ball = sum(d)
    CI_u = (G.degree(u)-1)*sigma_degree_ball
    CI_dic[u] = CI_u
    CI.append(CI_u)
# print(CI_dic)
# create max_heap for list CI.
heapq._heapify_max(CI)
# delete root of max_heap tree.
root = heapq._heappop_max(CI)
# node_with_maxCI = CI_dic.keys(root)
# print(root)
node_with_maxCI = list(CI_dic.keys())[list(CI_dic.values()).index(root)]
CI_dic.update({node_with_maxCI: 0})
# print(node_with_maxCI)
S.append(node_with_maxCI)
print("Node influence:", avgSize(G, S, p, iteration), "for k=", len(S))
# Recompute CI.
for i in range(k-1):
    # cteate L+1 distnce for node in seed(Figure 4 in article).
    edges = nx.bfs_edges(G, source=S[len(S)-1], depth_limit=l+1)
    nodes = [v for u, v in edges]
    # ud is list of degree.
    ud = []
    # For each node in distance L+1 CI values is updated.
    for j in range(len(nodes)):
        edges = nx.bfs_edges(G, source=nodes[j], depth_limit=l)
        update_nodes = [v for u, v in edges]
        for t in range(len(update_nodes)):
            for e in range(len(S)):
                if(G.number_of_edges(S[e], update_nodes[t])) > 0:
                    degree_in_ball = G.degree(update_nodes[t]) - G.number_of_edges(S[e], update_nodes[t])
                    ud.append(degree_in_ball)
                else:
                    degree_in_ball = G.degree(update_nodes[t])
                    ud.append(degree_in_ball)
        sigma_degree_ball = sum(ud)
        CI_u = (G.degree(nodes[j]) - 1) * sigma_degree_ball
        CI_dic.update({nodes[j]:  CI_u})
    key_CI_dic = list(CI_dic.keys())
    # If node was in seed, node delete of CI_dic.
    for h in range(len(key_CI_dic)):
        if key_CI_dic[h] in S:
            del CI_dic[key_CI_dic[h]]
    CI_update = list(CI_dic.values())
    # create max_heap for list CI_update.
    heapq._heapify_max(CI_update)
    # delete root of max_heap tree.
    root = heapq._heappop_max(CI_update)
    node_with_maxCI = list(CI_dic.keys())[list(CI_dic.values()).index(root)]
    CI_dic.update({node_with_maxCI: 0})
    S.append(node_with_maxCI)
    print("Node influence:", avgSize(G, S, p, iteration), "for k=", len(S))
print("seed:", S)
Time2 = dt.datetime.now()
Totaltime = Time2-Time1
print("Finished Run!!!")
print("Run Time:", int(Totaltime.total_seconds()*1000))
