import networkx as nx
import community
import datetime as dt
import math
import random
from networkx.algorithms.community import k_clique_communities
from igraph import *
from collections import Counter
from collections import defaultdict
import operator
import datetime as dt

def dsum(*dicts):
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)


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
k = 30
T = 30
Z = 30
# Insert Data sets
print("start Run")
G = nx.MultiGraph()
file = "gr.txt"
print("Data set:", file)
with open(file) as f:
    for line in f:
        e0, e1 = map(int, line.split())
        try:
            G[e0][e1]["weight"] += 1
        except KeyError:
            G.add_edge(e0, e1)
time1 = dt.datetime.now()
print(list(nx.find_cliques(G)))
c = list(k_clique_communities(G, 3))

running = True
tc = 0
co = []
while running:
    if tc == len(c):
        break
    results = [int(m) for m in c[tc]]
    co.append(results)
    tc = tc + 1
all_node_in_all_communities = sum(co, [])

n_t = G.number_of_nodes()
e_t = G.size()  # e_t is number of edges in the graph.
print("Number Nodes: ", n_t)
print("Number Edges: ", e_t)

# Community detection algorithm
# Time1 = dt.datetime.now()
# partition = community.best_partition(G)
# Time2 = dt.datetime.now()
# Totaltime = Time2 - Time1
# print("Run Time for louvain: ", int(Totaltime.total_seconds()))
# count = 0
# c = []
# for com in set(partition.values()):
#     count = count + 1.
#     list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
#     c.append(list_nodes)
c = sorted(c, key=len, reverse=True)
print("Number of communitiy:", len(c))
print("------------------------------------------")
node_out = {}  # node_out is dictionary of nodes as key and nodes neighbor in other communities.
edge = {}
degreenode_into_communitiy_dic = {}  # degreenode_into_community_dic is community index as key and node's degree
#  into community as value.
node_with_HighDegree_dic = {}  # node_with_HighDegree_dic is community index as key and node with
# max degree in community as value.
x_sp_p_ar_dic = {}
HighDegree_in_community = {}
Index_of_node_community = {}
Index_community_relation_with_node = {}
h = []
i = 0
nccc = []
D = {}
P_do = {}
P_dI = {}
normalize_P_dI = {}  # normalize_P_dI is normalized of P_dI
gh_dic = {}
gh_dic_duplicatenode = {}
# co = []  # list of communities.
n1 = {}  # n1 is number of nodes that relation with other communities and put in neighbor of node(level 1).
n2 = {}  # n2 is number of nodes that relation with other communities and put in neighbor of neighbor node(level 2).
w_c_dic = {}
DN_dic = {}  # DN_dic is dictionary for DN.
sigma_NeigbhorNeigbhor_degree_dic = {}
Degree_of_node = {}
NeighborNeighbor = {}

running = True
while running:
    if i == len(c):
        break
    # n is number of node in the community.
    n = len(c[i])
    # create Graph in community
    results = [int(m) for m in c[i]]
    Com_of_graph = nx.Graph()
    Com_of_graph.add_nodes_from(results)
    n_c = Com_of_graph.number_of_nodes()  # n_c is number of node in community.
    # print("Node of community: ", n_c)
    nccc.append(n_c)
    e = []
    e_out_node = 0
    no = []
    node_in_community_to_out = []  # node_in_community_to_out is list of nodes that node has edge with other communities
    for al in range(len(results)):
        Index_of_node_community[results[al]] = i
        list_of_edge = []
        edges = list(G.edges(results[al]))
        D[results[al]] = Com_of_graph.degree(results[al])
        edge[results[al]] = edges
        second_element_edges = list(list(zip(*edges))[1])
        h = []
        for lk in range(len(second_element_edges)):
            if second_element_edges[lk] in all_node_in_all_communities:
                h.append(second_element_edges[lk])
        list_of_e_out_node = []
        for se in range(len(h)):
            if h[se] in results:
                list_of_edge.append((results[al], h[se]))
            else:
                e_out_node = e_out_node+1

        second_element_listofedge = list(list(zip(*list_of_edge))[1])
        # node_out_community is list of nodes  that those have edge with other communities.
        node_out_community = list(set(h) - set(second_element_listofedge))

        # if len(node_out_community) > 0:
        #     for tt in range(len(node_out_community)+1):
        #         print("node_out_community[tt]", tt , node_out_community[tt])
        #         if (node_out_community[tt] in all_node_in_all_communities):
        #             pass
        #         else:
        #             node_out_community.remove(node_out_community[tt])
        if len(node_out_community) > 0:
            node_out[results[al]] = node_out_community
            node_in_community_to_out.append(results[al])
        if e_out_node > 0:
            no.append(results[al])
            list_of_e_out_node.append(e_out_node)
        Com_of_graph.add_edges_from(list_of_edge)
    # print("density:", nx.density(Com_of_graph))
    # print("node_in_community_to_out", node_in_community_to_out)
    for ul in range(len(results)):
        if results[ul] in node_in_community_to_out:
            pass
        else:
            edges = nx.bfs_edges(Com_of_graph, source=results[ul], depth_limit=1)
            nodes1 = [v for u, v in edges]
            zp1 = list(set(node_in_community_to_out).intersection(nodes1))
            n1[results[ul]] = (len(zp1))
            edges = nx.bfs_edges(Com_of_graph, source=results[ul], depth_limit=2)
            nodes2 = [v for u, v in edges]
            zp2 = list(set(node_in_community_to_out).intersection(nodes2))
            n2[results[ul]] = 2*(math.sqrt(len(zp2) - len(zp1)))
    e_o = sum(list_of_e_out_node)
    e_c = Com_of_graph.size()
    Degree = dict(Com_of_graph.degree())
    degreenode_into_communitiy_dic[i] = Degree
    maxvalue = max(Degree.values())
    node_with_HighDegree = [key for key in Degree.keys() if Degree[key] == maxvalue]
    node_with_HighDegree_dic[i] = node_with_HighDegree
    f = []
    for d in range(len(node_with_HighDegree)):
        q = Com_of_graph.degree(node_with_HighDegree[d])
        f.append(q)
        h.append(node_with_HighDegree[d])
    HighDegree_in_community[i] = q
    # print("node_with_HighDegree", node_with_HighDegree)
    # print(max(Degree.items(), key=lambda k: k[1]))
    disp = nx.dispersion(Com_of_graph, normalized=True)
    # print("disperson:", disp)
    # print("Edge of community: ", Com_of_graph.size())
    zz = disp.values()
    zz = list(zz)
    # print("ZZ", zz)
    sum_of_dispersion = []
    for j in range(len(zz)):
        a = zz[j]
        a = dict(a)
        s = sum(a.values())
        s = s/len(zz[j])  # len(zz[j]) is neighbors of node.
        sum_of_dispersion.append(s)
    sum_disp = sum(sum_of_dispersion)
    # print("sum_of_dispersion= ", sum_disp)
    assortativity = nx.degree_assortativity_coefficient(Com_of_graph)

    w_c = (((e_c + e_o) / e_t)*(n_c/n_t) + (math.log(sum_disp+1) + assortativity)*(n_c/n_t))


    w_c_dic[i] = w_c
    for l in range(len(node_with_HighDegree)):
        for w in range(len(node_in_community_to_out)):
            if nx.has_path(Com_of_graph, node_with_HighDegree[l], node_in_community_to_out[w]):
                x_sp = nx.shortest_path_length(Com_of_graph, node_with_HighDegree[l], node_in_community_to_out[w])
                # print("node_with_HighDegree", node_with_HighDegree[l], "has", x_sp, "lengh with",
                #       node_in_community_to_out[w])
                if x_sp == 0:
                    P_do[node_in_community_to_out[w]] = 1
                else:
                    p_ar = random.uniform(0, 1)
                    x_sp_p_ar = x_sp*p_ar
                    HD_c = Com_of_graph.degree(node_with_HighDegree[l])
                    gh = HD_c/x_sp_p_ar

                    if node_in_community_to_out[w] in gh_dic:
                        gh_dic_duplicatenode[node_in_community_to_out[w]] = gh
                    else:
                        gh_dic[node_in_community_to_out[w]] = gh
                    x_sp_p_ar_dic[node_in_community_to_out[w]] = x_sp_p_ar
                    # print(node_in_community_to_out[w], "x_sp_p_ar = ", x_sp_p_ar)
            else:
                # print("p_d is zero.")
                x_sp_p_ar = 0
                x_sp_p_ar_dic[node_in_community_to_out[w]] = x_sp_p_ar
                P_do[node_in_community_to_out[w]] = 0

    # calculate DN.
    for l in range(Com_of_graph.number_of_nodes()):
        mg = []
        mmg = []
        ne = [n for n in Com_of_graph[results[l]]]
        NeighborNeighbor_of_node = []
        for jjj in range(len(ne)):
            nne = [n for n in Com_of_graph[ne[jjj]]]
            NeighborNeighbor_of_node.append(nne)
            # print("nne", nne)
        NeighborNeighbor_of_node = sum(NeighborNeighbor_of_node, [])
        NeighborNeighbor[results[l]] = NeighborNeighbor_of_node
        Degree_for_node = Com_of_graph.degree(results[l])
        Degree_of_node[results[l]] = Degree_for_node
        # print("NeighborNeighbor_of_node", NeighborNeighbor_of_node)
        ha= 0
        while running:
            if ha == len(ne):
                break
            Neigbhor_degree = Com_of_graph.degree(ne[ha])
            mg.append(Neigbhor_degree)
            # print(Neigbhor_degree)
            ha = ha + 1
        sigma_Neigbhor_degree = sum(mg)
        # print("sigma_Neigbhor_degree node", results[l], "--->", sigma_Neigbhor_degree)
        g = 0
        while running:
            if g == len(NeighborNeighbor_of_node):
                break
            NeigbhorNeigbhor_degree = Com_of_graph.degree(NeighborNeighbor_of_node[g])
            mmg.append(NeigbhorNeigbhor_degree)
            g = g + 1
        sigma_NeigbhorNeigbhor_degree = sum(mmg)
        sigma_NeigbhorNeigbhor_degree_dic[results[l]] = sigma_NeigbhorNeigbhor_degree
        fb = []
        for nl in range(len(ne)):
            dd = Com_of_graph.degree(ne[nl])
            fb.append(dd)
        DN = sum(fb)
        DN_dic[results[l]] = DN
    # print("------------------------------------------")
    i = i+1

sigma_HD = sum(h)

no_ou = []  # list of nodes that they relations with other communities.
node_hub = []
for key in node_out:
    no_ou.append(key)
for i in gh_dic:
    gh_dic[i] = float(gh_dic[i]*(1/sigma_HD))
for i in gh_dic_duplicatenode:
    gh_dic_duplicatenode[i] = float(gh_dic_duplicatenode[i]*(1/sigma_HD))
for key, value in gh_dic.items():
    if key in gh_dic_duplicatenode:
        P_do[key] = gh_dic[key] + gh_dic_duplicatenode[key]
    else:
        P_do[key] = value

for i in range(len(co)):
    for j in range(len(co)):
        if i == j:
            pass
        else:
            o = list(set(co[i]).intersection(co[j]))
            if o in node_hub:
                pass
            else:
                node_hub.append(o)
            # print("community", i, "and ", j, "has overlapping ", o)
node_hub = sum(node_hub, [])
node_hub = list(set(node_hub))
nc = {}
n_o = {}
for ku in range(len(node_hub)):
    cc = 0
    # print("degree_node_hub", node_hub[ku], G.degree(node_hub[ku]))
    neighbor_of_nodehub = [n for n in G[node_hub[ku]]]
    nb = 0
    for hj in range(len(neighbor_of_nodehub)):
        if nx.degree(G, node_hub[ku]) > nx.degree(G, neighbor_of_nodehub[hj]):
            nb = nb + 1
        else:
            pass
    # print("nb_node_hub", node_hub[ku], nb)
    for th in range(len(co)):
        if node_hub[ku] in co[th]:
            cc = cc + 1
    nc[node_hub[ku]] = cc
    n_o[node_hub[ku]] = cc - nb
sorted_n_o = sorted(n_o.items(), key=operator.itemgetter(1))
sorted_n_o = dict(sorted_n_o)
# print("sorted_n_o", sorted_n_o)
initial_seed = []
xf = list(sorted_n_o.keys())
# print("xf", xf)
for qr in range(Z):
    initial_seed.append(xf[qr])


n1 = Counter(n1)
n2 = Counter(n2)
ans = dsum(n1, n2)

for key, value in ans.items():
    p_ar = random.uniform(0, 1)
    P_dI[key] = 1/math.exp(p_ar * value)
max_P_dI = max(P_dI.values())
min_P_dI = min(P_dI.values())
#  P_dI is normalized.
for key, value in ans.items():
    normalize_P_dI[key] = (value - min_P_dI) / (max_P_dI - min_P_dI)

#
# with open('n1.txt', 'w') as the_file:
#     for key, value in n1.items():
#         the_file.write(str(key)+"   "+str(value)+'\n')
# with open('n2.txt', 'w') as the_file:
#     for key, value in n2.items():
#         the_file.write(str(key)+"   "+str(value)+'\n')
# with open('answer.txt', 'w') as the_file:
#     for key, value in ans.items():
#         the_file.write(str(key)+"   "+str(value)+'\n')
# with open('pdi.txt', 'w') as the_file:
#     for key, value in P_dI.items():
#         the_file.write(str(key)+"   "+str(value)+'\n')
# with open('npdi.txt', 'w') as the_file:
#     for key, value in normalize_P_dI.items():
#         the_file.write(str(key)+"   "+str(value)+'\n')
# with open('pdo.txt', 'w') as the_file:
#     for key, value in P_do.items():
#         the_file.write(str(key)+"   "+str(value)+'\n')

P_dI = Counter(P_dI)
P_do = Counter(P_do)
P_dc = dsum(P_dI, P_do)
# with open('pdc.txt', 'w') as the_file:
#     for key, value in P_dc.items():
#         the_file.write(str(key)+"   "+str(value)+'\n')
# with open('DN.txt', 'w') as the_file:
#     for key, value in DN_dic.items():
#         the_file.write(str(key)+"   "+str(value)+'\n')

for key, value in node_out.items():
    mv = value
    vb = []
    for kl in range(len(mv)):
        for jh in range(len(co)):
            if mv[kl] in co[jh]:
                if jh in vb:
                    pass
                else:
                    vb.append(jh)
            Index_community_relation_with_node[key] = vb

#  calculate b_info.
for bn in range(len(co)):
    comu = co[bn]
    b_inf_com = {}
    if len(comu) > 200:
        for df in range(len(comu)):
            if comu[df] in P_dc:
                pass
            P_dc_node = P_dc[comu[df]]
            D_node = D[comu[df]]
            DDN_node = sigma_NeigbhorNeigbhor_degree_dic[comu[df]]
            DN_node = DN_dic[comu[df]]
            if comu[df] in Index_community_relation_with_node:
                tt = Index_community_relation_with_node[comu[df]]
                we = []
                for bt in range(len(tt)):
                    if math.isnan(w_c_dic[tt[bt]]):
                        w_c_dic[tt[bt]] = 0
                    else:
                        pass
                    we.append(w_c_dic[tt[bt]])
                sigma_wc = sum(we)
            else:
                sigma_wc = 0
            b_inf = (DDN_node)+(DN_node)*math.exp(P_dc_node+sigma_wc)
            # print("DDN_node", DDN_node)
            # print("DN_Node", DN_node)
            # print("parantez", P_dc_node+sigma_wc)
            # print("b_inf", b_inf)
            b_inf_com[comu[df]] = b_inf
        sorted_b_inf_com = sorted(b_inf_com.items(), key=operator.itemgetter(1))
        sorted_b_inf_com = dict(sorted_b_inf_com)
        # print("sorted_b_inf_com", sorted_b_inf_com)
        ds = list(sorted_b_inf_com.keys())
        km = len(ds)-1
        for vc in range(T):
            initial_seed.append(ds[km])
            km = km - 1
Influence_dic = {}
initial_influence = {}
for rp in range(len(initial_seed)):
    S = []
    S.append(initial_seed[rp])
    Influence = avgSize(G, S, 0.01, 1000)
    # print("Influence for ", initial_seed[rp], "is: ", Influence)
    Influence_dic[initial_seed[rp]] = Influence
sorted_Influence_dic = dict(sorted(Influence_dic.items(), key=operator.itemgetter(1)))
# naz = list(sorted_Influence_dic.keys())
# ham = len(naz)
# hamid = []
# for po in range(k):
#     hamid.append(naz[ham-1])
#     Influence = avgSize(G, hamid, 0.01, 1000)
#     print(naz[ham-1], Influence, len(hamid))
#     ham = ham - 1
seed = []
max_Influence = max(Influence_dic.items(), key=operator.itemgetter(1))[0]
print(max_Influence, "for k=1:", Influence_dic[max_Influence])
seed.append(max_Influence)
initial_seed.remove(max_Influence)
for ws in range(k-1):
    S = deepcopy(seed)
    for kp in range(len(initial_seed)):
        S = deepcopy(seed)
        S.append(initial_seed[kp])
        Influence = avgSize(G, S, 0.01, 1000)
        Influence_dic.update({initial_seed[kp]: Influence})
    max_Influence = max(Influence_dic.items(), key=operator.itemgetter(1))[0]
    seed.append(max_Influence)
    print(max_Influence, "for k=", len(seed), Influence_dic[max_Influence])
    initial_seed.remove(max_Influence)
time2 = dt.datetime.now()
totaltime = time2 - time1
print("Run Time:", int(totaltime.total_seconds()*1000))