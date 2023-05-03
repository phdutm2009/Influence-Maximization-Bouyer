import datetime as dt
import networkx as nx
import operator
import math
from copy import deepcopy
import random
import xlsxwriter

def runIC (G_base, S, p=.01):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''

    T = deepcopy(S)  # copy already selected nodes

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G_base[T[i]]:  # for neighbors of a selected node
            if v not in T:  # if it wasn't selected yet
                w = G_base.number_of_edges(T[i], v)  # count the number of edges between two nodes

                if random.uniform(0, 1) < 1-(1-p)**w:  # if at least one of edges propagate influence
                    # print (T[i], 'influences', v)
                    T.append(v)
        i += 1
    return T


def avgSize(G_base, S, p, iterations):
    avg = 0
    for i in range(iterations):
        avg += float(len(runIC(G_base, S, p)))/iterations
    return avg


print("Start run:")
# iteration is variable for Monte-carlo method.
iteration = 1000
# Enter  a file's name.
file ="email.txt"
#  G is graph in core.
G_base = nx.MultiGraph()
with open(file) as f:
    for line in f:
        e0, e1 = map(int, line.split())
        try:
            G_base[e0][e1]["weight"] += 1
        except KeyError:
            G_base.add_edge(e0, e1)
print("number of nodes", G_base.number_of_nodes())
print("number of edges", G_base.size())
#######################################################


# Insert Dataset
# G_IC is Graph.
G_IC = nx.Graph()
with open(file) as f:
    for line in f:
        e0, e1 = map(int, line.split())
        try:
            G_IC[e0][e1]["weight"] += 1
        except KeyError:
            G_IC.add_edge(e0, e1)
#######################################################
#  G is graph in core.
G = nx.Graph()
with open(file) as f:
    for line in f:
        e0, e1 = map(int, line.split())
        try:
            G[e0][e1]["weight"] += 1
        except KeyError:
            G.add_edge(e0, e1)

#######################################################
time1 = dt.datetime.now()
# Print information Graph
print("Data set:", file)
print("Number Nodes: ", G.number_of_nodes())
print("Number Edges: ", G.size())
nd = list(nx.nodes(G_IC))
# #########################################################
# edge_clustring = {}
# all_edge_clustring = []
# count = 0
# for i in range(len(nd)):
#     edges = list(G_IC.edges(nd[i]))
#     a_c_list = []
#     for j in range(len(edges)):
#         a = list(edges[j])
#         ne0 = [n for n in G_IC[a[0]]]
#         ne1 = [n for n in G_IC[a[1]]]
#         common_neighbour = list(set(ne0).intersection(ne1))
#         e_c = len(common_neighbour)/(G_IC.degree(a[0])*G_IC.degree(a[1]))
#         a_c_list.append(e_c)
#     edge_clustring[nd[i]] = sum(a_c_list)
#     all_edge_clustring.append(sum(a_c_list))
# edge_clustring_sorted = dict(sorted(edge_clustring.items(), key=operator.itemgetter(1)))
# print("edge_clustring_sorted", edge_clustring_sorted)
#########################################################
#  Find shell, nodes in shell and graph G in core.
r_d = 1
len_shell = []
seed_in_othershells = []
seed_shell = {}
running = True
list_rd = []
depth_shell_dic = {}
list_seperate_node_in_shell =[]
bo = []
# sell_and_nodes is list of dict that key is no. shell and values are nodes in shell based on depth.
shell_and_nodes = []
while running:
    remove_nodes = []
    di = 0
    degree = dict(nx.degree(G))
    for k, v in degree.items():
        if v == r_d:
            di = di + 1
            remove_nodes.append(k)
    if len(remove_nodes)== 0:
        pass
    else:
        bo.append(remove_nodes)
    if len(remove_nodes) == 0:
        r = {}
        r[r_d] = bo
        shell_and_nodes.append(r)
        bo = []

    if len(list_rd) == 0:
        list_rd.append(r_d)
    else:
        if list_rd[(len(list_rd)-1)] == r_d:
            list_rd.append(r_d)
        else:
            depth_shell = len(list_rd)
            depth_shell_dic[r_d-1] = depth_shell
            list_rd = []
    len_shell.append(len(remove_nodes))
    if len_shell[(len(len_shell))-1] == 0 and len_shell[(len(len_shell))-2]==0 and len_shell[(len(len_shell))-3]==0:
        break
    G.remove_nodes_from(remove_nodes)
    n_n = G.number_of_nodes()
    if n_n == 0:
        break
    if len(remove_nodes) == 0:
        r_d = r_d + 1

##############################################
# Find candid node in core.candid_node_in_core is CC.

core_nodes = list(G.nodes())

comp = list(nx.connected_components(G))
len_comp = []
for cm in range(len(comp)):
    len_comp.append(len(comp[cm]))
largest_cc = list(max(nx.connected_components(G), key=len))
deg = dict(G.degree())
deg_sorted = dict(sorted(deg.items(), key=operator.itemgetter(1)))
averge_degree_in_Gprime = sum(list(deg_sorted.values()))/len(list(G.nodes()))
fb = list(deg_sorted.keys())
degree_max_in_core = deg_sorted[fb[len(fb)-1]]
connection_core_node_with_othershell_dic = {}
for tg in range(len(core_nodes)):
    connection_core_node_with_othershell = G_base.degree(core_nodes[tg])-deg[core_nodes[tg]]
    connection_core_node_with_othershell_dic[core_nodes[tg]] = connection_core_node_with_othershell
connection_core_node_with_othershell_dic_sorted = dict(sorted(connection_core_node_with_othershell_dic.items(), key=operator.itemgetter(1)))
depth_shell_dic_sorted = dict(sorted(depth_shell_dic.items(), key=operator.itemgetter(1)))

# calculate assortativity after remove node in core.
assortativity_coefficient_G = nx.degree_assortativity_coefficient(G)
change_assosetivity_in_Gprime = {}
for op in range(len(largest_cc)):
    edge_of_node_removed = list(G.edges(largest_cc[op]))
    G.remove_node(largest_cc[op])
    assortativity_coefficient_Gprime = nx.degree_assortativity_coefficient(G)
    change_assosetivity_in_Gprime[largest_cc[op]] = assortativity_coefficient_Gprime
    G.add_node(largest_cc[op])
    G.add_edges_from(edge_of_node_removed)
change_assosetivity_in_Gprime_sorted = dict(sorted(change_assosetivity_in_Gprime.items(), key=operator.itemgetter(1)))

# calculate PI(v).
Inf_node_core = {}
for rf in range(len(largest_cc)):
    wq = ((deg_sorted[largest_cc[rf]]*math.pow(deg_sorted[largest_cc[rf]], 2)) *connection_core_node_with_othershell_dic[largest_cc[rf]])/ (change_assosetivity_in_Gprime[largest_cc[rf]])
    Inf_node_core[largest_cc[rf]] = wq
Inf_node_core_sorted = dict(sorted(Inf_node_core.items(), key=operator.itemgetter(1)))

#  select Influential node in core based "limit_node_core" by f in manuscript and ROI.
limit_node_core = int((60+len(largest_cc))/math.log(30, 10))
candid_node_in_core = []
if limit_node_core > len(largest_cc):
    candid_node_in_core.append(largest_cc)
    candid_node_in_core = candid_node_in_core[0]
else:
    sorted_node_base_Inf_core = list(Inf_node_core_sorted.keys())
    if assortativity_coefficient_G < 0:
        candid_node_in_core = sorted_node_base_Inf_core[0:limit_node_core]
    else:
        for to in range(limit_node_core):
            h = sorted_node_base_Inf_core.pop()
            candid_node_in_core.append(h)

#####################################################
#  Create set PC. candid_node_in_shell is set PC.
print()
node_in_core = list(nx.nodes(G))
cl = dict(nx.clustering(G_IC))
average_depth = sum(depth_shell_dic.values())/len(depth_shell_dic)
limit = math.ceil((len(depth_shell_dic)/(average_depth))+1)
# print("number of shell:", len(depth_shell_dic))
shell_with_nodes = []
ww = []
gho = 0
for qb in range(len(depth_shell_dic)-3):
    t = shell_and_nodes[qb]
    # print("t",t)
    w = list(t.values())
    y = w[0]
    # print("y", y)
    ko = []
    ko.append(len(sum(y, [])))
    # print("ko", sum(y, []))
    ko.append(len(y))
    ww.append(ko)
    for cv in range(len(y)):
        eb = y[cv]
        HA = {}
        cl_s = {}
        R = {}
        for gh in range(len(eb)):
            cl_s[eb[gh]] = cl[eb[gh]]
        avarage_clustring_in_depth = sum(list(cl_s.values()))/len(list(cl_s.values()))
        remove_nodes_based_clustring = []
        if qb > limit:
            for gh in range(len(eb)):
                ne = [n for n in G_base[eb[gh]]]
                for am in range(len(candid_node_in_core)):
                    if candid_node_in_core[am] in ne:
                        if G_base.degree(candid_node_in_core[am])> averge_degree_in_Gprime:
                            pass
                    else:
                        if cl_s[eb[gh]] > avarage_clustring_in_depth:
                            remove_nodes_based_clustring.append(eb[gh])
                            gho = gho + 1
                        else:
                            pass
        G_IC.remove_nodes_from(remove_nodes_based_clustring)
        if qb <= limit:
            G_IC.remove_nodes_from(sum(y, []))
            # gho = gho + len(sum(y, []))

# ww_tuple = tuple(ww)
# print(ww_tuple)
# workbook = xlsxwriter.Workbook('Example.xlsx')
# By default worksheet names in the spreadsheet will be
# Sheet1, Sheet2 etc., but we can also specify a name.
# worksheet = workbook.add_worksheet("My sheet")

# Start from the first cell. Rows and
# # columns are zero indexed.
# row = 0
# col = 0

# Iterate over the data and write it out row by row.
# 
# for name, score, ghb in (ww_tuple):
#     worksheet.write(row, col, name)
#     worksheet.write(row, col + 1, score)
#     worksheet.write(row, col + 2, ghb)
#     row += 1
# workbook.close()

deg = dict(G.degree())
deg_sorted = dict(sorted(deg.items(), key=operator.itemgetter(1)))
nodes_in_shell = list(set(list(G_IC.nodes())) - set(node_in_core))
# print("node shell baghimandeh", len(nodes_in_shell))
# print("node removed in shell", print(remove_nodes_based_clustring))
# print("gho",gho)
Inf_node_shell = {}
for ae in range(len(nodes_in_shell)):
    ne = [n for n in G_base[nodes_in_shell[ae]]]
    q = []
    for rr in range(len(candid_node_in_core)):
        if candid_node_in_core[rr] in ne:
            q.append(Inf_node_core_sorted[candid_node_in_core[rr]])
    sigma_wieghte_nodecoere_connected_with_nodeshell = sum(q)
    Inf_node_shell[nodes_in_shell[ae]] = (cl[nodes_in_shell[ae]])
Inf_node_shell_sorted = dict(sorted(Inf_node_shell.items(), key=operator.itemgetter(1)))
wt = list(Inf_node_shell_sorted.keys())
limit_node_shell = len(Inf_node_shell_sorted)
sorted_node_base_Inf_shell = list(Inf_node_shell_sorted.keys())
# print("sorted_node_base_Inf_shell", sorted_node_base_Inf_shell)
candid_node_in_shell = []
if assortativity_coefficient_G < 0:
    candid_node_in_shell = sorted_node_base_Inf_shell[0:limit_node_shell]
else:
    for to in range(limit_node_shell):
        h = sorted_node_base_Inf_shell.pop()
        candid_node_in_shell.append(h)
###################################################################
#  select seed node in step 2 with IN(vi).

candid_node_in_Gbase = candid_node_in_shell + candid_node_in_core
print("candid_node_in_shell", len(candid_node_in_shell))
print("candid_node_in_core", len(candid_node_in_core))
edge_simlarity = {}
all_edge_similarity = []
count = 0
for ar in range(len(candid_node_in_Gbase)):
    edges = list(G_base.edges(candid_node_in_Gbase[ar]))
    a_c_list = []
    for j in range(len(edges)):
        a = list(edges[j])
        ne0 = [n for n in G_base[a[0]]]
        ne1 = [n for n in G_base[a[1]]]
        common_neighbour = list(set(ne0).intersection(ne1))
        union = len(ne0) + len(ne1)
        e_c = (len(common_neighbour)+2)/math.pow((union-len(common_neighbour)), 2)
        a_c_list.append(e_c)
    edge_simlarity[candid_node_in_Gbase[ar]] = sum(a_c_list)
    all_edge_similarity.append(sum(a_c_list))
edge_simlarity_sorted = dict(sorted(edge_simlarity.items(), key=operator.itemgetter(1)))
# print(len(candid_node_in_Gbase))

# list_of_node = list(edge_clustring_sorted.keys())
# list_of_values = list(edge_clustring_sorted.values())
IN_dic = {}
for al in range(len(candid_node_in_Gbase)):
    degree_candid_node = G_base.degree(candid_node_in_Gbase[al])
    ne = [n for n in G_base[candid_node_in_Gbase[al]]]  # ne is neighbour of u.
    NeighborNeighbor_of_node = []
    for jjj in range(len(ne)):
        nne = [n for n in G_base[ne[jjj]]]
        NeighborNeighbor_of_node.append(nne)
    NeighborNeighbor_of_node = sum(NeighborNeighbor_of_node, [])
    D = []
    for i in range(len(NeighborNeighbor_of_node)):
        D.append(G_base.degree(NeighborNeighbor_of_node[i]))
    Degree_two_level = sum(D)
    IN = (math.pow(degree_candid_node, 2))/(degree_max_in_core*(edge_simlarity[candid_node_in_Gbase[al]]+1))
    IN_dic[candid_node_in_Gbase[al]] = IN
IN_dic_sorted = dict(sorted(IN_dic.items(), key=operator.itemgetter(1)))
l = list(IN_dic_sorted.keys())
seed_node = []
n = [1, 5, 10, 15, 20, 25, 30]
Result = []
for ba in range(30):
    h = l.pop()
    seed_node.append(h)

    print("Influence is ", avgSize(G_base, seed_node, 0.01, 1000), "for k=", ba + 1, ".")
    if ba+1 in n:
        Result.append(avgSize(G,seed_node, 0.01, 1000))
print(seed_node)
print("SRFM", Result)
time2 = dt.datetime.now()
totaltime = time2 - time1
print("Run time:", totaltime.total_seconds()*1000)
