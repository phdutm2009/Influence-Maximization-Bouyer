import networkx as nx
import datetime as rt
import operator
import copy
import math
import numpy as np
from networkx.algorithms.connectivity.edge_kcomponents import bridge_components
import statistics
from statistics import mode


def runIC(G_base, S, p):
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
                if random.uniform(0, 1) < 1 - (1 - p) ** w:  # if at least one of edges propagate influence
                    # print (T[i], 'influences', v)
                    T.append(v)
        i += 1
    return T


def avgSize(G_base, S, p, iterations):
    avg = 0
    for i in range(iterations):
        activenodes = runIC(G_base, S, p)
        c_a = []
        for j in range(len(activenodes)):
            c_a.append(core_num[activenodes[j]])
        mylist = list(dict.fromkeys(c_a))
        dict_count_nodeactive_shell = {}
        for l in range(len(mylist)):
            count = c_a.count(mylist[l])
            dict_count_nodeactive_shell[mylist[l]] = count
        list_key = list(dict_count_nodeactive_shell.keys())
        list_values = list(dict_count_nodeactive_shell.values())
        KS_nu_ac = []
        for w in range(len(dict_count_nodeactive_shell)):
            KS_nu_ac.append(list_key[w] * list_values[w])
        sigma = sum(KS_nu_ac)
        avg += float(sigma) / iterations
    return avg


print("start run:")
time1 = rt.datetime.now()
p = 0.01
iteration = 1000
k = 30
Seed = [867, 396, 1878, 24904, 5989, 24069, 17963, 21798, 23143, 24061, 11080, 16276, 21438, 5691, 4581, 7205, 406,
        24062, 1925, 24122, 25329, 2518, 8924, 10935, 20162, 1, 9252, 27867, 5832, 18669]
# read graph with multi edge.
file = "/content/drive/MyDrive/Colab Notebooks/Route views.txt"
print("Data set:", file)
G = nx.Graph()
with open(file) as f:
    for line in f:
        u, v = map(int, line.split())
        try:
            G[u][v]['weight'] += 1
        except:
            G.add_edge(u, v, weight=1)
print(nx.number_of_nodes(G))
G_base = nx.MultiGraph()
with open(file) as f:
    for line in f:
        u, v = map(int, line.split())
        try:
            G_base[u][v]['weight'] += 1
        except:
            G_base.add_edge(u, v, weight=1)

Degreeof_nodes = nx.degree(G_base)
Degreeof_nodes = dict(Degreeof_nodes)
sorted_Deg_dict = dict(sorted(Degreeof_nodes.items(), key=operator.itemgetter(1), reverse=True))
Sorted_node_Degree = list(sorted_Deg_dict.keys())
print("Sorted_node_Degree", Sorted_node_Degree)
module = []
module_dic = {}
node_sign = list(nx.nodes(G))
count = 0
for i in range(len(Sorted_node_Degree)):
    count = count + 1

    Node = Sorted_node_Degree[i]
    # if Node == 867:
    #   print("Node", Node)
    l = 0
    runnig = True
    while (runnig):
        if l == 2:
            break
        if len(module) == 0:
            # if Node == 867:
            #   print("K1")
            neighbors_of_Nodes = list(G.neighbors(Node))
            Cl_Node = nx.clustering(G, nodes=Node)
            dict_cl = {}
            neighbors_level1_to_module = []
            for j in range(len(neighbors_of_Nodes)):
                Cl = nx.clustering(G, nodes=neighbors_of_Nodes[j])
                if Cl_Node < Cl:
                    dict_cl[neighbors_of_Nodes[j]] = Cl
                    neighbors_level1_to_module.append(neighbors_of_Nodes[j])
            neighbors_level1_to_module_C = copy.deepcopy(neighbors_level1_to_module)
            for hh in range(len(neighbors_level1_to_module_C)):
                if neighbors_level1_to_module_C[hh] in node_sign:
                    node_sign.remove(neighbors_level1_to_module_C[hh])
                else:
                    neighbors_level1_to_module.remove(neighbors_level1_to_module_C[hh])
            module_dic[Node] = list(neighbors_level1_to_module)
            if len(dict_cl) == 0:
                module.append(list(Node))
                break
            else:
                l = l + 1  # level 1
                # print("l", l)
                TR = []
                neighbors_level2_all = []
                for q in range(len(module_dic[Node])):
                    RR = list(module_dic[Node])
                    # print(RR)
                    neighbors_of_RR = list(G.neighbors(RR[q]))
                    neighbors_level2_all.append(neighbors_of_RR)
                    Cl_neighbors_RRq = nx.clustering(G, RR[q])
                    if Node in neighbors_of_RR:
                        neighbors_of_RR.remove(Node)
                    neighbors_of_Nodes_c = copy.deepcopy(neighbors_of_Nodes)
                    for t in range(len(neighbors_of_Nodes_c)):
                        if neighbors_of_Nodes[t] in neighbors_of_RR:
                            neighbors_of_RR.remove(neighbors_of_Nodes[t])
                    neighbors_of_RR_C = copy.deepcopy(neighbors_of_RR)
                    for e in range(len(neighbors_of_RR_C)):
                        if nx.clustering(G, neighbors_of_RR_C[e]) < Cl_neighbors_RRq:
                            neighbors_of_RR.remove(neighbors_of_RR_C[e])
                    TR.append(neighbors_of_RR)
                # neighbors_of_Nodes.append([Node])
                # neighbors_level2_all = neighbors_level2_all + neighbors_of_Nodes
                neighbors_level2_all.append(neighbors_of_Nodes)
                neighbors_level2_all.append([Node])
                # print(neighbors_level2_all)
                Neigbor_all = sum(neighbors_level2_all, [])
                neighbors_level2_to_module = sum(TR, [])
                neighbors_level2_to_module_C = copy.deepcopy(neighbors_level2_to_module)
                for er in range(len(neighbors_level2_to_module_C)):
                    if neighbors_level2_to_module_C[er] in node_sign:
                        node_sign.remove(neighbors_level2_to_module_C[er])
                    else:
                        neighbors_level2_to_module.remove(neighbors_level2_to_module_C[er])
                SS = list(dict_cl.keys())
                if len(neighbors_level2_to_module) == 0:
                    neighbors_level1_to_module.append(Node)
                    neighbors_level1_to_module_C = copy.deepcopy(neighbors_level1_to_module)
                    for ll in range(len(neighbors_level1_to_module_C)):
                        if neighbors_level1_to_module_C[ll] in module_extend:
                            neighbors_level1_to_module.remove(neighbors_level1_to_module_C[ll])
                    module.append(neighbors_level1_to_module)
                    break
                neighbors_level2_to_module_n = neighbors_level2_to_module + neighbors_level1_to_module
                new = copy.deepcopy(neighbors_level2_to_module_n)
                module_dic[Node] = new
                if len(neighbors_level2_to_module) != 0:
                    l = l + 1  # level 2
                    # print("l", l)
                    PR = []
                    for y in range(len(neighbors_level2_to_module)):
                        neighbors_of_PR = list(G.neighbors(neighbors_level2_to_module[y]))
                        Cl_neighbors_PRq = nx.clustering(G, neighbors_level2_to_module[y])
                        for u in range(len(Neigbor_all)):
                            if Neigbor_all[u] in neighbors_of_PR:
                                neighbors_of_PR.remove(Neigbor_all[u])
                        neighbors_of_PR_C = copy.deepcopy(neighbors_of_PR)
                        for d in range(len(neighbors_of_PR_C)):
                            if nx.clustering(G, neighbors_of_PR_C[d]) < Cl_neighbors_PRq:
                                neighbors_of_PR.remove(neighbors_of_PR_C[d])
                        PR.append(neighbors_of_PR)
                    neighbors_level3_to_module = sum(PR, [])
                    neighbors_level3_to_module_C = copy.deepcopy(neighbors_level3_to_module)
                    for hj in range(len(neighbors_level3_to_module_C)):
                        if neighbors_level3_to_module_C[hj] in node_sign:
                            node_sign.remove(neighbors_level3_to_module_C[hj])
                        else:
                            neighbors_level3_to_module.remove(neighbors_level3_to_module_C[hj])
                    new3 = new + neighbors_level3_to_module
                    new3.append(Node)
                    new3_C = copy.deepcopy(new3)
                    # for ll in range(len(new3_C)):
                    #     if new3_C[ll] in module_extend:
                    #         new3.remove(new3_C[ll])
                    module_dic[Node] = new3
                    # print("before add new3", module)
                    module.append(new3)
                    # print("after add new3", new3 )

        else:
            # print("degree", G.degree(Node))
            # if Node == 867:
            #   print("k2")
            module_extend = []
            module_C = copy.deepcopy(module)
            for ff in range(len(module_C)):
                if Node in module_C[ff]:
                    module_extend = copy.deepcopy(module_C[ff])
                    module.remove(module_C[ff])
            if len(module_extend) != 0:
                neighbors_of_Nodes = list(G.neighbors(Node))
                neighbors_of_Nodes_C = copy.deepcopy(neighbors_of_Nodes)
                for hd in range(len(neighbors_of_Nodes_C)):
                    if neighbors_of_Nodes_C[hd] in node_sign:
                        node_sign.remove(neighbors_of_Nodes_C[hd])
                    else:
                        neighbors_of_Nodes.remove(neighbors_of_Nodes_C[hd])
                #########################################
                if Node == 867:
                    print("NE", neighbors_of_Nodes)
                if len(neighbors_of_Nodes) == 0:

                    if Node in module_extend:
                        pass
                    else:
                        module_extend.append(Node)
                    module.append(module_extend)
                    break
                else:
                    dict_cl = {}
                    neighbors_level1_to_module = []
                    for j in range(len(neighbors_of_Nodes)):
                        Cl = nx.clustering(G, nodes=neighbors_of_Nodes[j])
                        if Cl_Node < Cl:
                            dict_cl[neighbors_of_Nodes[j]] = Cl
                            neighbors_level1_to_module.append(neighbors_of_Nodes[j])
                    neighbors_level1_to_module_C = copy.deepcopy(neighbors_level1_to_module)
                    for hh in range(len(neighbors_level1_to_module_C)):
                        if neighbors_level1_to_module_C[hh] in node_sign:
                            node_sign.remove(neighbors_level1_to_module_C[hh])
                        else:
                            neighbors_level1_to_module.remove(neighbors_level1_to_module_C[hh])
                    module_dic[Node] = list(neighbors_level1_to_module)
                    # print("module_extend", module_extend)
                    if len(neighbors_level1_to_module) == 0:
                        if Node in module_extend:
                            pass
                        else:
                            module_extend.append(Node)
                        module.append(module_extend)
                        break
                    else:
                        l = l + 1  # level 1
                        TR = []
                        neighbors_level2_all = []
                        for q in range(len(module_dic[Node])):
                            RR = list(module_dic[Node])
                            # print(RR)
                            neighbors_of_RR = list(G.neighbors(RR[q]))
                            neighbors_level2_all.append(neighbors_of_RR)
                            Cl_neighbors_RRq = nx.clustering(G, RR[q])
                            if Node in neighbors_of_RR:
                                neighbors_of_RR.remove(Node)
                            neighbors_of_Nodes_c = copy.deepcopy(neighbors_of_Nodes)
                            for t in range(len(neighbors_of_Nodes_c)):
                                if neighbors_of_Nodes[t] in neighbors_of_RR:
                                    neighbors_of_RR.remove(neighbors_of_Nodes[t])
                            neighbors_of_RR_C = copy.deepcopy(neighbors_of_RR)
                            for e in range(len(neighbors_of_RR_C)):
                                if nx.clustering(G, neighbors_of_RR_C[e]) < Cl_neighbors_RRq:
                                    neighbors_of_RR.remove(neighbors_of_RR_C[e])
                            TR.append(neighbors_of_RR)
                        # neighbors_of_Nodes.append([Node])
                        # neighbors_level2_all = neighbors_level2_all + neighbors_of_Nodes
                        neighbors_level2_all.append(neighbors_of_Nodes)
                        neighbors_level2_all.append([Node])

                        Neigbor_all = sum(neighbors_level2_all, [])
                        neighbors_level2_to_module = sum(TR, [])
                        neighbors_level2_to_module_C = copy.deepcopy(neighbors_level2_to_module)
                        for er in range(len(neighbors_level2_to_module_C)):
                            if neighbors_level2_to_module_C[er] in node_sign:
                                node_sign.remove(neighbors_level2_to_module_C[er])
                            else:
                                neighbors_level2_to_module.remove(neighbors_level2_to_module_C[er])
                        SS = list(dict_cl.keys())
                        if len(neighbors_level2_to_module) == 0:
                            neighbors_level1_to_module.append(Node)
                            for ty in range(len(neighbors_level1_to_module)):
                                if neighbors_level1_to_module[ty] in module_extend:
                                    pass
                                else:
                                    module_extend.append(neighbors_level1_to_module[ty])
                            module.append(module_extend)
                            break
                        neighbors_level2_to_module_n = neighbors_level2_to_module + neighbors_level1_to_module
                        new = copy.deepcopy(neighbors_level2_to_module_n)
                        module_dic[Node] = new
                        if len(neighbors_level2_to_module) != 0:
                            l = l + 1  # level 2

                            PR = []
                            for y in range(len(neighbors_level2_to_module)):
                                neighbors_of_PR = list(G.neighbors(neighbors_level2_to_module[y]))
                                Cl_neighbors_PRq = nx.clustering(G, neighbors_level2_to_module[y])
                                for u in range(len(Neigbor_all)):
                                    if Neigbor_all[u] in neighbors_of_PR:
                                        neighbors_of_PR.remove(Neigbor_all[u])
                                neighbors_of_PR_C = copy.deepcopy(neighbors_of_PR)
                                for d in range(len(neighbors_of_PR_C)):
                                    if nx.clustering(G, neighbors_of_PR_C[d]) < Cl_neighbors_PRq:
                                        neighbors_of_PR.remove(neighbors_of_PR_C[d])
                                PR.append(neighbors_of_PR)
                            neighbors_level3_to_module = sum(PR, [])
                            neighbors_level3_to_module_C = copy.deepcopy(neighbors_level3_to_module)
                            for hj in rang(len(neighbors_level3_to_module_C)):
                                if neighbors_level3_to_module_C[hj] in node_sign:
                                    node_sign.remove(neighbors_level3_to_module_C[hj])
                                else:
                                    neighbors_level3_to_module.remove(neighbors_level3_to_module_C[hj])
                            new3 = new + neighbors_level3_to_module
                            module_dic[Node] = new3

                            new3.append(Node)
                            new3_C = copy.deepcopy(new3)
                            for ll in range(len(new3_C)):
                                if new3_C[ll] in module_extend:
                                    new3.remove(new3_C[ll])
                            for kl in range(len(new3)):
                                module_extend.append(new3[kl])
                            module.append(module_extend)

                            break
            else:
                neighbors_of_Nodes = list(G.neighbors(Node))
                Cl_Node = nx.clustering(G, nodes=Node)
                dict_cl = {}
                neighbors_level1_to_module = []
                for j in range(len(neighbors_of_Nodes)):
                    Cl = nx.clustering(G, nodes=neighbors_of_Nodes[j])
                    if Cl_Node < Cl:
                        dict_cl[neighbors_of_Nodes[j]] = Cl
                        neighbors_level1_to_module.append(neighbors_of_Nodes[j])
                neighbors_level1_to_module_C = copy.deepcopy(neighbors_level1_to_module)
                # if Node == 867:
                #   print("neighbor befor remove", neighbors_level1_to_module)
                for hh in range(len(neighbors_level1_to_module_C)):
                    if neighbors_level1_to_module_C[hh] in node_sign:
                        node_sign.remove(neighbors_level1_to_module_C[hh])
                    else:
                        neighbors_level1_to_module.remove(neighbors_level1_to_module_C[hh])
                        # if Node == 867:
                        #   print("+")

                module_dic[Node] = list(neighbors_level1_to_module)
                # if Node == 867:
                #   print("NOde leve 1 ", neighbors_level1_to_module)
                if len(neighbors_level1_to_module) == 0:
                    Ne_Module = []
                    for PTR in range(len(module)):
                        MO = module[PTR]
                        for po in range(len(neighbors_of_Nodes)):
                            if neighbors_of_Nodes[po] in MO:
                                Ne_Module.append(PTR)
                    counter = 0
                    # print("Ne_Module", Ne_Module)
                    if len(Ne_Module) != 0:

                        num = Ne_Module[0]

                        for ii in Ne_Module:
                            curr_frequency = Ne_Module.count(ii)
                            if (curr_frequency > counter):
                                counter = curr_frequency
                                num = ii
                        Module_max = num
                        NROP = module[Module_max]
                        module.remove(NROP)
                        NROP.append(Node)
                        # if Node == 867:
                        #   print("neighbors_level1_to_module is 0")
                        if Node in module_extend:
                            pass
                        else:
                            module_extend.append(Node)
                        module.append(NROP)
                        break
                    else:
                        if Node in module_extend:
                            pass
                        else:
                            module_extend.append(Node)
                        module.append(module_extend)
                        break

                else:
                    # if Node == 867:
                    #   print("K2")
                    l = l + 1  # level 1
                    # print("l", l)
                    TR = []
                    neighbors_level2_all = []
                    for q in range(len(module_dic[Node])):
                        RR = list(module_dic[Node])
                        # print(RR)
                        neighbors_of_RR = list(G.neighbors(RR[q]))
                        neighbors_level2_all.append(neighbors_of_RR)
                        Cl_neighbors_RRq = nx.clustering(G, RR[q])
                        if Node in neighbors_of_RR:
                            neighbors_of_RR.remove(Node)
                        neighbors_of_Nodes_c = copy.deepcopy(neighbors_of_Nodes)
                        for t in range(len(neighbors_of_Nodes_c)):
                            if neighbors_of_Nodes[t] in neighbors_of_RR:
                                neighbors_of_RR.remove(neighbors_of_Nodes[t])
                        neighbors_of_RR_C = copy.deepcopy(neighbors_of_RR)
                        for e in range(len(neighbors_of_RR_C)):
                            if nx.clustering(G, neighbors_of_RR_C[e]) < Cl_neighbors_RRq:
                                neighbors_of_RR.remove(neighbors_of_RR_C[e])
                        TR.append(neighbors_of_RR)
                    # neighbors_of_Nodes.append([Node])
                    # neighbors_level2_all = neighbors_level2_all + neighbors_of_Nodes
                    neighbors_level2_all.append(neighbors_of_Nodes)
                    neighbors_level2_all.append([Node])
                    # print(neighbors_level2_all)
                    Neigbor_all = sum(neighbors_level2_all, [])
                    neighbors_level2_to_module = sum(TR, [])
                    neighbors_level2_to_module_C = copy.deepcopy(neighbors_level2_to_module)
                    for er in range(len(neighbors_level2_to_module_C)):
                        if neighbors_level2_to_module_C[er] in node_sign:
                            node_sign.remove(neighbors_level2_to_module_C[er])
                        else:
                            neighbors_level2_to_module.remove(neighbors_level2_to_module_C[er])
                    SS = list(dict_cl.keys())
                    if len(neighbors_level2_to_module) == 0:
                        neighbors_level1_to_module.append(Node)
                        for ty in range(len(neighbors_level1_to_module)):
                            if neighbors_level1_to_module[ty] in module_extend:
                                pass
                            else:
                                module_extend.append(neighbors_level1_to_module[ty])
                        module.append(module_extend)
                        break
                    neighbors_level2_to_module_n = neighbors_level2_to_module + neighbors_level1_to_module
                    new = copy.deepcopy(neighbors_level2_to_module_n)
                    module_dic[Node] = new
                    if len(neighbors_level2_to_module) != 0:
                        l = l + 1  # level 2
                        # print("l", l)
                        PR = []
                        for y in range(len(neighbors_level2_to_module)):
                            neighbors_of_PR = list(G.neighbors(neighbors_level2_to_module[y]))
                            Cl_neighbors_PRq = nx.clustering(G, neighbors_level2_to_module[y])
                            for u in range(len(Neigbor_all)):
                                if Neigbor_all[u] in neighbors_of_PR:
                                    neighbors_of_PR.remove(Neigbor_all[u])
                            neighbors_of_PR_C = copy.deepcopy(neighbors_of_PR)
                            for d in range(len(neighbors_of_PR_C)):
                                if nx.clustering(G, neighbors_of_PR_C[d]) < Cl_neighbors_PRq:
                                    neighbors_of_PR.remove(neighbors_of_PR_C[d])
                            PR.append(neighbors_of_PR)
                        neighbors_level3_to_module = sum(PR, [])
                        neighbors_level3_to_module_C = copy.deepcopy(neighbors_level3_to_module)
                        for hj in range(len(neighbors_level3_to_module_C)):
                            if neighbors_level3_to_module_C[hj] in node_sign:
                                node_sign.remove(neighbors_level3_to_module_C[hj])
                            else:
                                neighbors_level3_to_module.remove(neighbors_level3_to_module_C[hj])
                        new3 = new + neighbors_level3_to_module
                        new3_C = copy.deepcopy(new3)
                        for ll in range(len(new3_C)):
                            if new3_C[ll] in module_extend:
                                new3.remove(new3_C[ll])
                        module_dic[Node] = new3

                        new3.append(Node)
                        for kl in range(len(new3)):
                            module_extend.append(new3[kl])
                        module.append(module_extend)

            #########################################
            # ???????????????????????????????????????
            # ???????????????????????????????????????
            # neighbor_in_module = []
            # for lp in range(len(neighbors_of_Nodes)):
            #     if neighbors_of_Nodes[lp] in module_extend:
            #         neighbor_in_module.append(neighbors_of_Nodes[lp])
            # print("neighbor_in_module", neighbor_in_module)
            # if len(neighbor_in_module) == 0:
            #     dict_cl = {}
            #     neighbors_level1_to_module = []
            #     for j in range(len(neighbors_of_Nodes)):
            #         Cl = nx.clustering(G, nodes=neighbors_of_Nodes[j])
            #         if Cl_Node < Cl:
            #             dict_cl[neighbors_of_Nodes[j]] = Cl
            #             neighbors_level1_to_module.append(neighbors_of_Nodes[j])
            #     neighbors_level1_to_module_C = copy.deepcopy(neighbors_level1_to_module)
            #     for hh in range(len(neighbors_level1_to_module_C)):
            #         if neighbors_level1_to_module_C[hh] in node_sign:
            #             node_sign.remove(neighbors_level1_to_module_C[hh])
            #         else:
            #             neighbors_level1_to_module.remove(neighbors_level1_to_module_C[hh])
            #     module_dic[Node] = list(neighbors_level1_to_module)
            #     print("neighbors_level1_to_module", neighbors_level1_to_module)
            #     # print("module_extend", module_extend)
            #     if len(neighbors_level1_to_module) == 0:
            #         module_extend.append(Node)
            #         module.append(module_extend)
            #         print("moudule after add module extend", module)
            #         break
            #     else:
            #         l = l + 1  # level 1
            #         print("l", l)
            #         TR = []
            #         neighbors_level2_all = []
            #         for q in range(len(module_dic[Node])):
            #             RR = list(module_dic[Node])
            #             # print(RR)
            #             neighbors_of_RR = list(G.neighbors(RR[q]))
            #             neighbors_level2_all.append(neighbors_of_RR)
            #             Cl_neighbors_RRq = nx.clustering(G, RR[q])
            #             if Node in neighbors_of_RR:
            #                 neighbors_of_RR.remove(Node)
            #             neighbors_of_Nodes_c = copy.deepcopy(neighbors_of_Nodes)
            #             for t in range(len(neighbors_of_Nodes_c)):
            #                 if neighbors_of_Nodes[t] in neighbors_of_RR:
            #                     neighbors_of_RR.remove(neighbors_of_Nodes[t])
            #             neighbors_of_RR_C = copy.deepcopy(neighbors_of_RR)
            #             for e in range(len(neighbors_of_RR_C)):
            #                 if nx.clustering(G, neighbors_of_RR_C[e]) < Cl_neighbors_RRq:
            #                     neighbors_of_RR.remove(neighbors_of_RR_C[e])
            #             TR.append(neighbors_of_RR)
            #         # neighbors_of_Nodes.append([Node])
            #         # neighbors_level2_all = neighbors_level2_all + neighbors_of_Nodes
            #         neighbors_level2_all.append(neighbors_of_Nodes)
            #         neighbors_level2_all.append([Node])
            #         print("neighbors_level2_all", neighbors_level2_all)
            #         Neigbor_all = sum(neighbors_level2_all, [])
            #         neighbors_level2_to_module = sum(TR, [])
            #         neighbors_level2_to_module_C = copy.deepcopy(neighbors_level2_to_module)
            #         for er in range(len(neighbors_level2_to_module_C)):
            #             if neighbors_level2_to_module_C[er] in node_sign:
            #                 node_sign.remove(neighbors_level2_to_module_C[er])
            #             else:
            #                 neighbors_level2_to_module.remove(neighbors_level2_to_module_C[er])
            #         SS = list(dict_cl.keys())
            #         if len(neighbors_level2_to_module) == 0:
            #             neighbors_level1_to_module.append(Node)
            #             for ty in range(len(neighbors_level1_to_module)):
            #                 module_extend.append(neighbors_level1_to_module[ty])
            #             module.append(module_extend)
            #             break
            #         neighbors_level2_to_module_n = neighbors_level2_to_module + neighbors_level1_to_module
            #         new = copy.deepcopy(neighbors_level2_to_module_n)
            #         module_dic[Node] = new
            #         if len(neighbors_level2_to_module) != 0:
            #             l = l + 1  # level 2
            #             print("l", l)
            #             PR = []
            #             for y in range(len(neighbors_level2_to_module)):
            #                 neighbors_of_PR = list(G.neighbors(neighbors_level2_to_module[y]))
            #                 Cl_neighbors_PRq = nx.clustering(G, neighbors_level2_to_module[y])
            #                 for u in range(len(Neigbor_all)):
            #                     if Neigbor_all[u] in neighbors_of_PR:
            #                         neighbors_of_PR.remove(Neigbor_all[u])
            #                 neighbors_of_PR_C = copy.deepcopy(neighbors_of_PR)
            #                 for d in range(len(neighbors_of_PR_C)):
            #                     if nx.clustering(G, neighbors_of_PR_C[d]) < Cl_neighbors_PRq:
            #                         neighbors_of_PR.remove(neighbors_of_PR_C[d])
            #                 PR.append(neighbors_of_PR)
            #             neighbors_level3_to_module = sum(PR, [])
            #             neighbors_level3_to_module_C = copy.deepcopy(neighbors_level3_to_module)
            #             for hj in rang(len(neighbors_level3_to_module_C)):
            #                 if neighbors_level3_to_module_C[hj] in node_sign:
            #                     node_sign.remove(neighbors_level3_to_module_C[hj])
            #                 else:
            #                     neighbors_level3_to_module.remove(neighbors_level3_to_module_C[hj])
            #             new3 = new + neighbors_level3_to_module
            #             module_dic[Node] = new3
            #             print("new3", new3)
            #             new3.append(Node)
            #             for kl in range(len(new3)):
            #                 module_extend.append(new3[kl])
            #             module.append(module_extend)
            #             print("moule extend3", module_extend)
            #             break
            # else:
            #     print("k3")
            #     print("modul_P", module_extend)
            #     neighbors_of_Nodes = list(G.neighbors(Node))
            #     Cl_Node = nx.clustering(G, nodes=Node)
            #     dict_cl = {}
            #     neighbors_level1_to_module = []
            #     for j in range(len(neighbors_of_Nodes)):
            #         Cl = nx.clustering(G, nodes=neighbors_of_Nodes[j])
            #         if Cl_Node < Cl:
            #             dict_cl[neighbors_of_Nodes[j]] = Cl
            #             neighbors_level1_to_module.append(neighbors_of_Nodes[j])
            #     neighbors_level1_to_module_C = copy.deepcopy(neighbors_level1_to_module)
            #     for hh in range(len(neighbors_level1_to_module_C)):
            #         if neighbors_level1_to_module_C[hh] in node_sign:
            #             node_sign.remove(neighbors_level1_to_module_C[hh])
            #         else:
            #             neighbors_level1_to_module.remove(neighbors_level1_to_module_C[hh])
            #     module_dic[Node] = list(neighbors_level1_to_module)
            #     print("neighbors_level1_to_module", neighbors_level1_to_module)
            #     if len(neighbors_level1_to_module) == 0:
            #         print("module_extend", module_extend)
            #         module_extend.append(Node)
            #         module.append(module_extend)
            #         print("new module", module)
            #         break
            #
            #     else:
            #         l = l + 1  # level 1
            #         # print("l", l)
            #         TR = []
            #         neighbors_level2_all = []
            #         for q in range(len(module_dic[Node])):
            #             RR = list(module_dic[Node])
            #             # print(RR)
            #             neighbors_of_RR = list(G.neighbors(RR[q]))
            #             neighbors_level2_all.append(neighbors_of_RR)
            #             Cl_neighbors_RRq = nx.clustering(G, RR[q])
            #             if Node in neighbors_of_RR:
            #                 neighbors_of_RR.remove(Node)
            #             neighbors_of_Nodes_c = copy.deepcopy(neighbors_of_Nodes)
            #             for t in range(len(neighbors_of_Nodes_c)):
            #                 if neighbors_of_Nodes[t] in neighbors_of_RR:
            #                     neighbors_of_RR.remove(neighbors_of_Nodes[t])
            #             neighbors_of_RR_C = copy.deepcopy(neighbors_of_RR)
            #             for e in range(len(neighbors_of_RR_C)):
            #                 if nx.clustering(G, neighbors_of_RR_C[e]) < Cl_neighbors_RRq:
            #                     neighbors_of_RR.remove(neighbors_of_RR_C[e])
            #             TR.append(neighbors_of_RR)
            #         # neighbors_of_Nodes.append([Node])
            #         # neighbors_level2_all = neighbors_level2_all + neighbors_of_Nodes
            #         neighbors_level2_all.append(neighbors_of_Nodes)
            #         neighbors_level2_all.append([Node])
            #         # print(neighbors_level2_all)
            #         Neigbor_all = sum(neighbors_level2_all, [])
            #         neighbors_level2_to_module = sum(TR, [])
            #         neighbors_level2_to_module_C = copy.deepcopy(neighbors_level2_to_module)
            #         for er in range(len(neighbors_level2_to_module_C)):
            #             if neighbors_level2_to_module_C[er] in node_sign:
            #                 node_sign.remove(neighbors_level2_to_module_C[er])
            #             else:
            #                 neighbors_level2_to_module.remove(neighbors_level2_to_module_C[er])
            #         SS = list(dict_cl.keys())
            #         if len(neighbors_level2_to_module) == 0:
            #             neighbors_level1_to_module.append(Node)
            #             for ty in range(len(neighbors_level1_to_module)):
            #                 module_extend.append(neighbors_level1_to_module[ty])
            #             module.append(module_extend)
            #             break
            #         neighbors_level2_to_module_n = neighbors_level2_to_module + neighbors_level1_to_module
            #         new = copy.deepcopy(neighbors_level2_to_module_n)
            #         module_dic[Node] = new
            #         if len(neighbors_level2_to_module) != 0:
            #             l = l + 1  # level 2
            #             # print("l", l)
            #             PR = []
            #             for y in range(len(neighbors_level2_to_module)):
            #                 neighbors_of_PR = list(G.neighbors(neighbors_level2_to_module[y]))
            #                 Cl_neighbors_PRq = nx.clustering(G, neighbors_level2_to_module[y])
            #                 for u in range(len(Neigbor_all)):
            #                     if Neigbor_all[u] in neighbors_of_PR:
            #                         neighbors_of_PR.remove(Neigbor_all[u])
            #                 neighbors_of_PR_C = copy.deepcopy(neighbors_of_PR)
            #                 for d in range(len(neighbors_of_PR_C)):
            #                     if nx.clustering(G, neighbors_of_PR_C[d]) < Cl_neighbors_PRq:
            #                         neighbors_of_PR.remove(neighbors_of_PR_C[d])
            #                 PR.append(neighbors_of_PR)
            #             neighbors_level3_to_module = sum(PR, [])
            #             neighbors_level3_to_module_C = copy.deepcopy(neighbors_level3_to_module)
            #             for hj in range(len(neighbors_level3_to_module_C)):
            #                 if neighbors_level3_to_module_C[hj] in node_sign:
            #                     node_sign.remove(neighbors_level3_to_module_C[hj])
            #                 else:
            #                     neighbors_level3_to_module.remove(neighbors_level3_to_module_C[hj])
            #             new3 = new + neighbors_level3_to_module
            #             module_dic[Node] = new3
            #             print("new3", new3)
            #             new3.append(Node)
            #             for kl in range(len(new3)):
            #                 module_extend.append(new3[kl])
            #             module.append(module_extend)

print(module)
module = sorted(module, key=len, reverse=True)
# neighbors_of_867 = list(G.neighbors(867))
# print("len neighbor 867", len(neighbors_of_867))
# for i in range(len(module)):
#   AAAA = module[i]
#   for j in range(len(neighbors_of_867)):
#     if neighbors_of_867[j] in AAAA:
#       print("neighorother----->", i)

print("module", module)
for i in range(len(module)):
    SDF = module[i]
    for j in range(len(SDF)):
        if SDF[j] in Seed:
            print("i=", i, "......", "..........", SDF[j], module[i])
Sum_Comunity = 0

for i in range(len(module)):
    if Sum_Comunity >= (5 * (nx.number_of_nodes(G_base))) / 100:
        print("Sum_Comunity", Sum_Comunity)
        print("5persent", (5 * (nx.number_of_nodes(G_base))) / 100)
        Useful_community = 1
        break
    Sum_Comunity = Sum_Comunity + len(module[i])
print("Useful_community", Useful_community)
G.remove_edges_from(nx.selfloop_edges(G))
core_num = nx.core_number(G)
pageRank = nx.pagerank(G, alpha=0.85)
Eig_vec = nx.eigenvector_centrality_numpy(G)
size_node_usfulcomunities = 0
for e in range(Useful_community):
    size_node_usfulcomunities = size_node_usfulcomunities + len(module[e])

seed = []
for j in range(Useful_community):
    limit_seed = math.ceil(k * (len(module[j]) / size_node_usfulcomunities)) + 1
    Ip_dict = {}
    for u in range(len(module[j])):
        candidate_com = module[j]
        # Colseness = nx.closeness_centrality(G, u= candidate_com[u])
        if candidate_com[u] in Seed:
            print("seed", candidate_com[u])
        Ip = (Degreeof_nodes[candidate_com[u]] ** 2) * (pageRank[candidate_com[u]] + Eig_vec[candidate_com[u]])
        Ip_dict[candidate_com[u]] = Ip
    sorted_Ip_dict = dict(sorted(Ip_dict.items(), key=operator.itemgetter(1), reverse=True))
    print("sorted_Ip_dict", sorted_Ip_dict)
    max_Inf = list(sorted_Ip_dict.keys())
    seed_community = max_Inf[0:(limit_seed - 1)]
    seed.append(seed_community)
seed = sum(seed, [])
count = 0
for i in range(len(seed)):
    if seed[i] in Seed:
        count = count + 1
print("seed in count", count)
time2 = rt.datetime.now()
totaltime = time2 - time1
print("Run time:", totaltime.total_seconds() * 1000)
S = []
for i in range(0, 30):
    D = seed[i]
    S.append(D)
    print("Influence is ", avgSize(G_base, S, 0.01, 1000), "for k=", i + 1, ".")
print("Run finished!!!")
print("Seed=", S)
