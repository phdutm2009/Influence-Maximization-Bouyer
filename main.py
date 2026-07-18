
from utils.Loader import load_graph
import time
import pandas as pd
from config import *
from utils.exporter import save_results
from utils.logger import log_runtime
# from algorithms.Adel_p1_Final_V5 import Myinfluence_maximization
from evaluation.ic import run_ic
from algorithms.Adel_p1_Final_V5 import celf
from algorithms.Adel_p1_Final_V5 import lir_method
from algorithms.lmp import lmp
from algorithms.mcgm import mcgm
from algorithms.highdegree import high_degree
from algorithms.srfm import srfm
from algorithms.ti_sc import ti_sc
from algorithms.csp import csp
from algorithms.fip import fip_method
from algorithms.CHOP_IM_benin_jamil import chop_im
from algorithms.Proposed_method import Myinfluence_maximization
# from algorithms.degree import degree_seeds
# from algorithms.pagerank import pagerank_seeds
# from algorithms.celf import celf


import itertools
import numpy as np
import networkx as nx

def compute_max_distance(G):
    """
    Maximum finite shortest path in each connected component.
    Used as penalty distance for disconnected nodes.
    """
    max_d = 0

    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        lengths = dict(nx.all_pairs_shortest_path_length(sub))

        for u in lengths:
            local_max = max(lengths[u].values())
            if local_max > max_d:
                max_d = local_max

    return max_d
#=====================================================================================
def coverage_radius(G, seeds):

    if len(seeds) == 0:
        return np.inf

    # max_dist = compute_max_distance(G)
    max_dist =6

    shortest_paths = {
        s: nx.single_source_shortest_path_length(G, s)
        for s in seeds
    }

    total = 0

    for v in G.nodes():

        best = max_dist

        for s in seeds:

            if v in shortest_paths[s]:
                best = min(best, shortest_paths[s][v])

        total += best

    return total / G.number_of_nodes()
#===================================================================================================

#===========================================================================================

def run_experiment():

    G,dataset_name = load_graph(dataset_name=None, FILE_FORMAT="adjlist")
    print('dataset_name:', dataset_name)

    methods = {
       # "MyMethod": Myinfluence_maximization,
       #  "LIR": lir_method,
       #  "CELF": celf,
       #  "LMP": lmp,
        "MCGM": mcgm,
        "High_Degree": high_degree,
        "SRFM": srfm,
        # "TI-SC": ti_sc,
        "CSP": csp,
       # "FIP": fip_method,
     #    "CHOP-IM": chop_im

    }

    K_values = K_VALUES         # [1, 5, 10, 20, 30]
    P_values = PROPAGATION_PROBS

    method_results = []

    print("\n" + "=" * 60)
    print("Running influence maximization experiments")
    print("=" * 60)

    for name, method in methods.items():

        print(f"\nMethod: {name}")
        flag=False
        topk=[]
        start = time.time()

        # -------------------------
        # seed generation
        # -------------------------
        if name == "CELF":
            seeds = method(G, 50, p=PROPAGATION_PROBS[0])
        else:
            seeds = method(G, 50)

        runtime = log_runtime(start)


        # -------------------------
        # evaluation loop
        # -------------------------
        spread=0
        coverage=0
        variance=0
        for p in P_values:
            for k in K_values:
                Kseeds = seeds[:k]
                if flag==False and k == 5:
                    topk = Kseeds
                    print("Seeds (top k for ADP):", topk)
                    flag=True

                spread, coverage, variance = run_ic(G, Kseeds, p, MC_SIMULATIONS)
                print(name, 'p:', p, '  K=',k,  '   Spread:', spread)
                method_results.append({
                    "Method": name,
                    "k": k,
                    "p": p,
                    "Spread": spread,
                    "Coverage": coverage,
                    "Variance": variance,
                    "Runtime": runtime
                })

        # -------------------------
        # for ADP computing
        # -------------------------

        # print("\nTopk shortest-path check:")
        DR = coverage_radius(G, topk)
        print(f"method={name} with   DR={DR:.4f}")
        method_results.append({"DR": DR})
        # for i in range(len(top5)):
        #     for j in range(i + 1, len(top5)):
        #
        #         u, v = top5[i], top5[j]
        #
        #         if nx.has_path(G, u, v):
        #             d = nx.shortest_path_length(G, u, v)
        #         else:
        #             d = compute_max_distance(G)
        #
        #         print(f"{u} <-> {v} : {d}")
    df = pd.DataFrame(method_results)
    save_results(df, RESULT_FOLDER + "/"+dataset_name+"_comparison.xlsx")


    print("\nExperiment finished. Results saved.")


if __name__ == "__main__":
    run_experiment()

