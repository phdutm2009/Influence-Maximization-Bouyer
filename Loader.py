
import os
import networkx as nx
import random
import numpy as np
from config import *


# ============================================================
# Reproducibility
# ============================================================

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Core Loader
# ============================================================

def load_graph(dataset_name=None, FILE_FORMAT="adjlist"):

    # set_random_seed(RANDOM_SEED)

    if dataset_name is None:
        dataset_name = DATASET_NAME

    path = os.path.join(DATASET_FOLDER, dataset_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    print("=" * 60)
    print("Loading dataset:", dataset_name)
    print("Path:", path)

    # --------------------------------------------------------
    # Graph Type
    # --------------------------------------------------------

    G = nx.DiGraph() if DIRECTED else nx.Graph()  #default DIRECTED=False in config.py
    # --------------------------------------------------------
    # File Format
    # --------------------------------------------------------

    if FILE_FORMAT == "adjlist":
        G = _load_adjlist(path, G)

    elif FILE_FORMAT == "edgelist":
        G = _load_edgelist(path, G)

    else:
        raise ValueError("Unsupported FILE_FORMAT in config")

    print("Initial Graph")
    _print_graph_stats(G)

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    if REMOVE_SELF_LOOPS:
        G.remove_edges_from(nx.selfloop_edges(G))

    if REMOVE_ISOLATED_NODES:
        G.remove_nodes_from(list(nx.isolates(G)))

    # if USE_LARGEST_COMPONENT:
    #     G = _largest_component(G)

    if RELABEL_TO_INTEGERS:
        G = nx.convert_node_labels_to_integers(G)

    print("After Preprocessing")
    _print_graph_stats(G)

    print("=" * 60)

    return G, dataset_name


# ============================================================
# Loaders
# ============================================================

def _load_adjlist(path, G):

    with open(path, "r") as file:
        for i, line in enumerate(file):
            neighbors = line.strip().split()

            for n in neighbors:
                if WEIGHTED:
                    node, weight = n.split(":")
                    G.add_edge(i, int(node), weight=float(weight))
                else:
                    G.add_edge(i, int(n))

    return G


def _load_edgelist(path, G):

    if WEIGHTED:
        G = nx.read_weighted_edgelist(path,
                                      nodetype=int,
                                      create_using=G)
    else:
        G = nx.read_edgelist(path,
                             nodetype=int,
                             create_using=G)

    return G


# ============================================================
# Utilities
# ============================================================

def _largest_component(G):

    if DIRECTED:
        largest_cc = max(nx.weakly_connected_components(G),
                         key=len)
        return G.subgraph(largest_cc).copy()
    else:
        largest_cc = max(nx.connected_components(G),
                         key=len)
        return G.subgraph(largest_cc).copy()


def _print_graph_stats(G):

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Density:", round(nx.density(G), 6))
    print("Avg Degree:",
          round(sum(dict(G.degree()).values()) /
                G.number_of_nodes(), 3))