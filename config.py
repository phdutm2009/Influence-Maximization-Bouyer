"""
Global configuration file Used across the entire Influence Maximization framework
"""

import os


# ===============================
# Dataset Configuration
# ===============================
DATASET_NAME = "brightkite.txt"  #brightkite.txt  protein
DATASET_FOLDER = "datasets/edgelist"
FILE_FORMAT = "edgelist"        # "adjlist" or "edgelist"

# DATASET_NAME = "ca_grqc.txt"   # dolphins email_enron, netscience, Citeseer,pgp,ca_hepth, ca_grqc, condmat_2005,dblp cora
# DATASET_FOLDER = "datasets/raw"
# FILE_FORMAT = "adjlist"        # "adjlist" or "edgelist"
DIRECTED = False
WEIGHTED = False


# ===============================
# Graph Preprocessing Options
# ===============================

REMOVE_SELF_LOOPS = True
REMOVE_ISOLATED_NODES = True
#USE_LARGEST_COMPONENT = True
RELABEL_TO_INTEGERS = True


# ===============================
# Experiment Settings
# ===============================

K_VALUES = [1, 5, 10, 20, 30]
PROPAGATION_PROBS = [ 0.01, 0.02, 0.05]
MC_SIMULATIONS = 2000
# RANDOM_SEED = 42


# ===============================
# Output Paths
# ===============================

RESULT_FOLDER = "results"
TABLE_FOLDER = os.path.join(RESULT_FOLDER, "tables")
FIGURE_FOLDER = os.path.join(RESULT_FOLDER, "figures")
LOG_FOLDER = os.path.join(RESULT_FOLDER, "logs")