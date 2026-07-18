import networkx as nx
import numpy as np
import random
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# ====================== Load Graph ======================
def load_graph(datasetname):
    G = nx.Graph()
    print('Start loading dataset!')
    with open('./datasets/' + datasetname + '.txt', 'r') as file:
        for i, line in enumerate(file):
            neighbors = line.strip().split()
            for neighbor in neighbors:
                G.add_edge(i, int(neighbor))
    print('Dataset loading was completed.')
    return G

# ====================== Stage 1: Select potential nodes (Pt) ======================
def select_potential_nodes(G, fraction_threshold=0.8):
    nodes = np.array(list(G.nodes()))
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    degrees = np.array([G.degree(v) for v in nodes])
    E = G.number_of_edges()
    threshold_deg = (E / n) ** 2
    mask1 = degrees > threshold_deg

    neighbor_deg_sum = np.zeros(n)
    neighbor_count = np.zeros(n, dtype=int)
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        neighbor_deg_sum[i] += degrees[j]
        neighbor_deg_sum[j] += degrees[i]
        neighbor_count[i] += 1
        neighbor_count[j] += 1
    neighbor_count = np.clip(neighbor_count, 1, None)
    avg_neighbor_deg = neighbor_deg_sum / neighbor_count
    mask2 = degrees >= avg_neighbor_deg - 1e-8

    try:
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc).copy()
        nodes_cc = np.array(list(G_cc.nodes()))
        n_cc = len(nodes_cc)
        idx_map = {node: i for i, node in enumerate(nodes_cc)}
        degrees_cc = np.array([G_cc.degree(v) for v in nodes_cc])
        A_cc = nx.to_scipy_sparse_array(G_cc, nodelist=nodes_cc, format='csr')
        d_sqrt_inv = np.sqrt(degrees_cc + 1e-12)
        D_inv_sqrt = diags(1.0 / d_sqrt_inv)
        L_norm = diags(np.ones(n_cc)) - D_inv_sqrt @ A_cc @ D_inv_sqrt
        eigenvalues, eigenvectors = eigsh(L_norm, k=2, which='SM', tol=1e-3, maxiter=500)
        fiedler = eigenvectors[:, 1]
        fiedler_abs = np.abs(fiedler)
        structural_importance_cc = degrees_cc * fiedler_abs
        threshold = np.percentile(structural_importance_cc, fraction_threshold * 100)
        mask3 = np.zeros(n, dtype=bool)
        for node in largest_cc:
            if node in idx_map:
                i_cc = idx_map[node]
                i_global = node_to_idx[node]
                mask3[i_global] = structural_importance_cc[i_cc] >= threshold
    except Exception as e:
        print(f"Warning: Spectral failed ({e}), using degree fallback")
        mask3 = degrees >= np.percentile(degrees, 80)

    final_mask = mask1 & mask2 & mask3
    Pt = nodes[final_mask].tolist()
    print(f"Pt selected: {len(Pt)} nodes out of {n} ({len(Pt)/n:.4%})")
    return Pt

def create_subgraph(G, Pt):
    return G.subgraph(Pt).copy()

# ====================== Stage 2: Ranking ======================
def motif_aware_discounting(G_prime, k, alpha=0.7, beta=0.3):
    scores = {v: G_prime.degree(v) for v in G_prime.nodes()}
    selected = []
    for _ in range(k):
        if not scores:
            break
        vi = max(scores, key=scores.get)
        selected.append(vi)
        del scores[vi]
        for vj in list(G_prime.neighbors(vi)):
            if vj not in scores: continue
            common = set(G_prime.neighbors(vi)) & set(G_prime.neighbors(vj))
            num_closed_triads = len(common)
            potential_wedges = G_prime.degree(vi) * (G_prime.degree(vj) - 1)
            num_open_wedges = max(0, potential_wedges - 2 * num_closed_triads)
            discount = alpha * num_closed_triads + beta * num_open_wedges
            scores[vj] -= discount
            if scores[vj] <= 0: del scores[vj]
    return selected

def vnrs_influence_estimation(G, Pt, p=0.02, num_samples=600):
    rr_sets = []
    def generate_rr_set():
        root = random.choice(list(G.nodes()))
        rr_set = set()
        queue = [root]
        while queue:
            cur = queue.pop(0)
            if cur in rr_set: continue
            rr_set.add(cur)
            for nei in G.neighbors(cur):
                if random.random() < p: queue.append(nei)
        return rr_set
    rr_sets = [generate_rr_set() for _ in range(num_samples)]
    Sv, Vv = {}, {}
    for v in Pt:
        coverage = [1 if v in rr else 0 for rr in rr_sets]
        mean_cov = np.mean(coverage)
        Sv[v] = mean_cov * G.number_of_nodes()
        var_cov = np.var(coverage)
        normalized_var = var_cov / (mean_cov**1.3 + 1e-6)
        Vv[v] = normalized_var
    return Sv, Vv

def hyperbolic_diffusion_depth(G, v, p=0.01, num_sim=500):
    depths = []
    for _ in range(num_sim):
        activated = set([v])
        queue = [(v, 0)]
        dists = []
        while queue:
            cur, depth = queue.pop(0)
            dists.append(depth)
            for nei in G.neighbors(cur):
                if nei not in activated and random.random() < p:
                    activated.add(nei)
                    queue.append((nei, depth+1))
        avg_depth = np.mean(dists) if dists else 0
        depths.append(avg_depth)
    DH = np.mean(depths)
    tanh_DH=np.tanh(DH - 2.5) + 1  # 2.5 تقریبی از Dmedian در اکثر شبکه‌ها
    return tanh_DH

def compute_sdiff(Sv, Vv, HDD):
    Sdiff = {}
    for v in set(Sv.keys()) & set(HDD.keys()):
        NV=Sv[v] * np.exp(-Vv[v])
        Sdiff[v] = (NV * HDD[v])+(np.exp(HDD[v])-1)  # فرمول اصلاح‌شده (واریانس جریمه می‌شود)
    return Sdiff

    # for v in set(Sv.keys()) & set(HDD.keys()):
    #     NV = Sv[v] * Vv[v] #np.exp(-Vv[v])
    #     Sdiff[v] = (NV * HDD[v]) + (np.exp(HDD[v])-1)
    #     print(f"Sv{v}= {Sv[v]}, and   Vv{v}={ Vv[v]},  and HDD{v} = {HDD[v]}, NV={ Sv[v] * Vv[v] }, and  Sdift1{v} = {(NV * HDD[v])}, and  Sdiff{v} = {Sdiff[v]}")
    # return Sdiff

def degree_based_influence(G, G_prime, Pt, k, alpha=0.7, beta=0.3):
    Pt_from_discount = motif_aware_discounting(G_prime, k, alpha, beta)
    sensitive_nodes = set(Pt_from_discount)
    scores = {}
    for v in G_prime.nodes():
        deg_v = G_prime.degree(v)
        sum_neighbor_deg = sum(G.degree(u) for u in G.neighbors(v))
        scores[v] = deg_v**2 + G.degree(v) + np.sqrt(sum_neighbor_deg)
    Pd = []
    selected_set = set()
    for _ in range(2*k):
        if not scores: break
        vi = max(scores, key=scores.get)
        Pd.append(vi)
        selected_set.add(vi)
        del scores[vi]
        for vj in list(G_prime.neighbors(vi)):
            if vj not in scores: continue
            common_neighbors = set(G_prime.neighbors(vi)) & set(G_prime.neighbors(vj))
            closed_triads = len(common_neighbors)
            potential_wedges = G_prime.degree(vi)*(G_prime.degree(vj)-1)
            open_wedges = max(0, potential_wedges - 2*closed_triads)
            penalty = alpha*closed_triads + beta*open_wedges
            scores[vj] -= penalty
            if scores[vj] <= 0: del scores[vj]
    return Pd

def stage_two(G, G_prime, Pt, k, p=0.02):
    Pt_from_discount = motif_aware_discounting(G_prime, k)
    Sv, Vv = vnrs_influence_estimation(G, Pt, p)
    HDD = {v: hyperbolic_diffusion_depth(G, v, p) for v in Pt}
    Sdiff = compute_sdiff(Sv, Vv, HDD)
    Ps = sorted(Sdiff, key=Sdiff.get, reverse=True)[:2*k]

    return Ps, Sdiff

# ====================== Stage 3: Final Seed Selection ======================
def quantum_inspired_walk_credit_filtered_egonet(G_prime, candidate_nodes, Sdiff,
                                                 w_q=0.5, w_s=0.5, eig_tol=1e-8,
                                                 min_egonet=3):
    qwc_scores = {}
    for v in candidate_nodes:
        neighbors = set(G_prime.neighbors(v))
        egonet_nodes = {v} | neighbors
        if len(egonet_nodes) < min_egonet:
            two_hop = set()
            for u in neighbors: two_hop.update(G_prime.neighbors(u))
            egonet_nodes |= two_hop
        egonet = G_prime.subgraph(egonet_nodes).copy()
        node_list = list(egonet.nodes())
        if not node_list: qwc_scores[v] = 0.0; continue
        idx_map_local = {node_list[i]: i for i in range(len(node_list))}
        A = nx.to_numpy_array(egonet, nodelist=node_list, dtype=float)
        degs_safe = np.where(A.sum(axis=1) > 0, A.sum(axis=1), 1.0)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degs_safe))
        H_mat = D_inv_sqrt @ A @ D_inv_sqrt
        try: eigvals, eigvecs = np.linalg.eigh(H_mat)
        except: eigvals, eigvecs = np.linalg.eig(H_mat + np.eye(H_mat.shape[0])*eig_tol)
        deg_array = np.array([egonet.degree(n) for n in node_list], dtype=float)
        psi0 = np.sqrt(deg_array)
        psi0 = psi0 / np.linalg.norm(psi0) if np.linalg.norm(psi0) > 0 else np.ones_like(psi0)
        coeffs = eigvecs.T.conj() @ psi0
        coeffs_sq = np.abs(coeffs)**2
        phi_sq = np.abs(eigvecs)**2
        P_avg_vec = phi_sq @ coeffs_sq
        p_center = float(P_avg_vec[idx_map_local[v]]) if v in idx_map_local else 0.0
        classical_center = float(deg_array[idx_map_local[v]] / deg_array.sum()) if deg_array.sum()>0 else 1.0/len(node_list)
        q_local = 0.5*p_center + 0.5*classical_center
        s_term = Sdiff[v]/(1+Sdiff[v]) if Sdiff else 0.0
        qwc_scores[v] = w_q*q_local + w_s*s_term
    total = sum(qwc_scores.values())
    if total > 0:
        for k in qwc_scores: qwc_scores[k] /= total
    return qwc_scores

def stage_three(Pt, Ps, Pd, k, G_prime, Sdiff, diversity=True):
    seeds, seen = [], set()
    Pt_set, Ps_set, Pd_set = set(Pt), set(Ps), set(Pd)
    intersection = list(Ps_set & Pd_set)
    intersection.sort(key=lambda v: Ps.index(v) if v in Ps else 1000)
    for v in intersection:
        if len(seeds) >= k: break
        if v in Pt_set and v not in seen:
            seeds.append(v)
            seen.add(v)
    if len(seeds) < k:
        candidates = (Ps_set | Pd_set) - seen
        candidates = [v for v in candidates if v in Pt_set]
        if candidates:
            qwc_scores = quantum_inspired_walk_credit_filtered_egonet(G_prime, candidates, Sdiff)
            ranked_candidates = sorted(candidates, key=lambda v: qwc_scores[v], reverse=True)
            for v in ranked_candidates:
                if len(seeds) >= k: break
                if diversity and any([v in G_prime.neighbors(s) for s in seeds]):
                    continue
                seeds.append(v)
                seen.add(v)
    return seeds #[:k]


# ====================== Main MyMethod ======================
def Myinfluence_maximization(G, k, p=0.01):
    print("Stage 1: Selecting potential nodes...")
    Pt = select_potential_nodes(G)
    print(f"|Pt| = {len(Pt)}")
    G_prime = create_subgraph(G, Pt)
    print("Stage 2: Ranking started...")
    Pd = degree_based_influence(G, G_prime, Pt, k)
    print("Stage 2: Pd computed")
    Ps, Sdiff = stage_two(G, G_prime, Pt, k, p)
    print("Stage 2: Ps computed")
    seeds = stage_three(Pt, Ps, Pd, k, G_prime, Sdiff)
    print("Stage 3: Final seeds:", seeds)
    return seeds