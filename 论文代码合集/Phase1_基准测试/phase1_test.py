"""
Phase 1b & 1c: Cross-topology testing + baseline algorithms + plots.

For each model trained on slice i (0-4):
  - Tests on slices i+1 to i+10
  - Computes MLU using a fixed test traffic sample
  - Compares against 3 baselines (ECMP, SPF, Distance-Weighted)

Output: Result/leo/mlu_results.csv + Result/leo/mlu_plot.png
"""
import json
import math
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, lil_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOPO_DIR = Path("Data/leo/topologies")
TUNNELS_FILE = Path("Data/leo/tunnels_full.txt")
TEST_HIST = Path("Data/leo/test/0.hist")
RESULT_DIR = Path("Result/leo")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

NUM_NODES = 96
NUM_COMMODITIES = NUM_NODES * (NUM_NODES - 1)  # 9120
HIST_LEN = 12
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# 1. Load test traffic (12 consecutive TMs from a random start)
# ============================================================
print("Loading test traffic...")
with open(TEST_HIST) as f:
    test_lines = f.readlines()
print(f"  Test samples available: {len(test_lines)}")

# Pick random starting point
start = np.random.randint(0, len(test_lines) - HIST_LEN)
print(f"  Using lines {start} to {start + HIST_LEN - 1} as input")

# Parse 12 consecutive TMs as input, 12th as the one to route
tms_input = []  # 12 past TMs as input
for i in range(HIST_LEN):
    vals = np.array([float(x) for x in test_lines[start + i].strip().split()])
    tms_input.append(vals)
tms_input = np.array(tms_input)  # shape: (12, 9216)

# The target TM (last of the 12) is what we route through the network
tm_target = tms_input[-1]  # shape: (9216,)

# Non-diagonal mask
diag_mask = np.eye(NUM_NODES, dtype=bool).flatten()
tm_non_diag = tm_target[~diag_mask]  # shape: (9120,)


# ============================================================
# 2. Helper functions
# ============================================================

def load_graph(json_path):
    """Load directed graph from FIGRET-style JSON."""
    with open(json_path) as f:
        data = json.load(f)
    G = {}
    for n in data['nodes']:
        G[n['id']] = {}
    for l in data['links']:
        src, dst = l['source'], l['target']
        # Keep capacity in raw bps (traffic also in bps, same normalization)
        G[src][dst] = {
            'capacity': l['capacity'],  # raw bps, same as .hist values
            'distance_km': l.get('distance_km', 0)
        }
    return G


def load_paths():
    """Load all paths from tunnels_full.txt."""
    paths = {}
    with open(TUNNELS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(':')
            sd = parts[0].split()
            src, dst = int(sd[0]), int(sd[1])
            path_list = []
            if len(parts) > 1 and parts[1]:
                for p_str in parts[1].split(','):
                    nodes = [int(n) for n in p_str.split('-')]
                    edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
                    path_list.append(edges)
            paths[(src, dst)] = path_list
    return paths


def get_edges_map(G):
    """Build edge index mapping."""
    eid = 0
    edges_map = {}
    capacity = []
    for s in G:
        for d in G[s]:
            edges_map[(s, d)] = eid
            capacity.append(G[s][d]['capacity'])
            eid += 1
    return edges_map, np.array(capacity)


def build_matrices(G, paths, edges_map):
    """Build paths_to_edges and commodities_to_paths matrices."""
    num_edges = len(edges_map)

    pte_rows, pte_cols, pte_data = [], [], []
    ctp_rows, ctp_cols, ctp_data = [], [], []

    path_id = 0
    path_valid = []
    comm_id = 0

    for src in range(NUM_NODES):
        for dst in range(NUM_NODES):
            if src == dst: continue

            path_list = paths.get((src, dst), [])
            for p in path_list:
                # Check if all edges exist in this topology
                valid = True
                edge_ids = []
                for e in p:
                    if e not in edges_map:
                        valid = False
                        break
                    edge_ids.append(edges_map[e])

                if valid:
                    for eid in edge_ids:
                        pte_rows.append(path_id)
                        pte_cols.append(eid)
                        pte_data.append(1)

                path_valid.append(valid)
                ctp_rows.append(comm_id)
                ctp_cols.append(path_id)
                ctp_data.append(1 if valid else 0)
                path_id += 1

            comm_id += 1

    total_paths = path_id
    pte = csr_matrix((pte_data, (pte_rows, pte_cols)), shape=(total_paths, num_edges))
    ctp = csr_matrix((ctp_data, (ctp_rows, ctp_cols)), shape=(NUM_COMMODITIES, total_paths))

    return pte, ctp, np.array(path_valid), total_paths


def compute_path_distances(paths, G):
    """Compute total distance for each path."""
    path_dist = []
    for src in range(NUM_NODES):
        for dst in range(NUM_NODES):
            if src == dst: continue
            for p in paths.get((src, dst), []):
                dist = sum(G[e[0]][e[1]].get('distance_km', 0) for e in p if e[0] in G and e[1] in G[e[0]])
                path_dist.append(dist if dist > 0 else 1e6)
    return np.array(path_dist)


def compute_mlu(G, paths, edges_map, capacities, split_ratios, tm):
    """Compute MLU given split ratios and traffic matrix."""
    pte, ctp, _, _ = build_matrices(G, paths, edges_map)

    # Demand per path
    demand = ctp.T.dot(tm)  # (num_paths,)
    demand_on_paths = demand * split_ratios  # elementwise
    flow_on_edges = pte.T.dot(demand_on_paths)  # (num_edges,)

    if len(capacities) == 0 or max(capacities) == 0:
        return 1.0

    congestion = flow_on_edges / capacities
    mlu = np.max(congestion) if len(congestion) > 0 else 0
    return mlu


def figret_predict(model_input, model_params):
    """Simulate FIGRET model forward pass.
    This is a simplified MLP forward pass matching FigretNetWork architecture.
    Since we can't easily load torch models here, we'll use a placeholder.
    For actual testing, we call figret.py in test mode.
    """
    pass  # We'll compute MLU from actual FIGRET model predictions


# ============================================================
# 3. Load paths
# ============================================================
print("\nLoading paths...")
paths = load_paths()
total_paths = sum(len(v) for v in paths.values())
print(f"  Total paths: {total_paths}")


# ============================================================
# 4. Baselines
# ============================================================
print("\nComputing baselines...")

results = {'slice': [], 'model_0000': [], 'model_0001': [], 'model_0002': [],
           'model_0003': [], 'model_0004': [],
           'ecmp': [], 'spf': [], 'dist_weight': []}

for step in range(189):
    topo_file = TOPO_DIR / f"step_{step:04d}.json"
    if not topo_file.exists(): continue

    G = load_graph(topo_file)
    edges_map, capacities = get_edges_map(G)
    pte, ctp, valid_mask, num_paths = build_matrices(G, paths, edges_map)

    num_valid = valid_mask.sum()
    if num_valid == 0: continue

    # ===== ECMP: equal split across all VALID paths =====
    ecmp_split = np.zeros(num_paths)
    # For each commodity, equally split across its valid paths
    comm_start = 0
    ecmp_splits_per_comm = []
    for comm in range(NUM_COMMODITIES):
        row = ctp[comm].toarray().flatten()
        valid_in_comm = row > 0
        n_valid = valid_in_comm.sum()
        if n_valid > 0:
            ecmp_split[valid_in_comm] = 1.0 / n_valid

    # Actually, we need per-commodity normalization
    # ECMP: for each commodity, equal weight to each valid path
    comm_path_count = np.array(ctp.sum(axis=1)).flatten()
    ecmp_split = np.zeros(num_paths)
    for comm in range(NUM_COMMODITIES):
        if comm_path_count[comm] > 0:
            ecmp_split[ctp[comm].indices] = 1.0 / comm_path_count[comm]

    mlu_ecmp = compute_mlu(G, paths, edges_map, capacities, ecmp_split, tm_non_diag)

    # ===== SPF: 100% on shortest path =====
    # (Shortest = fewest hops, i.e., first valid path per commodity from tunnels)
    spf_split = np.zeros(num_paths)
    for comm in range(NUM_COMMODITIES):
        row = ctp[comm]
        if row.nnz > 0:
            first_valid = row.indices[0]
            spf_split[first_valid] = 1.0

    mlu_spf = compute_mlu(G, paths, edges_map, capacities, spf_split, tm_non_diag)

    # ===== Distance-Weighted: inverse distance =====
    path_dists = compute_path_distances(paths, G)
    dw_split = np.zeros(num_paths)
    for comm in range(NUM_COMMODITIES):
        row = ctp[comm]
        if row.nnz > 0:
            path_ids = row.indices
            dists = path_dists[path_ids]
            if dists.sum() > 0:
                weights = 1.0 / (dists + 1e-6)
                dw_split[path_ids] = weights / weights.sum()

    mlu_dw = compute_mlu(G, paths, edges_map, capacities, dw_split, tm_non_diag)

    results['slice'].append(step)
    results['ecmp'].append(mlu_ecmp)
    results['spf'].append(mlu_spf)
    results['dist_weight'].append(mlu_dw)

    # Placeholder for model results (filled after training)
    for m in range(5):
        results[f'model_{m:04d}'].append(np.nan)

    if step % 20 == 0:
        print(f"  [{step:3d}/189] ECMP={mlu_ecmp:.4f} SPF={mlu_spf:.4f} DW={mlu_dw:.4f}")

# ============================================================
# 5. Save baseline results
# ============================================================
import csv
csv_path = RESULT_DIR / "baselines.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['slice', 'ecmp', 'spf', 'dist_weight'])
    for i in range(len(results['slice'])):
        writer.writerow([results['slice'][i], results['ecmp'][i],
                        results['spf'][i], results['dist_weight'][i]])
print(f"\nBaselines saved to {csv_path}")

# ============================================================
# 6. Quick plot of baselines
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
slices_arr = np.array(results['slice'])
ax.plot(slices_arr, results['ecmp'], label='ECMP (equal split)', linewidth=1.5, alpha=0.8)
ax.plot(slices_arr, results['spf'], label='SPF (shortest path first)', linewidth=1.5, alpha=0.8)
ax.plot(slices_arr, results['dist_weight'], label='Distance-Weighted', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Time Slice', fontsize=12)
ax.set_ylabel('MLU (Max Link Utilization)', fontsize=12)
ax.set_title('Baseline Algorithms across Time Slices', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plot_path = RESULT_DIR / "baselines_plot.png"
fig.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")

print("\nPhase 1 baselines done. Model results to be added after training.")
