"""
Run AFTER phase1_train.py completes.
Tests the 5 trained FIGRET models on their target time slices,
computes MLU, and generates the final comparison plot.
"""
import json, math, csv, torch, sys, numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path('src')))
from figret_env import FigretEnv
from figret_model import Figret, FigretDataset
from figret_net import FigretNetWork
from utils import normalize_size
import config

TOPO_DIR = Path("Data/leo/topologies")
RESULT_DIR = Path("Result/leo")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

HIST_LEN = 12
NUM_NODES = 96
NUM_COMMODITIES = NUM_NODES * (NUM_NODES - 1)

# Load test traffic
print("Loading test traffic...")
with open("Data/leo/test/0.hist") as f:
    test_lines = f.readlines()
np.random.seed(42)
start = np.random.randint(0, len(test_lines) - HIST_LEN)
print(f"  Using lines {start} to {start + HIST_LEN - 1}")

# ============================================================
# Define a test helper
# ============================================================
def test_model_on_slice(model, topo_json_path, tunnels_path, tm_12):
    """Compute MLU of a trained model on a specific topology + traffic."""
    from argparse import Namespace
    props = Namespace(
        topo_name=None,  # We build env manually
        hist_len=HIST_LEN,
        alpha=0.03,
        paths_file=str(tunnels_path),
        mode='test',
        batch_size=1,
        epochs=1,
        num_layer=3,
        opt_name=''
    )

    # Build env from the given topology JSON
    env = FigretEnv.__new__(FigretEnv)
    env.props = props
    env.topo_name = Path(topo_json_path).stem

    # Hack: override DATA_DIR to read from exact JSON path
    import networkx as nx
    from networkx.readwrite import json_graph
    with open(topo_json_path) as f:
        data = json.load(f)
    env.G = json_graph.node_link_graph(data)
    env.num_nodes = env.G.number_of_nodes()
    env.num_edges = env.G.number_of_edges()

    import numpy as np
    from collections import defaultdict

    # Adjacency
    env.adj = np.zeros((env.num_nodes, env.num_nodes))
    for s in range(env.num_nodes):
        for d in range(env.num_nodes):
            if s == d: continue
            if d in env.G[s]:
                env.adj[s, d] = 1

    # Edges map
    eid = 0
    env.edges_map = {}
    env.capacity = []
    for i in range(env.num_nodes):
        for j in range(env.num_nodes):
            if env.adj[i, j] == 1:
                env.edges_map[(i, j)] = eid
                cap = env.G[i][j].get('capacity', 1e9)
                env.capacity.append(normalize_size(cap))
                eid += 1

    # Paths
    env.pij = defaultdict(list)
    with open(tunnels_path) as f:
        lines = sorted(f.readlines())
        lines_dict = {line.split(":")[0]: line for line in lines if line.strip()}
    for src in range(env.num_nodes):
        for dst in range(env.num_nodes):
            if src == dst: continue
            key = f"{src} {dst}"
            if key in lines_dict:
                line = lines_dict[key].strip()
                if line:
                    paths_str = line.split(":")[1].split(",")
                    for p_ in paths_str:
                        node_list = list(map(int, p_.split("-")))
                        edge_list = [(v1, v2) for v1, v2 in zip(node_list, node_list[1:])]
                        env.pij[(src, dst)].append(edge_list)

    # paths_to_edges
    paths_arr = []
    env.valid_path_mask = []
    for i in range(env.num_nodes):
        for j in range(env.num_nodes):
            if i == j: continue
            for p in env.pij.get((i, j), []):
                try:
                    p_ = [env.edges_map[e] for e in p]
                    p__ = np.zeros((int(env.num_edges),))
                    for k in p_:
                        p__[k] = 1
                    paths_arr.append(p__)
                    env.valid_path_mask.append(True)
                except KeyError:
                    p__ = np.zeros((int(env.num_edges),))
                    paths_arr.append(p__)
                    env.valid_path_mask.append(False)
    env.valid_path_mask = np.array(env.valid_path_mask)
    env.paths_to_edges = csr_matrix(np.stack(paths_arr))
    env.num_paths = env.paths_to_edges.shape[0]

    # commodities_to_paths
    from scipy.sparse import lil_matrix
    commodities_to_paths = lil_matrix((env.num_nodes * (env.num_nodes - 1), env.num_paths))
    commid, pathid = 0, 0
    env.commodities_to_path_nums = []
    for src in range(env.num_nodes):
        for dst in range(env.num_nodes):
            if src == dst: continue
            n_paths = len(env.pij.get((src, dst), []))
            for _ in range(n_paths):
                if env.valid_path_mask[pathid]:
                    commodities_to_paths[commid, pathid] = 1
                pathid += 1
            env.commodities_to_path_nums.append(n_paths)
            commid += 1
    env.commodities_to_paths = csr_matrix(commodities_to_paths)
    env.constant_pathlen = env.commodities_to_path_nums.count(env.commodities_to_path_nums[0]) == len(env.commodities_to_path_nums)

    # Build Figret object for loss computation
    device = torch.device('cpu')
    figret = Figret.__new__(Figret)
    figret.env = env
    figret.device = device

    # Build sparse tensors
    ctp_coo = env.commodities_to_paths.tocoo()
    figret.commodities_to_paths = torch.sparse_coo_tensor(
        np.vstack((ctp_coo.row, ctp_coo.col)),
        torch.DoubleTensor(ctp_coo.data),
        torch.Size(ctp_coo.shape)).to(device)

    pte_coo = env.paths_to_edges.tocoo()
    figret.paths_to_edges = torch.sparse_coo_tensor(
        np.vstack((pte_coo.row, pte_coo.col)),
        torch.DoubleTensor(pte_coo.data),
        torch.Size(pte_coo.shape)).to(device)

    figret.edges_capacity = torch.tensor(env.capacity).unsqueeze(1).to(device)
    figret.valid_path_mask = torch.BoolTensor(env.valid_path_mask).to(device)

    # Dummy tm_hist_std (not needed for MLU computation, only for sensitivity)
    figret.tm_hist_std = torch.ones(NUM_COMMODITIES).to(device)
    figret.props = props

    # Model forward pass
    model.eval()
    with torch.no_grad():
        # Prepare input: flatten 12 past TMs
        X = tm_12.flatten()
        X_tensor = torch.DoubleTensor(X).unsqueeze(0)
        y_pred = model(X_tensor)  # shape: (1, num_paths)

        # Compute MLU from model's split ratios
        y_pred = y_pred + 1e-16
        paths_weight = y_pred.squeeze(0).unsqueeze(1)

        # Zero invalid paths
        invalid_mask = ~figret.valid_path_mask
        paths_weight[invalid_mask] = 1e-16

        # Normalize per commodity
        commodity_total_weight = figret.commodities_to_paths.matmul(paths_weight)
        paths_over_total = figret.commodities_to_paths.transpose(0, 1).matmul(1.0 / commodity_total_weight)
        split_ratios = paths_weight.mul(paths_over_total)

        # Traffic on edges
        tm_tensor = torch.DoubleTensor(normalize_size(tm_12[-1]))
        tm_non_diag = tm_tensor[~np.eye(NUM_NODES, dtype=bool).flatten()]
        y_true = tm_non_diag.unsqueeze(0)

        tmp_demand = figret.commodities_to_paths.transpose(0, 1).matmul(y_true.transpose(0, 1))
        demand_on_paths = tmp_demand.mul(split_ratios)
        flow_on_edges = figret.paths_to_edges.transpose(0, 1).matmul(demand_on_paths)
        congestion = flow_on_edges.divide(figret.edges_capacity)
        mlu = torch.max(congestion.flatten()).item()

    return max(mlu, 0)


# ============================================================
# Load paths reference
# ============================================================
TUNNELS_PATH = Path("Data/leo/tunnels_full.txt")

# Load test TMs
test_tms = []
for i in range(start, start + HIST_LEN):
    vals = np.array([float(x) for x in test_lines[i].strip().split()])
    test_tms.append(vals)
tm_input = np.array(test_tms)

# ============================================================
# Test each model on its target slices
# ============================================================
models_def = {
    'model_0000': (0, list(range(1, 11))),   # trained on 0, test on 1-10
    'model_0001': (1, list(range(2, 12))),   # trained on 1, test on 2-11
    'model_0002': (2, list(range(3, 13))),   # etc.
    'model_0003': (3, list(range(4, 14))),
    'model_0004': (4, list(range(5, 15))),
}

# Load baseline results
baseline_csv = RESULT_DIR / "baselines.csv"
results = {'slice': [], 'ecmp': [], 'spf': [], 'dist_weight': []}
for k in models_def:
    results[k] = []

if baseline_csv.exists():
    with open(baseline_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results['slice'].append(int(row['slice']))
            results['ecmp'].append(float(row['ecmp']))
            results['spf'].append(float(row['spf']))
            results['dist_weight'].append(float(row['dist_weight']))
            for k in models_def:
                results[k].append(np.nan)

print("\nTesting trained models...")
for model_name, (train_slice, test_slices) in models_def.items():
    model_path = f"Model/leo_step_{train_slice:04d}.pt"
    if not Path(model_path).exists():
        print(f"  {model_name}: model not found ({model_path}), skipping")
        continue

    model = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"  {model_name}: testing on slices {test_slices}")

    for ts in test_slices:
        if ts >= 189:
            continue
        topo_json = TOPO_DIR / f"step_{ts:04d}.json"
        if not topo_json.exists():
            continue

        try:
            mlu = test_model_on_slice(model, str(topo_json), str(TUNNELS_PATH), tm_input)
            # Find the row for this slice
            idx = results['slice'].index(ts) if ts in results['slice'] else -1
            if idx >= 0:
                results[model_name][idx] = mlu
            print(f"    slice {ts:04d}: MLU = {mlu:.4f}")
        except Exception as e:
            print(f"    slice {ts:04d}: ERROR - {e}")

# Save combined results
combined_csv = RESULT_DIR / "all_results.csv"
with open(combined_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['slice', 'ecmp', 'spf', 'dist_weight'] + list(models_def.keys())
    writer.writerow(header)
    for i in range(len(results['slice'])):
        row = [results['slice'][i], results['ecmp'][i], results['spf'][i],
               results['dist_weight'][i]]
        for k in models_def:
            row.append(results[k][i])
        writer.writerow(row)
print(f"\nAll results saved to {combined_csv}")

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(16, 7))

slices = np.array(results['slice'])

# Baselines
ax.plot(slices, results['ecmp'], label='ECMP (equal split)', linewidth=2, alpha=0.7, linestyle='--', color='gray')
ax.plot(slices, results['spf'], label='SPF (shortest path)', linewidth=2, alpha=0.7, linestyle='--', color='brown')
ax.plot(slices, results['dist_weight'], label='Dist-Weighted', linewidth=2, alpha=0.7, linestyle='--', color='orange')

# FIGRET models
colors = ['blue', 'green', 'red', 'purple', 'teal']
for i, (model_name, (train_slice, _)) in enumerate(models_def.items()):
    vals = np.array(results[model_name])
    mask = ~np.isnan(vals)
    if mask.sum() > 0:
        ax.plot(slices[mask], vals[mask], label=f'FIGRET (trained on slice {train_slice})',
                linewidth=2.5, color=colors[i], marker='o', markersize=3)

ax.set_xlabel('Time Slice', fontsize=12)
ax.set_ylabel('MLU (Max Link Utilization)', fontsize=12)
ax.set_title('FIGRET vs Baselines across Time Slices', fontsize=14)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='MLU = 1.0 (saturation)')

fig.tight_layout()
final_plot = RESULT_DIR / "phase1_final_plot.png"
fig.savefig(final_plot, dpi=150)
print(f"Final plot saved to {final_plot}")
print("\nPhase 1 complete.")
