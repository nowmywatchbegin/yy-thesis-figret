"""
Usage: python phase1_test_model.py <train_slice> [<test_start> <test_end>]

Example: python phase1_test_model.py 0        # test model 0000 on slices 1-50
         python phase1_test_model.py 1  2 10  # test model 0001 on slices 2-10
         python phase1_test_model.py 2  3 52  # test model 0002 on slices 3-52
"""
import json, csv, torch, numpy as np, sys
from pathlib import Path
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict
from networkx.readwrite import json_graph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----- Args -----
TRAIN_SLICE = int(sys.argv[1])
TEST_START = int(sys.argv[2]) if len(sys.argv) > 2 else TRAIN_SLICE + 1
TEST_END = int(sys.argv[3]) if len(sys.argv) > 3 else TRAIN_SLICE + 50

MODEL_PATH = f"Model/leo_step_{TRAIN_SLICE:04d}.pt"
TOPO_DIR = Path("Data/leo/topologies")
TUNNELS_FILE = Path("Data/leo/tunnels_full.txt")
BASELINE_CSV = Path("Result/leo/baselines.csv")
RESULT_DIR = Path("Result/leo")
HIST_LEN, NUM_NODES = 12, 96

# ----- Load data -----
np.random.seed(42)
with open("Data/leo/test/0.hist") as f:
    lines = f.readlines()
start = np.random.randint(0, len(lines) - HIST_LEN)
tm_input = np.array([[float(x) for x in lines[start + i].strip().split()]
                      for i in range(HIST_LEN)])

model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.eval()


# ----- Build env from topology JSON -----
def build_env(topo_path):
    with open(topo_path) as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data)
    nn, ne = G.number_of_nodes(), G.number_of_edges()
    adj = np.zeros((nn, nn))
    for s in range(nn):
        for d in range(nn):
            if s != d and d in G[s]: adj[s, d] = 1
    eid, emap, cap = 0, {}, []
    for i in range(nn):
        for j in range(nn):
            if adj[i, j] == 1:
                emap[(i, j)] = eid
                cap.append(G[i][j]['capacity'] / 1e9)
                eid += 1
    pij = defaultdict(list)
    with open(TUNNELS_FILE) as f:
        ld = {}
        for line in sorted(f.readlines()):
            line = line.strip()
            if ':' in line: ld[line.split(':')[0]] = line
    for src in range(nn):
        for dst in range(nn):
            if src == dst: continue
            k = f"{src} {dst}"
            if k in ld:
                for p_ in ld[k].split(':')[1].split(','):
                    nl = list(map(int, p_.split('-')))
                    pij[(src, dst)].append([(v1, v2) for v1, v2 in zip(nl, nl[1:])])
    pa, vm = [], []
    for i in range(nn):
        for j in range(nn):
            if i == j: continue
            for p in pij.get((i, j), []):
                try:
                    r = np.zeros((ne,))
                    for e in p: r[emap[e]] = 1
                    pa.append(r); vm.append(True)
                except KeyError:
                    pa.append(np.zeros((ne,))); vm.append(False)
    pte = csr_matrix(np.stack(pa))
    vm = np.array(vm)
    ctp = lil_matrix((nn*(nn-1), pte.shape[0]))
    cid, pid = 0, 0
    for i in range(nn):
        for j in range(nn):
            if i == j: continue
            n = len(pij.get((i, j), []))
            for _ in range(n):
                if vm[pid]: ctp[cid, pid] = 1
                pid += 1
            cid += 1
    return ne, cap, pte, vm, csr_matrix(ctp)


# ----- Compute MLU -----
def compute_mlu(model, env, tm_12):
    ne, cap, pte, vm, ctp = env
    ctp_c, pte_c = ctp.tocoo(), pte.tocoo()
    C = torch.sparse_coo_tensor(np.vstack((ctp_c.row, ctp_c.col)),
        torch.DoubleTensor(ctp_c.data), torch.Size(ctp_c.shape))
    P = torch.sparse_coo_tensor(np.vstack((pte_c.row, pte_c.col)),
        torch.DoubleTensor(pte_c.data), torch.Size(pte_c.shape))
    cap_t = torch.tensor(cap).unsqueeze(1)
    vmask = torch.BoolTensor(vm)
    dm = ~np.eye(NUM_NODES, dtype=bool).flatten()
    X = torch.DoubleTensor((tm_12[:, dm] / 1e9).flatten()).unsqueeze(0)
    with torch.no_grad():
        y = model(X) + 1e-16
        pw = y.squeeze(0).unsqueeze(1)
        pw[~vmask] = 1e-16
        tw = C.matmul(pw)
        split = pw.mul(C.transpose(0,1).matmul(1.0/tw))
        yt = torch.DoubleTensor((tm_12[-1][dm] / 1e9)).unsqueeze(0)
        flow = P.transpose(0,1).matmul(C.transpose(0,1).matmul(yt.transpose(0,1)).mul(split))
        return max(0, torch.max(flow.divide(cap_t).flatten()).item())


# ----- Test -----
print(f"Model: leo_step_{TRAIN_SLICE:04d}.pt (trained on slice {TRAIN_SLICE})")
print(f"Test:   slices {TEST_START}~{TEST_END}")

figret = {}
for ts in range(TEST_START, TEST_END + 1):
    env = build_env(str(TOPO_DIR / f"step_{ts:04d}.json"))
    figret[ts] = compute_mlu(model, env, tm_input)
    if ts % 25 == 0 or ts == TEST_START:
        print(f"  slice {ts:04d}: MLU = {figret[ts]:.4f}")

with open(BASELINE_CSV) as f:
    base = {int(r['slice']): {'e': float(r['ecmp']), 's': float(r['spf']),
            'd': float(r['dist_weight'])} for r in csv.DictReader(f)}

# Find failures
failures = []
for ts in range(TEST_START, TEST_END + 1):
    b = base[ts]
    if figret[ts] > b['e']: failures.append((ts, 'ECMP'))
    if figret[ts] > b['s']: failures.append((ts, 'SPF'))
    if figret[ts] > b['d']: failures.append((ts, 'DW'))

first = min(set(f[0] for f in failures)) if failures else None
print(f"First failure: slice {first}")
print(f"Total slices where FIGRET lost to at least 1 baseline: {len({f[0] for f in failures})}/{TEST_END - TEST_START + 1}")

# ----- Plot -----
fig, ax = plt.subplots(figsize=(16, 7))
slices = list(range(TEST_START, TEST_END + 1))
ax.plot(slices, [base[s]['e'] for s in slices], 'gray', linewidth=1.2, alpha=0.5, label='ECMP')
ax.plot(slices, [base[s]['s'] for s in slices], 'brown', linewidth=1.2, alpha=0.5, label='SPF')
ax.plot(slices, [base[s]['d'] for s in slices], 'orange', linewidth=1.2, alpha=0.5, label='Dist-Weighted')
ax.plot(slices, [figret[s] for s in slices], 'blue', linewidth=2.5, marker='.', markersize=4,
        label=f'FIGRET (trained on slice {TRAIN_SLICE})')

if first:
    ax.axvline(x=first, color='red', linestyle='--', alpha=0.6)
    ax.annotate(f' First failure at slice {first}', xy=(first, figret[first]),
                fontsize=11, color='red', fontweight='bold')

ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.4)
ax.set_xlabel('Time Slice', fontsize=13)
ax.set_ylabel('MLU', fontsize=13)
ax.set_title(f'FIGRET Trained on Slice {TRAIN_SLICE}  ->  Tested on Slices {TEST_START}-{TEST_END}', fontsize=15, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = RESULT_DIR / f"figret_000{TRAIN_SLICE}_trained_on_slice{TRAIN_SLICE}_tested_on_{TEST_START}to{TEST_END}.png"
fig.savefig(out, dpi=150)
print(f"Plot saved to {out}")
