"""
Phase 2: Multi-topology training with path survival penalty.

Key features:
- Each batch uses a randomly selected topology from training set (slices 0-49)
- Path survival rates computed from training set only
- Loss = MLU + alpha * sensitivity + beta * path_survival_penalty
"""
import json, os, torch, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix
from networkx.readwrite import json_graph
from tqdm import tqdm

# ----- Config -----
TRAIN_SLICES = list(range(50))
TOPO_DIR = Path("Data/leo/topologies")
TUNNELS_FILE = Path("Data/leo/tunnels_full.txt")
TRAIN_HIST = Path("Data/leo/train/0.hist")
TEST_HIST = Path("Data/leo/test/0.hist")
MODEL_DIR = Path("Model")
RESULT_DIR = Path("Result/leo")
DEVICE = 'cpu'

HIST_LEN = 12
NUM_NODES = 96
NUM_COMMODITIES = NUM_NODES * (NUM_NODES - 1)
ALPHA = 0.03
BETA = 0.05
EPOCHS = 2  # quick test
BATCH_SIZE = 1


# ----- Load data -----
print("Loading data...")
with open(TRAIN_HIST) as f:
    tm_data = np.array([[float(x) for x in l.strip().split()] for l in f.readlines()])
# Filter diagonal
diag_mask = np.eye(NUM_NODES, dtype=bool).flatten()
tm_data = tm_data[:, ~diag_mask]  # (N, 9120)
# Normalize
tm_data = tm_data / 1e9

# Build X, Y for FIGRET
X_list, Y_list = [], []
for i in range(len(tm_data) - HIST_LEN):
    X_list.append(tm_data[i:i+HIST_LEN].flatten())
    Y_list.append(np.append(tm_data[i+HIST_LEN], 1.0))  # append dummy opt
X_train = np.array(X_list)
Y_train = np.array(Y_list)
print(f"  Train: {len(X_train)} samples (2400 - {HIST_LEN})")


# ----- Load path survival (training set only) -----
with open("Data/leo/path_survival_train.json") as f:
    path_survival = torch.tensor(json.load(f)['survival_rates']).double().to(DEVICE)
print(f"  Path survival: {len(path_survival)} paths")


# ----- Pre-build topology data for each training slice -----
def build_topology_data(topo_json):
    with open(topo_json) as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data)
    nn, ne = G.number_of_nodes(), G.number_of_edges()

    # edges_map & capacity
    emap = {}
    cap = []

    # adjacency
    for s in range(nn):
        for d in range(nn):
            if s != d and d in G[s]:
                emap[(s, d)] = len(emap)
                cap.append(G[s][d]['capacity'] / 1e9)

    # paths from tunnels
    pij = defaultdict(list)
    with open(TUNNELS_FILE) as f:
        ld = {}
        for l in sorted(f.readlines()):
            l = l.strip()
            if ':' in l: ld[l.split(':')[0]] = l
    for src in range(nn):
        for dst in range(nn):
            if src == dst: continue
            k = f"{src} {dst}"
            if k in ld:
                for p_ in ld[k].split(':')[1].split(','):
                    nl = list(map(int, p_.split('-')))
                    pij[(src, dst)].append([(v1, v2) for v1, v2 in zip(nl, nl[1:])])

    # paths_to_edges + valid_mask
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
    vmask = np.array(vm)

    # commodities_to_paths
    ctp = lil_matrix((NUM_COMMODITIES, pte.shape[0]))
    cid, pid = 0, 0
    pn = []
    for i in range(nn):
        for j in range(nn):
            if i == j: continue
            n = len(pij.get((i, j), []))
            for _ in range(n):
                if vmask[pid]: ctp[cid, pid] = 1
                pid += 1
            pn.append(n); cid += 1

    # Convert to torch sparse
    ctp_c = ctp.tocoo()
    C = torch.sparse_coo_tensor(
        np.vstack((ctp_c.row, ctp_c.col)),
        torch.DoubleTensor(ctp_c.data), torch.Size(ctp_c.shape))

    pte_c = pte.tocoo()
    P = torch.sparse_coo_tensor(
        np.vstack((pte_c.row, pte_c.col)),
        torch.DoubleTensor(pte_c.data), torch.Size(pte_c.shape))

    capacity_t = torch.tensor(cap).unsqueeze(1).double()
    vmask_t = torch.BoolTensor(vmask)

    return C, P, capacity_t, vmask_t, pn, ne


print("Building topology data for 50 training slices...")
topo_data = {}
for step in tqdm(TRAIN_SLICES):
    topo_data[step] = build_topology_data(str(TOPO_DIR / f"step_{step:04d}.json"))
print(f"  Done: {len(topo_data)} topologies loaded")


# ----- Model -----
from src.figret_net import FigretNetWork

# Compute tm_hist_std from training data
tm_hist_std = torch.tensor(tm_data.std(axis=0)).double().to(DEVICE)

input_dim = HIST_LEN * NUM_COMMODITIES
output_dim = len(path_survival)
model = FigretNetWork(input_dim, output_dim, 3).double().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())


# ----- Training -----
def compute_loss(y_pred, y_true, C, P, cap, vmask):
    """FIGRET loss with path survival penalty."""
    batch_size = y_pred.shape[0]
    losses, loss_vals = [], []

    for i in range(batch_size):
        yp = y_pred[[i]] + 1e-16
        yt = y_true[[i]]
        opt = yt[0, -1].item()
        yt = yt[:, :NUM_COMMODITIES]

        pw = yp.squeeze(0).unsqueeze(1)
        pw[~vmask] = 1e-16

        total_w = C.matmul(pw)
        paths_over = C.transpose(0, 1).matmul(1.0 / total_w)
        split = pw.mul(paths_over)
        split_clone = split.clone()  # for survival penalty

        demand = C.transpose(0, 1).matmul(yt.transpose(0, 1))
        routed = demand.mul(split)
        flow = P.transpose(0, 1).matmul(routed)
        cong = flow.divide(cap)
        max_cong = torch.max(cong.flatten())

        # Sensitivity penalty
        # Split per commodity for max sensitivity
        split_flat = split.squeeze()
        max_sens_list = []
        start = 0
        for n in topo_data[list(topo_data.keys())[0]][4]:
            if n > 0:
                max_sens_list.append(torch.max(split_flat[start:start+n]))
            start += n
        if max_sens_list:
            max_sens = torch.stack(max_sens_list)
            wgt_sens = max_sens * tm_hist_std
            sum_sens = wgt_sens.mean()
        else:
            sum_sens = torch.tensor(0.0).to(DEVICE)

        # Path survival penalty
        surv_pen = torch.mean(split_clone.squeeze() * (1.0 - path_survival))

        if max_cong.item() == 0.0:
            loss = 1.0 - max_cong
        else:
            loss = (max_cong / max_cong.item()
                    + ALPHA * sum_sens / (sum_sens.item() + 1e-16)
                    + BETA * surv_pen / (surv_pen.item() + 1e-16))

        loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt
        losses.append(loss)
        loss_vals.append(loss_val)

    return sum(losses) / len(losses), sum(loss_vals) / len(loss_vals)


print(f"\nTraining {EPOCHS} epochs with multi-topology...")
print(f"  alpha={ALPHA}, beta={BETA}")

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = []
    indices = np.random.permutation(len(X_train))

    with tqdm(total=len(indices), desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
        for idx in indices:
            # Randomly pick a training topology
            topo_idx = np.random.choice(TRAIN_SLICES)
            C, P, cap, vmask, pn, ne = topo_data[topo_idx]

            X_batch = torch.DoubleTensor(X_train[idx]).unsqueeze(0).to(DEVICE)
            Y_batch = torch.DoubleTensor(Y_train[idx]).unsqueeze(0).to(DEVICE)

            y_pred = model(X_batch)
            loss, loss_val = compute_loss(y_pred, Y_batch, C, P, cap, vmask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss_val)
            pbar.update(1)
            pbar.set_postfix(loss_val=loss_val)

    avg_val = sum(epoch_losses) / len(epoch_losses)
    print(f"  Epoch {epoch+1}: avg loss_val = {avg_val:.4f}")

# Save model
model_name = f"leo_phase2_a{ALPHA}_b{BETA}.pt"
torch.save(model, MODEL_DIR / model_name)
print(f"\nModel saved to {MODEL_DIR / model_name}")
print("Phase 2 training complete.")
