"""
Phase 2 grid search: alpha=0.03 fixed, test beta = 0.01, 0.05, 0.1.
Multi-topology training with 2 epochs each.
"""
import sys, json, torch, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix
from networkx.readwrite import json_graph
from tqdm import tqdm

TRAIN_SLICES = list(range(50))
TOPO_DIR = Path("Data/leo/topologies")
TUNNELS_FILE = Path("Data/leo/tunnels_full.txt")
TRAIN_HIST = Path("Data/leo/train/0.hist")
MODEL_DIR = Path("Model")
HIST_LEN, NUM_NODES, NUM_COMMODITIES = 12, 96, 96*95
ALPHA = 0.03
BETAS = [0.01, 0.05, 0.1]
EPOCHS = 2

# Load data once
print("Loading data...")
with open(TRAIN_HIST) as f:
    tm_data = np.array([[float(x) for x in l.strip().split()] for l in f.readlines()])
diag_mask = np.eye(NUM_NODES, dtype=bool).flatten()
tm_data = tm_data[:, ~diag_mask] / 1e9

X_list, Y_list = [], []
for i in range(len(tm_data) - HIST_LEN):
    X_list.append(tm_data[i:i+HIST_LEN].flatten())
    Y_list.append(np.append(tm_data[i+HIST_LEN], 1.0))
X_train = np.array(X_list)
Y_train = np.array(Y_list)
print(f"  Train samples: {len(X_train)}")

with open("Data/leo/path_survival_train.json") as f:
    path_survival = torch.tensor(json.load(f)['survival_rates']).double()

# Pre-build topology data
def build_topo(step):
    with open(TOPO_DIR / f"step_{step:04d}.json") as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data)
    nn, ne = G.number_of_nodes(), G.number_of_edges()
    adj = np.zeros((nn, nn))
    for s in range(nn):
        for d in range(nn):
            if s != d and d in G[s]: adj[s,d] = 1
    emap, cap = {}, []
    for i in range(nn):
        for j in range(nn):
            if adj[i,j]==1: emap[(i,j)]=len(emap); cap.append(G[i][j]['capacity']/1e9)
    pij = defaultdict(list)
    with open(TUNNELS_FILE) as f: ld = {l.split(':')[0]:l.strip() for l in sorted(f.readlines()) if ':' in l}
    for src in range(nn):
        for dst in range(nn):
            if src==dst: continue
            k = f"{src} {dst}"
            if k in ld:
                for p_ in ld[k].split(':')[1].split(','):
                    nl = list(map(int, p_.split('-')))
                    pij[(src,dst)].append([(v1,v2) for v1,v2 in zip(nl, nl[1:])])
    pa,vm = [],[]
    for i in range(nn):
        for j in range(nn):
            if i==j: continue
            for p in pij.get((i,j),[]):
                try:
                    r=np.zeros((ne,)); [r.__setitem__(emap[e],1) for e in p]
                    pa.append(r); vm.append(True)
                except KeyError: pa.append(np.zeros((ne,))); vm.append(False)
    pte = csr_matrix(np.stack(pa)); vmask=np.array(vm)
    ctp = lil_matrix((NUM_COMMODITIES, pte.shape[0]))
    cid,pid=0,0
    for i in range(nn):
        for j in range(nn):
            if i==j: continue
            for _ in pij.get((i,j),[]):
                if vmask[pid]: ctp[cid,pid]=1
                pid+=1
            cid+=1
    ctp_c=ctp.tocoo(); pte_c=pte.tocoo()
    C=torch.sparse_coo_tensor(np.vstack((ctp_c.row,ctp_c.col)),torch.DoubleTensor(ctp_c.data),torch.Size(ctp_c.shape))
    P=torch.sparse_coo_tensor(np.vstack((pte_c.row,pte_c.col)),torch.DoubleTensor(pte_c.data),torch.Size(pte_c.shape))
    return C,P,torch.tensor(cap).unsqueeze(1).double(),torch.BoolTensor(vmask),ne

print("Building topologies...")
topos = {s: build_topo(s) for s in tqdm(TRAIN_SLICES)}
tm_hist_std = torch.tensor(tm_data.std(axis=0)).double()

from src.figret_net import FigretNetWork

results = []
for beta in BETAS:
    print(f"\n{'='*50}\nbeta={beta}\n{'='*50}")
    model = FigretNetWork(HIST_LEN*NUM_COMMODITIES, len(path_survival), 3).double()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(EPOCHS):
        model.train()
        epoch_vals = []
        indices = np.random.permutation(len(X_train))
        with tqdm(total=len(indices), desc=f"b={beta} e{epoch+1}/{EPOCHS}") as pbar:
            for idx in indices:
                ts = np.random.choice(TRAIN_SLICES)
                C, P, cap, vmask, ne = topos[ts]
                Xb = torch.DoubleTensor(X_train[idx]).unsqueeze(0)
                Yb = torch.DoubleTensor(Y_train[idx]).unsqueeze(0)
                yp = model(Xb)

                # Loss
                yp_ = yp[0]+1e-16; yt_=Yb[0,:NUM_COMMODITIES]; opt=Yb[0,-1].item()
                pw = yp_.unsqueeze(1)
                pw[~vmask] = 1e-16
                tw = C.matmul(pw); split = pw.mul(C.t().matmul(1.0/tw))
                sc = split.clone()
                dmd = C.t().matmul(yt_.unsqueeze(1))
                flow = P.t().matmul(dmd.mul(split))
                mc = torch.max(flow.divide(cap).flatten())

                # Simplified sensitivity (max split per commodity ≈ variance of routing)
                sf = split.squeeze()
                sum_sens = torch.mean(sf)

                surv_pen = torch.mean(sc.squeeze() * (1.0 - path_survival))
                surv_val = surv_pen.item()

                loss = mc/mc.item() + ALPHA*sum_sens/(sum_sens.item()+1e-16) + beta*surv_pen/(surv_val+1e-16)
                loss_val = mc.item()/opt if opt!=0 else 1.0

                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_vals.append(loss_val)
                pbar.update(1); pbar.set_postfix(mlu=loss_val)

        avg = sum(epoch_vals)/len(epoch_vals)
        print(f"  Epoch {epoch+1}: avg MLU = {avg:.4f}")

    final = epoch_vals[-1] if epoch_vals else 999
    avg_all = sum(epoch_vals)/len(epoch_vals) if epoch_vals else 999
    results.append((beta, avg_all))
    torch.save(model, MODEL_DIR / f"leo_phase2_b{beta}.pt")
    print(f"  Saved leo_phase2_b{beta}.pt | avg MLU={avg_all:.4f}")

print(f"\n{'='*50}")
print("Grid Search Results:")
for b, mlu in sorted(results, key=lambda x: x[1]):
    marker = " < BEST" if b == sorted(results, key=lambda x: x[1])[0][0] else ""
    print(f"  beta={b:.2f}  avg MLU={mlu:.4f}{marker}")
