"""
Phase 2 final training: multi-topology + beta=0.01, 10 epochs.
"""
import json, torch, numpy as np
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
HIST_LEN, NN, NC = 12, 96, 96*95
ALPHA, BETA, EPOCHS = 0.03, 0.01, 10

# Load data
print("Loading data...")
with open(TRAIN_HIST) as f:
    tm_data = np.array([[float(x) for x in l.strip().split()] for l in f.readlines()])
dm = ~np.eye(NN, dtype=bool).flatten()
tm_data = tm_data[:, dm] / 1e9
Xl, Yl = [], []
for i in range(len(tm_data) - HIST_LEN):
    Xl.append(tm_data[i:i+HIST_LEN].flatten())
    Yl.append(np.append(tm_data[i+HIST_LEN], 1.0))
X = np.array(Xl); Y = np.array(Yl)
print(f"  {len(X)} samples")

with open("Data/leo/path_survival_train.json") as f:
    ps = torch.tensor(json.load(f)['survival_rates']).double()
tm_std = torch.tensor(tm_data.std(axis=0)).double()

# Build topologies
def build(step):
    with open(TOPO_DIR / f"step_{step:04d}.json") as f: data = json.load(f)
    G = json_graph.node_link_graph(data)
    nn, ne = G.number_of_nodes(), G.number_of_edges()
    adj = np.zeros((nn,nn), dtype=bool)
    for s in range(nn):
        for d in range(nn):
            if s!=d and d in G[s]: adj[s,d]=True
    em, cap = {}, []
    for i in range(nn):
        for j in range(nn):
            if adj[i,j]: em[(i,j)]=len(em); cap.append(G[i][j]['capacity']/1e9)
    pij = defaultdict(list)
    with open(TUNNELS_FILE) as f:
        ld = {l.split(':')[0]:l.strip() for l in sorted(f.readlines()) if ':' in l}
    for src in range(nn):
        for dst in range(nn):
            if src==dst: continue
            k=f"{src} {dst}"
            if k in ld:
                for p_ in ld[k].split(':')[1].split(','):
                    nl=list(map(int,p_.split('-')))
                    pij[(src,dst)].append([(v1,v2) for v1,v2 in zip(nl,nl[1:])])
    pa,vm=[],[]
    for i in range(nn):
        for j in range(nn):
            if i==j: continue
            for p in pij.get((i,j),[]):
                try:
                    r=np.zeros((ne,))
                    for e in p: r[em[e]]=1
                    pa.append(r);vm.append(True)
                except KeyError: pa.append(np.zeros((ne,)));vm.append(False)
    pte=csr_matrix(np.stack(pa)); vmask=np.array(vm)
    ctp=lil_matrix((NC,pte.shape[0]))
    cid,pid=0,0
    for i in range(nn):
        for j in range(nn):
            if i==j: continue
            for _ in pij.get((i,j),[]):
                if vmask[pid]: ctp[cid,pid]=1
                pid+=1
            cid+=1
    ctpc=ctp.tocoo(); ptec=pte.tocoo()
    C=torch.sparse_coo_tensor(np.vstack((ctpc.row,ctpc.col)),torch.DoubleTensor(ctpc.data),torch.Size(ctpc.shape))
    P=torch.sparse_coo_tensor(np.vstack((ptec.row,ptec.col)),torch.DoubleTensor(ptec.data),torch.Size(ptec.shape))
    return C,P,torch.tensor(cap).unsqueeze(1).double(),torch.BoolTensor(vmask)

print("Building 50 topologies...")
topos={s:build(s) for s in tqdm(TRAIN_SLICES)}

from src.figret_net import FigretNetWork
model = FigretNetWork(HIST_LEN*NC, len(ps), 3).double()
opt = torch.optim.Adam(model.parameters())

print(f"\nTraining {EPOCHS} epochs (alpha={ALPHA}, beta={BETA})...")
for epoch in range(EPOCHS):
    model.train()
    vals=[]
    idxs=np.random.permutation(len(X))
    with tqdm(total=len(idxs), desc=f"E{epoch+1}/{EPOCHS}") as pbar:
        for idx in idxs:
            ts=np.random.choice(TRAIN_SLICES)
            C,P,cap,vmask=topos[ts]
            Xb=torch.DoubleTensor(X[idx]).unsqueeze(0)
            Yb=torch.DoubleTensor(Y[idx]).unsqueeze(0)
            yp=model(Xb)

            yp_=yp[0]+1e-16; yt_=Yb[0,:NC]; opt_val=Yb[0,-1].item()
            pw=yp_.unsqueeze(1)
            pw[~vmask]=1e-16
            tw=C.matmul(pw); split=pw.mul(C.t().matmul(1.0/tw))
            sc=split.clone()
            flow=P.t().matmul(C.t().matmul(yt_.unsqueeze(1)).mul(split))
            mc=torch.max(flow.divide(cap).flatten())

            sf=split.squeeze()
            sens=torch.mean(sf)
            surv=torch.mean(sc.squeeze()*(1.0-ps))

            loss=mc/mc.item() + ALPHA*sens/(sens.item()+1e-16) + BETA*surv/(surv.item()+1e-16)
            loss_val=mc.item()/opt_val if opt_val!=0 else 1.0

            opt.zero_grad(); loss.backward(); opt.step()
            vals.append(loss_val)
            pbar.update(1); pbar.set_postfix(mlu=loss_val)

    avg=sum(vals)/len(vals)
    print(f"  Epoch {epoch+1}: avg MLU = {avg:.4f}")

mp=MODEL_DIR/"leo_phase2_final.pt"
torch.save(model, mp)
print(f"\nSaved {mp}")
