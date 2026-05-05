"""
Full comparison across ALL 189 slices: baselines + old FIGRET + new FIGRET.
"""
import json, csv, torch, numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict
from networkx.readwrite import json_graph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOPO_DIR = Path("Data/leo/topologies")
TUNNELS_FILE = Path("Data/leo/tunnels_full.txt")
BASELINE_CSV = Path("Result/leo/baselines.csv")
RESULT_DIR = Path("Result/leo")
HIST_LEN, NN, NC = 12, 96, 96*95

np.random.seed(42)
with open("Data/leo/test/0.hist") as f: lines = f.readlines()
start = np.random.randint(0, len(lines) - HIST_LEN)
tm_input = np.array([[float(x) for x in lines[start+i].strip().split()] for i in range(HIST_LEN)])

def build_env(topo):
    with open(topo) as f: data=json.load(f)
    G=json_graph.node_link_graph(data)
    nn,ne=G.number_of_nodes(),G.number_of_edges()
    adj=np.zeros((nn,nn),dtype=bool)
    for s in range(nn):
        for d in range(nn):
            if s!=d and d in G[s]: adj[s,d]=True
    em,cap={},[]
    for i in range(nn):
        for j in range(nn):
            if adj[i,j]: em[(i,j)]=len(em); cap.append(G[i][j]['capacity']/1e9)
    pij=defaultdict(list)
    with open(TUNNELS_FILE) as f:
        ld={l.split(':')[0]:l.strip() for l in sorted(f.readlines()) if ':' in l}
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
                    r=np.zeros((ne,)); [r.__setitem__(em[e],1) for e in p]
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

def mlu(model, env, tm_12):
    C,P,cap,vmask=env
    dm=np.eye(NN,dtype=bool).flatten()
    X=torch.DoubleTensor((tm_12[:,~dm]/1e9).flatten()).unsqueeze(0)
    with torch.no_grad():
        y=model(X)+1e-16; pw=y.squeeze(0).unsqueeze(1)
        pw[~vmask]=1e-16
        tw=C.matmul(pw); split=pw.mul(C.t().matmul(1.0/tw))
        yt=torch.DoubleTensor((tm_12[-1][~dm]/1e9)).unsqueeze(0)
        flow=P.t().matmul(C.t().matmul(yt.t()).mul(split))
        return max(0,torch.max(flow.divide(cap).flatten()).item())

print("Loading models...")
old_model = torch.load("Model/leo_step_0000.pt", map_location='cpu', weights_only=False).eval()
new_model = torch.load("Model/leo.pt", map_location='cpu', weights_only=False).eval()

print("Testing all 189 slices...")
old_mlu, new_mlu = {}, {}
for ts in range(0, 189):
    env = build_env(str(TOPO_DIR / f"step_{ts:04d}.json"))
    old_mlu[ts] = mlu(old_model, env, tm_input)
    new_mlu[ts] = mlu(new_model, env, tm_input)
    if ts % 25 == 0: print(f"  {ts:04d}: old={old_mlu[ts]:.4f} new={new_mlu[ts]:.4f}")

with open(BASELINE_CSV) as f:
    base = {int(r['slice']): {'e':float(r['ecmp']),'s':float(r['spf']),'d':float(r['dist_weight'])} for r in csv.DictReader(f)}

# Plot
fig, ax = plt.subplots(figsize=(20, 8))
slices = list(range(0, 189))

ax.plot(slices, [base[s]['e'] for s in slices], 'gray', linewidth=1.2, alpha=0.5, label='ECMP')
ax.plot(slices, [base[s]['s'] for s in slices], 'brown', linewidth=1.2, alpha=0.5, label='SPF')
ax.plot(slices, [base[s]['d'] for s in slices], 'orange', linewidth=1.2, alpha=0.5, label='Dist-Weighted')
ax.plot(slices, [old_mlu[s] for s in slices], 'blue', linewidth=2.5, alpha=0.85, label='FIGRET Phase 1 (trained on slice 0)')
ax.plot(slices, [new_mlu[s] for s in slices], 'green', linewidth=2.5, alpha=0.85, label='FIGRET Phase 2 (full graph + β=0.1)')

# Training set region (0-49)
ax.axvspan(0, 49, alpha=0.06, color='gray', label='Training set (slices 0-49)')

# Train/test boundary
ax.axvline(x=49.5, color='black', linestyle='-', linewidth=1.5, alpha=0.4)
ax.text(51, 0.5, 'Test set →', fontsize=10, color='black', alpha=0.6)

ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.4)
ax.set_xlabel('Time Slice', fontsize=14)
ax.set_ylabel('MLU', fontsize=14)
ax.set_title('FIGRET Phase 1 vs Phase 2 vs Baselines (All 189 Time Slices)', fontsize=16, fontweight='bold')
ax.legend(fontsize=10, loc='upper right', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 188)

fig.tight_layout()
out = RESULT_DIR / "phase2_full_comparison.png"
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")
