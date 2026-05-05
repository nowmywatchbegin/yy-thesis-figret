"""
Quick test: Phase 2 model (5 epoch) vs old FIGRET vs baselines, slices 50-188 ONLY.
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

def compute_mlu(model, env, tm_12):
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

# Load models
print("Loading models...")
model_old = torch.load("Model/leo_step_0000.pt", map_location='cpu', weights_only=False).eval()
model_new = torch.load("Model/leo.pt", map_location='cpu', weights_only=False).eval()

# Test on slices 50-188 only (pure test set)
TEST_SLICES = list(range(50, 189))
print(f"Testing slices {TEST_SLICES[0]}-{TEST_SLICES[-1]}...")
old_mlu, new_mlu = {}, {}
for ts in TEST_SLICES:
    env = build_env(str(TOPO_DIR / f"step_{ts:04d}.json"))
    old_mlu[ts] = compute_mlu(model_old, env, tm_input)
    new_mlu[ts] = compute_mlu(model_new, env, tm_input)
    if ts % 25 == 0: print(f"  slice {ts:04d}: old={old_mlu[ts]:.4f} new={new_mlu[ts]:.4f}")

with open(BASELINE_CSV) as f:
    base = {int(r['slice']): {'e':float(r['ecmp']),'s':float(r['spf']),'d':float(r['dist_weight'])} for r in csv.DictReader(f)}

# Failures
old_fails, new_fails = [], []
for ts in TEST_SLICES:
    b=base[ts]; worst_b=min(b['e'],b['s'],b['d'])
    if old_mlu[ts]>worst_b: old_fails.append(ts)
    if new_mlu[ts]>worst_b: new_fails.append(ts)

old_first = old_fails[0] if old_fails else None
new_first = new_fails[0] if new_fails else None
print(f"\nOld FIGRET: first fail slice {old_first}, {len(old_fails)}/{len(TEST_SLICES)} fail")
print(f"New FIGRET: first fail slice {new_first}, {len(new_fails)}/{len(TEST_SLICES)} fail")

if new_first and old_first and new_first > old_first:
    improvement = new_first - old_first
    print(f"IMPROVEMENT: failure delayed by {improvement} slices!")
elif new_first and old_first and new_first == old_first:
    print("Same failure point, checking MLU averages...")
    old_avg = np.mean(list(old_mlu.values()))
    new_avg = np.mean(list(new_mlu.values()))
    print(f"  Old avg MLU: {old_avg:.4f}, New avg MLU: {new_avg:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(18, 7))
slices=TEST_SLICES
ax.plot(slices,[base[s]['e'] for s in slices],'gray',linewidth=1,alpha=0.4,label='ECMP')
ax.plot(slices,[base[s]['s'] for s in slices],'brown',linewidth=1,alpha=0.4,label='SPF')
ax.plot(slices,[base[s]['d'] for s in slices],'orange',linewidth=1,alpha=0.4,label='Dist-Weighted')
ax.plot(slices,[old_mlu[s] for s in slices],'blue',linewidth=2,label='FIGRET Phase 1')
ax.plot(slices,[new_mlu[s] for s in slices],'green',linewidth=2.5,label='FIGRET Phase 2 (β=0.1)')

if old_first:
    ax.axvline(x=old_first,color='blue',linestyle='--',alpha=0.5)
    ax.text(old_first,1.55,f'Old fails@{old_first}',color='blue',fontsize=9)
if new_first:
    ax.axvline(x=new_first,color='green',linestyle='--',alpha=0.5)
    ax.text(new_first,1.50,f'New fails@{new_first}',color='green',fontsize=9)

ax.axhline(y=1.0,color='red',linestyle=':',alpha=0.4)
ax.set_xlabel('Time Slice',fontsize=13)
ax.set_ylabel('MLU',fontsize=13)
ax.set_title(f'Phase 2: FIGRET + β Survival Penalty (test on slices 50-188)',fontsize=14,fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True,alpha=0.3)

fig.tight_layout()
out=RESULT_DIR/"phase2_quick_result.png"
fig.savefig(out,dpi=150)
print(f"\nPlot saved to {out}")
