"""
Phase 0b: Generate candidate paths (tunnels.txt) for FIGRET.

Uses Dijkstra + bounded-depth search for speed.
K=3 paths per S-D pair, 9120 pairs total.
"""
import json
import networkx as nx
from pathlib import Path

TOPO_FILE = Path("Data/leo/leo.json")
OUT_FILE = Path("Data/leo/tunnels.txt")
K_PATHS = 3
HOP_MARGIN = 2  # search for paths up to (shortest_path_hops + this)


def load_graph(json_path):
    with open(json_path) as f:
        data = json.load(f)
    G = nx.DiGraph()
    for node in data['nodes']:
        G.add_node(node['id'])
    for link in data['links']:
        G.add_edge(link['source'], link['target'])
    return G


print("Phase 0b: Generating candidate paths...")
print(f"  Loading graph from {TOPO_FILE}")

G = load_graph(TOPO_FILE)
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print(f"  Graph: {num_nodes} nodes, {num_edges} edges")

total_pairs = num_nodes * (num_nodes - 1)
print(f"  S-D pairs: {total_pairs}")

lines = []
done = 0
failed = 0

for src in sorted(G.nodes()):
    for dst in sorted(G.nodes()):
        if src == dst:
            continue
        try:
            d = nx.shortest_path_length(G, source=src, target=dst, weight=None)
            cutoff = d + HOP_MARGIN
            paths = list(nx.all_simple_paths(G, source=src, target=dst, cutoff=cutoff))
            # Sort by length, take K shortest
            paths.sort(key=len)
            k = min(K_PATHS, len(paths))
            if k < K_PATHS:
                failed += 1
            path_strs = ['-'.join(str(n) for n in p) for p in paths[:k]]
            lines.append(f"{src} {dst}:{','.join(path_strs)}")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            failed += 1
            lines.append(f"{src} {dst}:")

        done += 1
        if done % 2000 == 0:
            print(f"  [{done:5d}/{total_pairs}] {done*100/total_pairs:.0f}%")

with open(OUT_FILE, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"\n=== Summary ===")
print(f"  Output: {OUT_FILE} ({len(lines)} lines)")
print(f"  Pairs with <{K_PATHS} paths: {failed}/{total_pairs}")
print("Phase 0b complete.")
