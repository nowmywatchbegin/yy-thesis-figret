"""
Phase 0a: Parse LEO txt files, compute ISL distances, filter by threshold, generate topology JSON.

Input:  Data/leo_new/step_*.txt (189 files)
Output: Data/leo/leo.json               (time slice 0 topology, for FIGRET training)
        Data/leo/topologies/step_*.json  (all 189 filtered topologies)
        Data/leo/link_survival.json      (each link's survival rate across all slices)
"""
import json
import math
from pathlib import Path

DATA_DIR = Path("Data/leo_raw")
OUT_DIR = Path("Data/leo")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOPOLOGY_DIR = OUT_DIR / "topologies"
TOPOLOGY_DIR.mkdir(parents=True, exist_ok=True)

DISTANCE_THRESHOLD_KM = 4300  # links >= this are considered broken
CAPACITY_GBPS = 1  # ISL link bandwidth in Gbps (early optical ISL, academic reference)


def latlon_to_xyz(lat, lon, alt):
    """Convert geodetic coordinates (deg, deg, km) to 3D Cartesian (km)."""
    R = 6371  # Earth radius in km
    phi = math.radians(lat)
    lam = math.radians(lon)
    r = R + alt
    x = r * math.cos(phi) * math.cos(lam)
    y = r * math.cos(phi) * math.sin(lam)
    z = r * math.sin(phi)
    return x, y, z


def distance_3d(lat1, lon1, alt1, lat2, lon2, alt2):
    """Compute straight-line 3D distance between two satellites in km."""
    x1, y1, z1 = latlon_to_xyz(lat1, lon1, alt1)
    x2, y2, z2 = latlon_to_xyz(lat2, lon2, alt2)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def parse_leo_file(filepath):
    """Parse one LEO step file. Returns (nodes_dict, links_list)."""
    with open(filepath) as f:
        content = f.read()

    nodes = {}  # id -> {name, lat, lon, alt}
    links = []  # [(src_id, dst_id)]

    section = None
    for line in content.strip().split('\n'):
        line = line.strip()
        if line.startswith('['):
            section = line
            continue
        if not line or section is None:
            continue

        if section == '[NODES]':
            parts = line.split(', ')
            if len(parts) >= 5 and parts[0].isdigit():
                nid = int(parts[0])
                nodes[nid] = {
                    'name': parts[1],
                    'lat': float(parts[2]),
                    'lon': float(parts[3]),
                    'alt': float(parts[4])
                }
        elif section == '[LINKS]':
            parts = line.split(', ')
            if len(parts) >= 3 and parts[0] == 'ISL':
                src, dst = int(parts[1]), int(parts[2])
                links.append((src, dst))

    return nodes, links


def links_with_distances(nodes, links):
    """Compute 3D distance for each link. Returns [(src, dst, dist_km)]."""
    result = []
    for src, dst in links:
        s, d = nodes[src], nodes[dst]
        dist = distance_3d(s['lat'], s['lon'], s['alt'],
                          d['lat'], d['lon'], d['alt'])
        result.append((src, dst, dist))
    return result


def build_topology_json(nodes, links_with_dist, threshold):
    """Build a NetworkX node-link JSON dict, filtering links by threshold."""
    graph = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": nid} for nid in sorted(nodes.keys())],
        "links": []
    }
    for src, dst, dist in links_with_dist:
        if dist < threshold:
            graph["links"].append({
                "capacity": CAPACITY_GBPS * 1e9,  # FIGRET uses bps, normalize_size divides by 1e9
                "source": src,
                "target": dst,
                "distance_km": round(dist, 2)
            })
    return graph


# ============================================================
# MAIN
# ============================================================
print("Phase 0a: Generating topology JSONs...")
print(f"  Distance threshold: {DISTANCE_THRESHOLD_KM} km")
print(f"  ISL capacity: {CAPACITY_GBPS} Gbps")

files = sorted(DATA_DIR.glob("step_*.txt"))
print(f"  Input files: {len(files)}")

all_slice_data = []
link_survival_counter = {}  # (src, dst) -> count of slices where it survives

for step_idx, fpath in enumerate(files):
    nodes, links = parse_leo_file(fpath)
    lwd = links_with_distances(nodes, links)

    # Build and save topology JSON
    topo = build_topology_json(nodes, lwd, DISTANCE_THRESHOLD_KM)
    out_name = f"step_{step_idx:04d}.json"
    with open(TOPOLOGY_DIR / out_name, 'w') as f:
        json.dump(topo, f, indent=2)

    # Track link survival
    surviving = set()
    for src, dst, dist in lwd:
        if dist < DISTANCE_THRESHOLD_KM:
            surviving.add((src, dst))
            link_survival_counter[(src, dst)] = link_survival_counter.get((src, dst), 0) + 1

    all_slice_data.append({
        'step': step_idx,
        'num_links': len(topo['links']),
        'num_nodes': len(topo['nodes'])
    })

    if step_idx % 20 == 0:
        print(f"  [{step_idx:3d}/{len(files)}] {fpath.name} -> {len(topo['links'])} links survive")

# Save time slice 0 as the main training topology (FIGRET expects <topo_name>.json)
with open(TOPOLOGY_DIR / "step_0000.json") as f:
    step0_topo = json.load(f)
with open(OUT_DIR / "leo.json", 'w') as f:
    json.dump(step0_topo, f, indent=2)

# Save link survival statistics
survival_rates = {}
for (src, dst), count in link_survival_counter.items():
    survival_rates[f"{src},{dst}"] = {
        "src": src, "dst": dst,
        "survival_count": count,
        "survival_rate": round(count / len(files), 4)
    }
with open(OUT_DIR / "link_survival.json", 'w') as f:
    json.dump({
        "total_slices": len(files),
        "threshold_km": DISTANCE_THRESHOLD_KM,
        "links": list(survival_rates.values())
    }, f, indent=2)

# Summary
num_links_per_slice = [d['num_links'] for d in all_slice_data]
print(f"\n=== Summary ===")
print(f"  Training topology: Data/leo/leo.json ({num_links_per_slice[0]} links, {all_slice_data[0]['num_nodes']} nodes)")
print(f"  All topologies:    Data/leo/topologies/step_XXXX.json")
print(f"  Link survival:     Data/leo/link_survival.json")
print(f"  Links per slice:   min={min(num_links_per_slice)} max={max(num_links_per_slice)} mean={sum(num_links_per_slice)/len(num_links_per_slice):.0f}")
varying = sum(1 for n in num_links_per_slice if n != num_links_per_slice[0])
print(f"  Slices differing from slice 0: {varying}/{len(files)}")
print("\nPhase 0a complete.")
