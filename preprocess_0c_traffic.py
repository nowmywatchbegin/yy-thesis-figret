"""
Phase 0c: Generate simulated traffic for LEO network using Facebook statistics.

1. Extract statistical features from Facebook_pod_a training data
2. Assign traffic profiles to LEO S-D pairs
3. Generate time-series traffic (.hist files)
4. Generate dummy .opt files (needed by FIGRET's data loader, not used in training loss)
"""
import numpy as np
from pathlib import Path

FACEBOOK_TRAIN_DIR = Path("Data/Facebook_pod_a/train")
OUT_DIR = Path("Data/leo")
TRAIN_OUT = OUT_DIR / "train"
TEST_OUT = OUT_DIR / "test"
TRAIN_OUT.mkdir(parents=True, exist_ok=True)
TEST_OUT.mkdir(parents=True, exist_ok=True)

NUM_NODES = 96
NUM_TIMESTEPS = 3000  # total time steps to generate
TRAIN_RATIO = 0.8     # 80% train, 20% test
HIST_LEN = 12         # FIGRET's history length (need extra steps for sliding window)
TRAFFIC_SCALE = 70  # scale Facebook bps to match 1 Gbps ISL capacity
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ============================================================
# Step 1: Extract Facebook statistical features
# ============================================================
print("Step 1: Extracting Facebook traffic statistics...")

all_tms = []
for fpath in sorted(FACEBOOK_TRAIN_DIR.glob("*.hist")):
    with open(fpath) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            all_tms.append(vals)

data = np.array(all_tms)  # shape: (N, 16) for 4-node topology
print(f"  Facebook samples: {len(data)}")
print(f"  Values per sample: {data.shape[1]}")

# Extract non-diagonal values (12 S-D pairs for 4 nodes)
diag_mask = np.eye(4, dtype=bool).flatten()
sd_data = data[:, ~diag_mask]  # shape: (N, 12)

# Per-pair statistics
means = sd_data.mean(axis=0)
stds = sd_data.std(axis=0)
cvs = stds / (means + 1e-8)

# Burst probability per pair (>2x mean)
burst_probs = np.array([(sd_data[:, i] > 2 * means[i]).mean() for i in range(12)])

print(f"\n  Facebook S-D pair statistics (scaled by {TRAFFIC_SCALE}x):")
for i in range(12):
    print(f"    pair {i:2d}: mean={means[i]*TRAFFIC_SCALE:8.0f}  std={stds[i]*TRAFFIC_SCALE:8.0f}  cv={cvs[i]:.2f}  burst_prob={burst_probs[i]:.3f}")

# ============================================================
# Step 2: Assign traffic profiles to LEO S-D pairs
# ============================================================
print(f"\nStep 2: Assigning traffic profiles to {NUM_NODES * (NUM_NODES - 1)} LEO S-D pairs...")

num_sd_pairs = NUM_NODES * (NUM_NODES - 1)

# Each LEO S-D pair randomly picks a Facebook profile
profile_idx = np.random.choice(len(means), size=num_sd_pairs)
leo_means = means[profile_idx] * TRAFFIC_SCALE
leo_stds = stds[profile_idx] * TRAFFIC_SCALE
leo_burst_probs = burst_probs[profile_idx]

print(f"  LEO means:    {leo_means.min():.0f} ~ {leo_means.max():.0f}")
print(f"  LEO stds:     {leo_stds.min():.0f} ~ {leo_stds.max():.0f}")
print(f"  LEO CVs:      {(leo_stds / (leo_means + 1e-8)).min():.2f} ~ {(leo_stds / (leo_means + 1e-8)).max():.2f}")

# ============================================================
# Step 3: Generate time-series traffic
# ============================================================
print(f"\nStep 3: Generating {NUM_TIMESTEPS} time steps of traffic...")

# Generate traffic matrix: (num_timesteps, num_sd_pairs)
tm = np.zeros((NUM_TIMESTEPS, num_sd_pairs))

for t in range(NUM_TIMESTEPS):
    # Base: normal distribution around mean
    noise = np.random.randn(num_sd_pairs) * leo_stds
    tm[t] = leo_means + noise

    # Burst: randomly trigger bursts
    burst_mask = np.random.random(num_sd_pairs) < leo_burst_probs
    burst_amount = np.random.exponential(leo_stds * 3, size=num_sd_pairs)
    tm[t] += burst_mask * burst_amount

    # Clip to non-negative
    tm[t] = np.maximum(tm[t], 0)

    if t % 500 == 0:
        print(f"  [{t:4d}/{NUM_TIMESTEPS}]")

print(f"\n  Traffic stats:")
print(f"    Mean: {tm.mean():.0f}")
print(f"    Std:  {tm.std():.0f}")
print(f"    Min:  {tm.min():.0f}")
print(f"    Max:  {tm.max():.0f}")
print(f"    Zero fraction: {(tm == 0).mean()*100:.1f}%")

# ============================================================
# Step 4: Write .hist and .opt files
# ============================================================
print("\nStep 4: Writing .hist and .opt files...")

# Expand non-diagonal values to full matrix (with zeros on diagonal)
def expand_to_full_matrix(sd_values, num_nodes):
    """Expand (num_sd_pairs,) to (num_nodes * num_nodes,) with diagonal=0."""
    full = np.zeros(num_nodes * num_nodes)
    non_diag_mask = ~np.eye(num_nodes, dtype=bool).flatten()
    full[non_diag_mask] = sd_values
    return full

# Convert all timesteps
full_tm = np.array([expand_to_full_matrix(tm[t], NUM_NODES) for t in range(NUM_TIMESTEPS)])

# Split into train/test
n_train = int(NUM_TIMESTEPS * TRAIN_RATIO)
train_data = full_tm[:n_train]
test_data = full_tm[n_train:]

print(f"  Train: {len(train_data)} timesteps")
print(f"  Test:  {len(test_data)} timesteps")

# Write .hist files (one line per timestep, space-separated floats)
with open(TRAIN_OUT / "0.hist", 'w') as f:
    for row in train_data:
        f.write(' '.join(f'{x:.6f}' for x in row) + '\n')

with open(TEST_OUT / "0.hist", 'w') as f:
    for row in test_data:
        f.write(' '.join(f'{x:.6f}' for x in row) + '\n')

# Write dummy .opt files (all 1.0 - not used in training loss, only for monitoring)
with open(TRAIN_OUT / "0.opt", 'w') as f:
    f.write('\n'.join('1.0' for _ in range(len(train_data))) + '\n')

with open(TEST_OUT / "0.opt", 'w') as f:
    f.write('\n'.join('1.0' for _ in range(len(test_data))) + '\n')

print(f"\nPhase 0c complete.")
print(f"  Train: {TRAIN_OUT}/0.hist, {TRAIN_OUT}/0.opt")
print(f"  Test:  {TEST_OUT}/0.hist, {TEST_OUT}/0.opt")
