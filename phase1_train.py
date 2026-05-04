"""
Phase 1a: Train FIGRET on time slices 0-4.
Each model is saved as Model/leo_step_XXXX.pt
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path("Model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TOPO_DIR = Path("Data/leo/topologies")
TRAIN_TOPO = Path("Data/leo/leo.json")
FULL_TOPO = Path("Data/leo/full_leo.json")
TUNNELS_SRC = Path("Data/leo/tunnels_full.txt")
TUNNELS_DST = Path("Data/leo/tunnels.txt")

EPOCHS = 5  # model 0000 already did 30; for the rest, 5 is enough (link survival is the focus)
TRAIN_SLICES = [1, 2, 3, 4]  # slice 0 already trained (30 epochs)

# Use tunnels_full.txt as the active tunnels
shutil.copy(TUNNELS_SRC, TUNNELS_DST)
print(f"Set active tunnels: {TUNNELS_DST} (from full graph)")

for i, step in enumerate(TRAIN_SLICES):
    print(f"\n{'='*60}")
    print(f"Training model {i+1}/5: time slice {step:04d}")
    print(f"{'='*60}")

    # Copy this slice's topology as the training topology
    src_topo = TOPO_DIR / f"step_{step:04d}.json"
    shutil.copy(src_topo, TRAIN_TOPO)

    # Run FIGRET training
    cmd = [
        sys.executable, "figret.py",
        "--topo_name", "leo",
        "--epochs", str(EPOCHS),
        "--hist_len", "12",
        "--mode", "train"
    ]
    result = subprocess.run(cmd, cwd=".")
    if result.returncode != 0:
        print(f"ERROR: Training failed for slice {step:04d}")
        sys.exit(1)

    # Save model with slice-specific name
    trained_model = MODEL_DIR / "leo.pt"
    saved_model = MODEL_DIR / f"leo_step_{step:04d}.pt"
    shutil.move(str(trained_model), str(saved_model))
    print(f"Model saved: {saved_model}")

# Restore tunnels.txt to full version
shutil.copy(TUNNELS_SRC, TUNNELS_DST)

print(f"\n{'='*60}")
print("All 5 models trained!")
for step in TRAIN_SLICES:
    print(f"  Model/leo_step_{step:04d}.pt")
print(f"{'='*60}")
