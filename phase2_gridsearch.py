"""
Phase 2 grid search: find optimal alpha and beta for path survival penalty.
Tests 9 combinations, 5 epochs each, compares final MLU.
"""
import subprocess, sys, json, shutil
from pathlib import Path

ALPHAS = [0.01, 0.03, 0.05]
BETAS  = [0.01, 0.05, 0.1]
EPOCHS = 3

TOPO_DIR = Path("Data/leo/topologies")
TRAIN_TOPO = Path("Data/leo/leo.json")
RESULTS = []

print("Phase 2 Grid Search: alpha x beta")
print(f"  {len(ALPHAS)} x {len(BETAS)} = {len(ALPHAS)*len(BETAS)} combinations @ {EPOCHS} epochs each\n")

# Use slice 0 topology for all
shutil.copy(TOPO_DIR / "step_0000.json", TRAIN_TOPO)

for alpha in ALPHAS:
    for beta in BETAS:
        label = f"a{alpha}_b{beta}"
        print(f"[{label}] Training...", end=" ", flush=True)

        result = subprocess.run([
            sys.executable, "figret.py",
            "--topo_name", "leo",
            "--epochs", str(EPOCHS),
            "--hist_len", "12",
            "--alpha", str(alpha),
            "--beta", str(beta),
            "--opt_name", label
        ], capture_output=True, text=True, cwd=".")

        if result.returncode != 0:
            print(f"ERROR: {result.stderr[-200:]}")
            continue

        # Extract final loss_val from training output
        # The model saves as Model/leo_a0.01_b0.01.pt etc.
        # Run a quick test to get MLU
        test_result = subprocess.run([
            sys.executable, "figret.py",
            "--topo_name", "leo",
            "--mode", "test",
            "--hist_len", "12",
            "--alpha", str(alpha),
            "--beta", str(beta),
            "--opt_name", label
        ], capture_output=True, text=True, cwd=".")

        # Read test results
        result_file = Path(f"Result/leo/Figret/result.txt")
        if result_file.exists():
            with open(result_file) as f:
                lines = [float(l.strip()) for l in f if l.strip()]
            mlu = sum(lines) / len(lines) if lines else 999
            print(f"MLU={mlu:.4f}")
            RESULTS.append((alpha, beta, mlu, lines[-1] if lines else 0))
        else:
            print("no test results")
            RESULTS.append((alpha, beta, 999, 0))

# Summary
print(f"\n{'='*55}")
print(f"{'Alpha':<8} {'Beta':<8} {'Mean MLU':<12} {'Final MLU'}")
print(f"{'-'*55}")
RESULTS.sort(key=lambda x: x[2])
for a, b, mean_mlu, final_mlu in RESULTS:
    marker = " ← BEST" if (a, b) == (RESULTS[0][0], RESULTS[0][1]) else ""
    print(f"{a:<8} {b:<8} {mean_mlu:<12.4f} {final_mlu:<.4f}{marker}")

print(f"\nBest: alpha={RESULTS[0][0]}, beta={RESULTS[0][1]}, MLU={RESULTS[0][2]:.4f}")
