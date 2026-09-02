"""Deploy 2-GPU Kaggle Pilot Runners (All 84 Runs at Once).

Splits the 84-run manifest into exactly 2 halves (42 runs each) and deploys them
to Kaggle's 2 concurrent GPU slots simultaneously.
Runs completely in the cloud without needing local PC to remain powered on.
"""

import os
import sys
import json
import subprocess

USERNAME = "vaibhavchavanpatil"
BASE_DIR = os.path.abspath(".")
KERNELS_DIR = os.path.join(BASE_DIR, "kaggle_kernels")


def setup_and_push_half(half_idx: int):
    half_dir = os.path.join(KERNELS_DIR, f"half_{half_idx}")
    os.makedirs(half_dir, exist_ok=True)

    kernel_id = f"{USERNAME}/label-noise-pilot-half-{half_idx}"
    title = f"Label Noise Pilot Half {half_idx} (Runs {half_idx*42} to {(half_idx+1)*42-1})"

    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": "half_runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }

    with open(os.path.join(half_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    script_content = f"""# Autonomous 42-run Cloud Execution for Label Noise Pilot
import os
import sys
import subprocess
import tarfile
import shutil

HALF_INDEX = {half_idx}
NUM_HALVES = 2

print(f"=== STARTING KAGGLE PILOT HALF {{HALF_INDEX}}/{{NUM_HALVES}} (42 RUNS) ===")

# 1. Clone repository
repo_dir = "/kaggle/working/repo"
if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)

subprocess.run(["git", "clone", "https://github.com/VAIBHAV7848/label-noise-generalisation.git", repo_dir], check=True)
os.chdir(repo_dir)

# 2. Download CIFAR-10
subprocess.run([sys.executable, "scripts/download_cifar10.py"], check=True)

# 3. Output directory for this half
out_dir = f"/kaggle/working/results_half_{{HALF_INDEX}}"
os.makedirs(out_dir, exist_ok=True)

# 4. Run pilot 42 runs
cmd = [
    sys.executable,
    "-m",
    "src.training.run_pilot",
    "--batch-index", str(HALF_INDEX),
    "--num-batches", str(NUM_HALVES),
    "--data-dir", "data",
    "--output-dir", out_dir,
    "--device", "cuda"
]
print("Executing command:", " ".join(cmd))
subprocess.run(cmd, check=True)

# 5. Archive results to root output directory
archive_path = f"/kaggle/working/pilot_results_half_{{HALF_INDEX}}.tar.gz"
with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(out_dir, arcname="pilot_results")

# Clean up working files to minimize archive footprint
shutil.rmtree(repo_dir, ignore_errors=True)
shutil.rmtree(out_dir, ignore_errors=True)

print(f"=== HALF {{HALF_INDEX}} COMPLETED ALL 42 RUNS. ARCHIVE READY AT {{archive_path}} ===")
"""

    with open(os.path.join(half_dir, "half_runner.py"), "w") as f:
        f.write(script_content)

    print(f"Pushing Kernel {kernel_id} to Kaggle GPU...")
    res = subprocess.run(["python3", "-m", "kaggle", "kernels", "push", "-p", half_dir], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"  -> Successfully pushed {kernel_id}: {res.stdout.strip()}")
        return kernel_id
    else:
        print(f"  -> Error pushing {kernel_id}: {res.stderr.strip() or res.stdout.strip()}")
        return None


def main():
    print("=" * 70)
    print("DEPLOYING ALL 84 RUNS AT ONCE ACROSS 2 KAGGLE GPU KERNELS")
    print("=" * 70)

    k0 = setup_and_push_half(0)
    k1 = setup_and_push_half(1)

    print("\n" + "=" * 70)
    print("DEPLOYMENT COMPLETE — BOTH KERNELS RUNNING IN PARALLEL ON KAGGLE")
    print("=" * 70)
    print(f"Kernel 0 (Runs 0 to 41, 42 runs)  : https://www.kaggle.com/code/{k0}")
    print(f"Kernel 1 (Runs 42 to 83, 42 runs) : https://www.kaggle.com/code/{k1}")
    print("=" * 70)


if __name__ == "__main__":
    main()
