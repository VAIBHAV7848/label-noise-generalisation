"""Kaggle GPU Pilot Orchestrator for 84-Run Benchmark Grid.

Manages 4 parallel Kaggle GPU kernels to execute the 84-run manifest,
monitors execution progress, downloads result archives, and validates results.
"""

import os
import sys
import json
import time
import shutil
import tarfile
import subprocess

USERNAME = "vaibhavchavanpatil"
NUM_BATCHES = 4
BASE_DIR = os.path.abspath(".")
KERNELS_DIR = os.path.join(BASE_DIR, "kaggle_kernels")
OUTPUT_DIR = os.path.join(BASE_DIR, "05_RESULTS", "pilot")


def create_batch_kernel_files(batch_idx: int):
    """Create directory, python script, and metadata for a batch kernel."""
    batch_dir = os.path.join(KERNELS_DIR, f"batch_{batch_idx}")
    os.makedirs(batch_dir, exist_ok=True)

    kernel_id = f"{USERNAME}/label-noise-pilot-batch-{batch_idx}"
    title = f"Label Noise Pilot Batch {batch_idx}"

    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": "batch_runner.py",
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

    with open(os.path.join(batch_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    script_content = f"""# Auto-generated batch runner for Kaggle Pilot Execution
import os
import sys
import subprocess
import tarfile
import shutil

BATCH_INDEX = {batch_idx}
NUM_BATCHES = {NUM_BATCHES}

print(f"=== STARTING KAGGLE PILOT BATCH {{BATCH_INDEX}}/{{NUM_BATCHES}} ===")

# 1. Clone repository
repo_dir = "/kaggle/working/repo"
if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)

subprocess.run(["git", "clone", "https://github.com/VAIBHAV7848/label-noise-generalisation.git", repo_dir], check=True)
os.chdir(repo_dir)

# 2. Download CIFAR-10
subprocess.run([sys.executable, "scripts/download_cifar10.py"], check=True)

# 3. Output directory for this batch
out_dir = f"/kaggle/working/results_batch_{{BATCH_INDEX}}"
os.makedirs(out_dir, exist_ok=True)

# 4. Run pilot batch
cmd = [
    sys.executable,
    "-m",
    "src.training.run_pilot",
    "--batch-index", str(BATCH_INDEX),
    "--num-batches", str(NUM_BATCHES),
    "--data-dir", "data",
    "--output-dir", out_dir,
    "--device", "cuda"
]
print("Executing command:", " ".join(cmd))
res = subprocess.run(cmd)

# 5. Archive results to root output directory
archive_path = f"/kaggle/working/pilot_results_batch_{{BATCH_INDEX}}.tar.gz"
with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(out_dir, arcname="pilot_results")

# Clean up working files to minimize archive footprint
shutil.rmtree(repo_dir, ignore_errors=True)
shutil.rmtree(out_dir, ignore_errors=True)

print(f"=== BATCH {{BATCH_INDEX}} FINISHED. ARCHIVE SAVED AT {{archive_path}} ===")
"""

    with open(os.path.join(batch_dir, "batch_runner.py"), "w") as f:
        f.write(script_content)

    return batch_dir, kernel_id


def push_all_kernels():
    """Push all 4 batch kernels to Kaggle."""
    kernel_ids = []
    for b in range(NUM_BATCHES):
        b_dir, k_id = create_batch_kernel_files(b)
        print(f"Pushing Kernel for Batch {b} ({k_id})...")
        res = subprocess.run(["python3", "-m", "kaggle", "kernels", "push", "-p", b_dir], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error pushing batch {b}: {res.stderr}")
        else:
            print(f"  -> {res.stdout.strip()}")
        kernel_ids.append(k_id)
    return kernel_ids


def get_kernel_status(kernel_id: str) -> str:
    """Get the current execution status of a Kaggle kernel."""
    res = subprocess.run(["python3", "-m", "kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
    if res.returncode == 0:
        out = res.stdout.strip()
        if "COMPLETE" in out:
            return "COMPLETE"
        elif "RUNNING" in out or "QUEUED" in out:
            return "RUNNING"
        elif "ERROR" in out or "FAILED" in out:
            return "ERROR"
        else:
            return out
    return "UNKNOWN"


def download_and_extract_results(kernel_id: str, batch_idx: int):
    """Download output archive and extract JSON results and checkpoints."""
    download_dir = os.path.join(KERNELS_DIR, f"output_batch_{batch_idx}")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoints_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"Downloading outputs for {kernel_id}...")
    subprocess.run(["python3", "-m", "kaggle", "kernels", "output", kernel_id, "-p", download_dir], check=True)

    # Search for tarball
    archive_path = os.path.join(download_dir, f"pilot_results_batch_{batch_idx}.tar.gz")
    if os.path.exists(archive_path):
        print(f"Extracting {archive_path} into {OUTPUT_DIR}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.startswith("pilot_results/"):
                    rel_path = member.name[len("pilot_results/"):]
                    if not rel_path:
                        continue
                    if member.name.endswith(".json"):
                        # Extract JSON directly into 05_RESULTS/pilot/
                        f = tar.extractfile(member)
                        if f:
                            target_file = os.path.join(OUTPUT_DIR, os.path.basename(member.name))
                            with open(target_file, "wb") as out_f:
                                out_f.write(f.read())
                    elif member.name.endswith(".pt"):
                        # Extract checkpoint into 05_RESULTS/pilot/checkpoints/
                        f = tar.extractfile(member)
                        if f:
                            target_file = os.path.join(checkpoints_dir, os.path.basename(member.name))
                            with open(target_file, "wb") as out_f:
                                out_f.write(f.read())
        print(f"Batch {batch_idx} results synchronized successfully.")
        return True
    else:
        print(f"Archive {archive_path} not found in {download_dir}.")
        return False


def main():
    print("=" * 70)
    print("KAGGLE GPU PILOT ORCHESTRATOR — 84-RUN BENCHMARK")
    print("=" * 70)

    kernel_ids = push_all_kernels()
    print("\nAll 4 batch kernels pushed. Entering monitoring loop...")

    completed = set()
    start_time = time.time()

    while len(completed) < NUM_BATCHES:
        time.sleep(30)
        elapsed = time.time() - start_time
        print(f"\n--- Monitoring Update [T + {elapsed/60:.1f} min] ---")

        for b, k_id in enumerate(kernel_ids):
            if b in completed:
                print(f"Batch {b} ({k_id}): COMPLETED & SYNCHRONIZED")
                continue

            status = get_kernel_status(k_id)
            print(f"Batch {b} ({k_id}): {status}")

            if status == "COMPLETE":
                print(f"Batch {b} finished execution on Kaggle! Downloading results...")
                success = download_and_extract_results(k_id, b)
                if success:
                    completed.add(b)
            elif status == "ERROR":
                print(f"WARNING: Batch {b} encountered an error. Re-pushing to retry...")
                subprocess.run(["python3", "-m", "kaggle", "kernels", "push", "-p", os.path.join(KERNELS_DIR, f"batch_{b}")])

    print("\n" + "=" * 70)
    print("ALL 4 BATCHES COMPLETED. RUNNING FINAL MANIFEST VALIDATION...")
    print("=" * 70)

    subprocess.run([
        sys.executable,
        "-m",
        "src.training.run_pilot",
        "--validate-manifest",
        "--output-dir", OUTPUT_DIR,
    ])


if __name__ == "__main__":
    main()
