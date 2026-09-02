"""Kaggle GPU Pilot Orchestrator for 84-Run Benchmark Grid.

Manages a 2-GPU concurrency queue across Kaggle, pushes batches,
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
MAX_CONCURRENT_GPU = 2
BASE_DIR = os.path.abspath(".")
KERNELS_DIR = os.path.join(BASE_DIR, "kaggle_kernels")
OUTPUT_DIR = os.path.join(BASE_DIR, "05_RESULTS", "pilot")
MANIFEST_PATH = os.path.join(BASE_DIR, "04_EXPERIMENTS", "pilot_run_manifest.json")


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
subprocess.run(cmd, check=True)

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


def push_kernel(batch_idx: int) -> bool:
    """Push a specific batch kernel to Kaggle."""
    b_dir, k_id = create_batch_kernel_files(batch_idx)
    res = subprocess.run(["python3", "-m", "kaggle", "kernels", "push", "-p", b_dir], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Pushed Batch {batch_idx} ({k_id}): {res.stdout.strip()}")
        return True
    else:
        print(f"Push failed for Batch {batch_idx}: {res.stderr.strip() or res.stdout.strip()}")
        return False


def get_kernel_status(kernel_id: str) -> str:
    """Get the current execution status of a Kaggle kernel."""
    for attempt in range(3):
        res = subprocess.run(["python3", "-m", "kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
        if res.returncode == 0:
            out = res.stdout.strip().upper()
            if "COMPLETE" in out:
                return "COMPLETE"
            elif "RUNNING" in out or "QUEUED" in out:
                return "RUNNING"
            elif "ERROR" in out or "FAILED" in out or "CANCELLED" in out:
                return "ERROR"
        time.sleep(2)
    return "UNKNOWN"


def download_and_extract_results(kernel_id: str, batch_idx: int) -> bool:
    """Download output archive and extract JSON results and checkpoints."""
    download_dir = os.path.join(KERNELS_DIR, f"output_batch_{batch_idx}")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoints_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"Downloading outputs for {kernel_id}...")
    subprocess.run(["python3", "-m", "kaggle", "kernels", "output", kernel_id, "-p", download_dir], check=True)

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
                        f = tar.extractfile(member)
                        if f:
                            target_file = os.path.join(OUTPUT_DIR, os.path.basename(member.name))
                            with open(target_file, "wb") as out_f:
                                out_f.write(f.read())
                    elif member.name.endswith(".pt"):
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


def count_completed_manifest_runs() -> tuple[int, int]:
    """Count valid completed runs in 05_RESULTS/pilot/ against the 84 manifest."""
    if not os.path.exists(MANIFEST_PATH):
        return 0, 84
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    runs = manifest["runs"]
    completed = 0
    for r in runs:
        res_file = os.path.join(OUTPUT_DIR, f"{r['experiment_id']}.json")
        if os.path.exists(res_file):
            try:
                with open(res_file, "r") as rf:
                    data = json.load(rf)
                if data.get("status") == "COMPLETED" and "metrics" in data:
                    completed += 1
            except Exception:
                pass
    return completed, len(runs)


def main():
    print("=" * 70)
    print("KAGGLE GPU PILOT QUEUE ORCHESTRATOR — 84-RUN BENCHMARK")
    print("=" * 70)

    # Initialize batch metadata
    for b in range(NUM_BATCHES):
        create_batch_kernel_files(b)

    # State tracking: PENDING, RUNNING, COMPLETED, ERROR
    batch_status = {b: "PENDING" for b in range(NUM_BATCHES)}
    kernel_ids = {b: f"{USERNAME}/label-noise-pilot-batch-{b}" for b in range(NUM_BATCHES)}

    # Check initially running kernels on Kaggle
    for b in range(NUM_BATCHES):
        k_status = get_kernel_status(kernel_ids[b])
        if k_status == "RUNNING":
            batch_status[b] = "RUNNING"
            print(f"Batch {b} is already RUNNING on Kaggle.")
        elif k_status == "COMPLETE":
            # Attempt to download if not already synchronized
            download_and_extract_results(kernel_ids[b], b)
            batch_status[b] = "COMPLETED"

    start_time = time.time()

    while True:
        completed_count = sum(1 for s in batch_status.values() if s == "COMPLETED")
        if completed_count == NUM_BATCHES:
            break

        # Poll running kernels
        for b in range(NUM_BATCHES):
            if batch_status[b] == "RUNNING":
                st = get_kernel_status(kernel_ids[b])
                if st == "COMPLETE":
                    print(f"\n[EVENT] Batch {b} completed execution on Kaggle! Downloading results...")
                    success = download_and_extract_results(kernel_ids[b], b)
                    if success:
                        batch_status[b] = "COMPLETED"
                    else:
                        print(f"Failed to synchronize results for Batch {b}. Will retry download...")
                elif st == "ERROR":
                    print(f"\n[EVENT] Batch {b} encountered an infrastructure error. Re-queueing...")
                    batch_status[b] = "PENDING"

        # Check active running kernels
        active_running = sum(1 for s in batch_status.values() if s == "RUNNING")

        # Launch pending kernels if below capacity
        for b in range(NUM_BATCHES):
            if batch_status[b] == "PENDING" and active_running < MAX_CONCURRENT_GPU:
                print(f"\nQueue slot available. Launching Batch {b} ({kernel_ids[b]})...")
                success = push_kernel(b)
                if success:
                    batch_status[b] = "RUNNING"
                    active_running += 1
                    time.sleep(5)

        # Monitoring Progress Report
        done_runs, total_runs = count_completed_manifest_runs()
        elapsed_sec = time.time() - start_time
        elapsed_min = elapsed_sec / 60.0

        if done_runs > 0:
            rate = done_runs / elapsed_sec
            rem_sec = (total_runs - done_runs) / rate
            rem_min = rem_sec / 60.0
            eta_str = f"{rem_min:.1f} min"
        else:
            eta_str = "calculating..."

        print(f"\n--- Progress: [{done_runs}/{total_runs} Runs Completed ({done_runs/total_runs*100:.1f}%)] | Active Batches: {[b for b, s in batch_status.items() if s == 'RUNNING']} | Elapsed: {elapsed_min:.1f}m | ETA: {eta_str} ---")
        for b in range(NUM_BATCHES):
            print(f"  Batch {b}: {batch_status[b]} ({kernel_ids[b]})")

        time.sleep(30)

    print("\n" + "=" * 70)
    print("ALL 4 BATCHES COMPLETED. EXECUTING FINAL MANIFEST VALIDATION...")
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
