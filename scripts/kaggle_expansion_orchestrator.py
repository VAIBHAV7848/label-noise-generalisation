"""Kaggle GPU Expansion Orchestrator for 77-Run Benchmark Grid.

Manages Phase 2 Expansion (N=5 seeds for synthetic noise + CIFAR-10N human noise benchmark).
Deploys 4 batch kernels to Kaggle Nvidia Tesla T4 GPUs, pushes initial batches,
and monitors / downloads results.
"""

import os
import sys
import json
import time
import shutil
import socket
import tarfile
import subprocess
import numpy as np

# Force IPv4 resolution to prevent timeouts on broken IPv6/NAT64 routes
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, *args, **kwargs):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, *args, **kwargs)
socket.getaddrinfo = _ipv4_getaddrinfo

BASE_DIR = os.path.abspath(".")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.environ["PYTHONPATH"] = f"{BASE_DIR}/scripts:{os.environ.get('PYTHONPATH', '')}"

from src.training.run_pilot import validate_provenance_record

USERNAME = "vaibhavchavanpatil"
NUM_BATCHES = 4
MAX_CONCURRENT_GPU = 2
KERNELS_DIR = os.path.join(BASE_DIR, "kaggle_kernels", "expansion")
OUTPUT_DIR = os.path.join(BASE_DIR, "05_RESULTS", "pilot")
MANIFEST_PATH = os.path.join(BASE_DIR, "04_EXPERIMENTS", "expansion_run_manifest.json")


def create_batch_kernel_files(batch_idx: int):
    """Create directory, python script, and metadata for an expansion batch kernel."""
    batch_dir = os.path.join(KERNELS_DIR, f"batch_{batch_idx}")
    os.makedirs(batch_dir, exist_ok=True)

    kernel_id = f"{USERNAME}/label-noise-expansion-batch-{batch_idx}"
    title = f"Label Noise Expansion Batch {batch_idx}"

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
        "machine_shape": "NvidiaTeslaT4",
    }

    with open(os.path.join(batch_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    script_content = f"""# Auto-generated batch runner for Kaggle Expansion Execution
import os
import sys
import subprocess
import tarfile
import shutil

BATCH_INDEX = {batch_idx}
NUM_BATCHES = {NUM_BATCHES}

print(f"=== STARTING KAGGLE EXPANSION BATCH {{BATCH_INDEX}}/{{NUM_BATCHES}} ===")

# 1. Clone repository from GitHub main
repo_dir = "/kaggle/working/repo"
if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)

GITHUB_AUTH_URL = os.environ.get("GITHUB_CLONE_URL", "https://github.com/VAIBHAV7848/label-noise-generalisation.git")
subprocess.run(["git", "clone", GITHUB_AUTH_URL, repo_dir], check=True)
os.chdir(repo_dir)

# 2. Download CIFAR-10 & CIFAR-10N human annotations
subprocess.run([sys.executable, "scripts/download_cifar10.py"], check=True)

# 3. Output directory for this batch
out_dir = f"/kaggle/working/results_expansion_batch_{{BATCH_INDEX}}"
os.makedirs(out_dir, exist_ok=True)

# 4. Run expansion pilot batch
cmd = [
    sys.executable,
    "-m",
    "src.training.run_pilot",
    "--manifest", "04_EXPERIMENTS/expansion_run_manifest.json",
    "--batch-index", str(BATCH_INDEX),
    "--num-batches", str(NUM_BATCHES),
    "--data-dir", "data",
    "--output-dir", out_dir,
    "--device", "cuda"
]
print("Executing command:", " ".join(cmd))
subprocess.run(cmd, check=True)

# 5. Archive results to root output directory
archive_path = f"/kaggle/working/expansion_results_batch_{{BATCH_INDEX}}.tar.gz"
with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(out_dir, arcname="expansion_results")

# Clean up working files to minimize archive footprint
shutil.rmtree(repo_dir, ignore_errors=True)
shutil.rmtree(out_dir, ignore_errors=True)

print(f"=== EXPANSION BATCH {{BATCH_INDEX}} FINISHED. ARCHIVE SAVED AT {{archive_path}} ===")
"""

    with open(os.path.join(batch_dir, "batch_runner.py"), "w") as f:
        f.write(script_content)

    return batch_dir, kernel_id


def push_kernel(batch_idx: int) -> bool:
    """Push an expansion batch kernel to Kaggle targeting Tesla T4."""
    b_dir, k_id = create_batch_kernel_files(batch_idx)
    res = subprocess.run([
        "python3", "-m", "kaggle", "kernels", "push",
        "-p", b_dir,
        "--accelerator", "NvidiaTeslaT4"
    ], capture_output=True, text=True, env=os.environ)
    if res.returncode == 0:
        print(f"Pushed Expansion Batch {batch_idx} ({k_id}): {res.stdout.strip()}")
        return True
    else:
        print(f"Push failed for Batch {batch_idx}: {res.stderr.strip() or res.stdout.strip()}")
        return False


_kaggle_api = None
def get_kaggle_api():
    global _kaggle_api
    if _kaggle_api is None:
        from kaggle.api.kaggle_api_extended import KaggleApi
        _kaggle_api = KaggleApi()
        _kaggle_api.authenticate()
    return _kaggle_api


def get_kernel_status(kernel_id: str) -> str:
    """Get the current execution status of a Kaggle kernel via in-process API."""
    try:
        api = get_kaggle_api()
        res = api.kernels_status(kernel_id)
        st = getattr(res, "status", None) or (res.get("status") if isinstance(res, dict) else str(res))
        st = str(st).upper()
        if "COMPLETE" in st:
            return "COMPLETE"
        elif "RUNNING" in st or "QUEUED" in st:
            return "RUNNING"
        elif "ERROR" in st or "FAILED" in st or "CANCEL" in st:
            return "ERROR"
        return "UNKNOWN"
    except Exception as e:
        print(f"Error checking status for {kernel_id}: {e}")
        return "UNKNOWN"


def is_batch_completed(batch_idx: int) -> bool:
    """Check if all runs in this batch chunk are validly completed on local disk."""
    if not os.path.exists(MANIFEST_PATH):
        return False
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    all_runs = manifest["runs"]
    chunk_size = int(np.ceil(len(all_runs) / NUM_BATCHES))
    start_idx = batch_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(all_runs))
    batch_runs = all_runs[start_idx:end_idx]

    if not batch_runs:
        return False

    for r in batch_runs:
        res_file = os.path.join(OUTPUT_DIR, f"{r['experiment_id']}.json")
        if not os.path.exists(res_file):
            return False
        try:
            with open(res_file, "r") as rf:
                data = json.load(rf)
            if not validate_provenance_record(data):
                return False
        except Exception:
            return False
    return True


def download_and_extract_results(kernel_id: str, batch_idx: int) -> bool:
    """Download output archive and extract JSON results into OUTPUT_DIR."""
    download_dir = os.path.join(KERNELS_DIR, f"output_batch_{batch_idx}")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    archive_path = os.path.join(download_dir, f"expansion_results_batch_{batch_idx}.tar.gz")
    print(f"Downloading outputs for {kernel_id}...")
    import requests
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        creds = json.load(f)
    auth = (creds["username"], creds["key"])
    url = f"https://www.kaggle.com/api/v1/kernels/output/download/{kernel_id}/expansion_results_batch_{batch_idx}.tar.gz"

    download_ok = False
    try:
        r = requests.get(url, auth=auth, allow_redirects=True, stream=True, timeout=30)
        if r.status_code == 200:
            with open(archive_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            download_ok = True
    except Exception as e:
        print(f"Direct download failed: {e}")

    if not download_ok:
        res = subprocess.run(["python3", "-m", "kaggle", "kernels", "output", kernel_id, "-p", download_dir], capture_output=True, text=True, env=os.environ)
        if res.returncode != 0:
            print(f"Failed to download output for {kernel_id}: {res.stderr.strip()}")
            return False

    if not os.path.exists(archive_path):
        print(f"Archive {archive_path} not found in {download_dir}.")
        return False

    print(f"Extracting {archive_path} into {OUTPUT_DIR}...")
    extracted_count = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.startswith("expansion_results/"):
                rel_path = member.name[len("expansion_results/"):]
                if not rel_path or not member.name.endswith(".json"):
                    continue
                f = tar.extractfile(member)
                if f is not None:
                    dest = os.path.join(OUTPUT_DIR, os.path.basename(rel_path))
                    with open(dest, "wb") as df:
                        df.write(f.read())
                    extracted_count += 1

    print(f"Extracted {extracted_count} expansion result JSONs into {OUTPUT_DIR}.")
    return is_batch_completed(batch_idx)


def launch_initial_phase():
    """Launch Batches 0 and 1 on Kaggle GPU."""
    print("=== LAUNCHING PHASE 2 EXPANSION (BATCHES 0 & 1) ===")
    s0 = push_kernel(0)
    time.sleep(3)
    s1 = push_kernel(1)
    print(f"Launch summary: Batch 0: {s0}, Batch 1: {s1}")
    return s0 and s1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kaggle Expansion Orchestrator")
    parser.add_argument("--launch-initial", action="store_true", help="Launch Batches 0 and 1")
    parser.add_argument("--batch", type=int, default=None, help="Push specific batch index")
    parser.add_argument("--status", action="store_true", help="Print status of all expansion batches")
    args = parser.parse_args()

    if args.launch_initial:
        launch_initial_phase()
    elif args.batch is not None:
        push_kernel(args.batch)
    elif args.status:
        for b in range(NUM_BATCHES):
            kid = f"{USERNAME}/label-noise-expansion-batch-{b}"
            st = get_kernel_status(kid)
            comp = is_batch_completed(b)
            print(f"Batch {b} ({kid}): Status={st}, LocallyCompleted={comp}")
    else:
        print("Use --launch-initial, --batch <idx>, or --status")
