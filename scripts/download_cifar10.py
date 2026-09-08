"""Robust multi-source downloader and preparer for CIFAR-10."""

import os
import sys
import tarfile
import urllib.request
import torchvision.datasets as datasets
import time

DATA_DIR = os.path.abspath("data")


def ensure_cifar10_ready():
    os.makedirs(DATA_DIR, exist_ok=True)
    target_dir = os.path.join(DATA_DIR, "cifar-10-batches-py")
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) >= 6:
        print("CIFAR-10 already extracted and ready.")
        return True

    # Method 1: Use Torchvision native downloader with retries
    print("Attempting torchvision native CIFAR-10 download...")
    for attempt in range(5):
        try:
            datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
            datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
            if os.path.exists(target_dir) and len(os.listdir(target_dir)) >= 6:
                print("CIFAR-10 prepared successfully via torchvision.")
                return True
        except Exception as e:
            print(f"Torchvision download attempt {attempt+1} failed: {e}. Retrying in 3s...")
            time.sleep(3.0)

    # Method 2: Direct UofT mirror stream
    url = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    print(f"Attempting direct stream download from {url}...")
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as response, open(tar_path, "wb") as out_file:
                while True:
                    buf = response.read(1024 * 1024)
                    if not buf:
                        break
                    out_file.write(buf)
            print("Extracting downloaded tarball...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=DATA_DIR)
            if os.path.exists(target_dir) and len(os.listdir(target_dir)) >= 6:
                print("CIFAR-10 extracted successfully.")
                return True
        except Exception as e:
            print(f"Direct download attempt {attempt+1} failed: {e}. Retrying...")
            time.sleep(3.0)

    return False


def ensure_cifar10n_ready():
    """Ensure CIFAR-10_human.pt is downloaded and available."""
    target_file = os.path.join(DATA_DIR, "CIFAR-10_human.pt")
    if os.path.exists(target_file) and os.path.getsize(target_file) > 1000000:
        print("CIFAR-10N human annotations already downloaded and ready.")
        return True

    url = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-10_human.pt"
    print(f"Downloading CIFAR-10N human annotations from {url}...")
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as response, open(target_file, "wb") as out_file:
                while True:
                    buf = response.read(1024 * 1024)
                    if not buf:
                        break
                    out_file.write(buf)
            if os.path.exists(target_file) and os.path.getsize(target_file) > 1000000:
                print("CIFAR-10N human annotations downloaded successfully.")
                return True
        except Exception as e:
            print(f"CIFAR-10N download attempt {attempt+1} failed: {e}. Retrying in 2s...")
            time.sleep(2.0)

    return False


def main():
    cifar10_ok = ensure_cifar10_ready()
    cifar10n_ok = ensure_cifar10n_ready()
    return 0 if (cifar10_ok and cifar10n_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

