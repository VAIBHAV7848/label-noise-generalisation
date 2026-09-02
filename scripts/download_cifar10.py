"""Parallel multi-connection downloader for CIFAR-10 python tarball with streaming and progress."""

import os
import sys
import tarfile
import urllib.request
import concurrent.futures
import time

URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
TOTAL_SIZE = 170498071  # exact byte size
DATA_DIR = os.path.abspath("data")
TAR_PATH = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
NUM_CHUNKS = 16  # 16 chunks of ~10.6 MB each


def download_chunk(chunk_id, start_byte, end_byte):
    chunk_file = os.path.join(DATA_DIR, f"cifar_part_{chunk_id:02d}.tmp")
    expected_len = end_byte - start_byte + 1
    if os.path.exists(chunk_file) and os.path.getsize(chunk_file) == expected_len:
        print(f"Chunk {chunk_id:02d} already downloaded.")
        return chunk_id, True

    req = urllib.request.Request(URL, headers={"Range": f"bytes={start_byte}-{end_byte}", "User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                with open(chunk_file, "wb") as f:
                    downloaded = 0
                    while True:
                        buf = response.read(65536)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                if downloaded == expected_len:
                    print(f"Chunk {chunk_id:02d} complete ({downloaded/1024/1024:.2f} MB).")
                    return chunk_id, True
        except Exception as e:
            print(f"Chunk {chunk_id:02d} attempt {attempt+1} failed: {e}")
            time.sleep(2.0)
    return chunk_id, False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    target_dir = os.path.join(DATA_DIR, "cifar-10-batches-py")
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) >= 6:
        print("CIFAR-10 already extracted and ready.")
        return 0

    chunk_size = TOTAL_SIZE // NUM_CHUNKS
    ranges = []
    for i in range(NUM_CHUNKS):
        start = i * chunk_size
        end = TOTAL_SIZE - 1 if i == NUM_CHUNKS - 1 else (i + 1) * chunk_size - 1
        ranges.append((i, start, end))

    print(f"Downloading CIFAR-10 in {NUM_CHUNKS} parallel streaming connections...")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CHUNKS) as executor:
        futures = [executor.submit(download_chunk, cid, s, e) for cid, s, e in ranges]
        for f in concurrent.futures.as_completed(futures):
            cid, success = f.result()
            if not success:
                print(f"Failed chunk {cid}")
                return 1

    dt = time.time() - t0
    print(f"All {NUM_CHUNKS} chunks downloaded in {dt:.1f}s ({TOTAL_SIZE/dt/1024/1024:.2f} MB/s). Assembling...")

    # Assemble
    with open(TAR_PATH, "wb") as outfile:
        for i in range(NUM_CHUNKS):
            chunk_file = os.path.join(DATA_DIR, f"cifar_part_{i:02d}.tmp")
            with open(chunk_file, "rb") as infile:
                outfile.write(infile.read())
            os.remove(chunk_file)

    print("Extracting tarball...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)

    print("CIFAR-10 extracted successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
