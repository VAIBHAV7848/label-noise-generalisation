# Reproduction Instructions

This document provides exact step-by-step instructions for reproducing the theoretical derivations, configurations, and experimental runs (active starting Phase 1).

---

## 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/VAIBHAV7848/label-noise-generalisation.git
cd label-noise-generalisation

# Option A: Conda
conda env create -f 09_REPRODUCIBILITY/environment.yml
conda activate label-noise-gen

# Option B: Pip virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r 09_REPRODUCIBILITY/requirements.txt
```

---

## 2. Directory Structure Verification

```bash
# Verify repository integrity
ls -la 00_PROJECT/ 01_LITERATURE/ 02_THEORY/ 03_METHOD/ 04_EXPERIMENTS/
```

---

## 3. Configuration Management
All experimental runs in subsequent phases will be parameterized via YAML files stored in `09_REPRODUCIBILITY/config/`.
