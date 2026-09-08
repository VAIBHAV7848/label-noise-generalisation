# Reproducibility Guide: Excess Risk Bounds, Inversion Conditioning, and Calibration Distortion under Class-Conditional Label Noise

This document provides complete instructions for reproducing the 84-run empirical study, statistical analysis, figures, and publication tables presented in the manuscript.

---

## 1. Repository and System Requirements

### Hardware Environment
- **GPU**: NVIDIA Tesla T4 (16 GB VRAM) or equivalent.
- **CPU**: Intel Xeon or AMD EPYC (4+ cores recommended).
- **RAM**: 16 GB minimum.
- **Disk**: 15 GB free space for CIFAR-10 data, checkpoints, and logs.

### Software Environment
- **Operating System**: Ubuntu 22.04 LTS / Debian Linux.
- **Python**: Version 3.10+.
- **PyTorch**: Version 2.0+ with CUDA support.
- **Key Dependencies**: `torchvision`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.

---

## 2. Environment Setup

Clone the repository and install dependencies:
```bash
git clone https://github.com/VAIBHAV7848/label-noise-generalisation.git
cd label-noise-generalisation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pandas matplotlib seaborn scikit-learn
```

Verify the environment and unit tests:
```bash
python3 -m unittest discover tests/
# Expected: 33 tests pass (100% success rate)
```

---

## 3. Experimental Protocol and Dataset Partitions

### Benchmark Dimensions
- **Dataset**: CIFAR-10 (60,000 $32 \times 32$ colour images, 10 classes).
- **Model Backbone**: PreActResNet-18 (batch size 128).
- **Optimization**: SGD (momentum 0.9, weight decay $5 \times 10^{-4}$, learning rate 0.05, cosine annealing schedule for 30 epochs).
- **Data Augmentation**: Random horizontal flip ($p=0.5$), random crop ($32 \times 32$, padding 4).
- **Random Seeds**: $\{42, 1337, 2024\}$.
- **Total Experimental Runs**: $4 \text{ Regimes} \times 7 \text{ Diagnostic Tracks} \times 3 \text{ Seeds} = \mathbf{84 \text{ Runs}}$.

### Partitions
1. **Noisy Training Split**: 35,000 examples (corrupted according to regime).
2. **Clean Validation Split**: 5,000 clean examples (used for model monitoring and clean TS).
3. **Corrupted Validation Split**: 5,000 corrupted examples (used to evaluate corrupted TS trap).
4. **Buffer Split**: 5,000 held-out examples.
5. **Clean Test Split**: 10,000 clean test examples (used strictly for final evaluation).

### Noise Regimes
1. `clean`: $\eta = 0.0$, $T = I$.
2. `symmetric_0.2`: $\eta = 0.2$, off-diagonal $T_{ij} = 0.2/9 \approx 0.0222$.
3. `symmetric_0.5`: $\eta = 0.5$, off-diagonal $T_{ij} = 0.5/9 \approx 0.0556$.
4. `asymmetric_0.4`: $\eta = 0.4$, class-conditional pair flips:
   - Truck $\to$ Automobile
   - Bird $\to$ Airplane
   - Deer $\to$ Horse
   - Cat $\leftrightarrow$ Dog

### Diagnostic Tracks
1. `CE`: Standard Cross-Entropy.
2. `GCE`: Generalized Cross-Entropy ($q=0.7$).
3. `SCE`: Symmetric Cross-Entropy ($\alpha=0.1, \beta=1.0$).
4. `ForwardCorrection_TrueT`: Forward Loss Correction with ground-truth matrix $T$.
5. `ForwardCorrection_AnchorT`: Forward Loss Correction with $\hat{T}$ estimated via 97th percentile anchor points.
6. `ForwardCorrection_ConfidentLearningT`: Forward Loss Correction with $\hat{T}$ estimated via Confident Learning (3-fold CV).
7. `ForwardCorrection_BadT`: Forward Loss Correction with perturbed matrix $\hat{T} = 0.5 T + 0.5 \mathbf{U}$.

---

## 4. Reproducing the Experiments

### Running Individual Tracks
To execute a single experiment:
```bash
python3 src/main.py \
    --dataset cifar10 \
    --noise_type asymmetric \
    --noise_rate 0.4 \
    --track ForwardCorrection_TrueT \
    --seed 42 \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.05 \
    --output_dir results/
```

### Reproducing the Complete 84-Run Grid
The complete campaign can be reproduced using the pilot execution runner:
```bash
python3 scripts/run_pilot_campaign.py --output_dir 05_RESULTS/raw/
```

---

## 5. Statistical Analysis and Metric Compilation

To process raw logs, compute paired $t$-tests ($N=3, \text{df}=2$), Holm-Bonferroni corrections, Cohen's $d$, and temperature scaling deltas:
```bash
python3 scripts/run_statistical_analysis.py \
    --input_dir 05_RESULTS/raw/ \
    --output_dir 05_RESULTS/statistical_analysis/
```
Primary output file: `05_RESULTS/statistical_analysis/comprehensive_statistical_analysis.json`.

---

## 6. Regenerating Figures and Tables

### Regenerate All 10 Publication Figures:
```bash
python3 scripts/generate_figures.py \
    --data 05_RESULTS/statistical_analysis/comprehensive_statistical_analysis.json \
    --output_dir 08_PAPER/IEEE_FINAL/figures/
```
Generated figures:
1. `fig1_accuracy_vs_noise.pdf`
2. `fig2_ece_vs_noise.pdf`
3. `fig3_brier_vs_noise.pdf`
4. `fig4_transition_matrix_error.pdf`
5. `fig5_performance_vs_frobenius.pdf`
6. `fig6_performance_vs_condition_number.pdf`
7. `fig7_reliability_diagrams.pdf`
8. `fig8_clean_vs_corrupted_ts_delta.pdf`
9. `fig9_training_generalization_dynamics.pdf`
10. `fig10_seed_variability_and_memorization.pdf`

### Regenerate All 5 LaTeX Tables:
```bash
python3 scripts/generate_tables.py \
    --data 05_RESULTS/statistical_analysis/comprehensive_statistical_analysis.json \
    --output_dir 08_PAPER/IEEE_FINAL/tables/
```
Generated tables:
1. `tab1_main_performance.tex`
2. `tab2_calibration_metrics.tex`
3. `tab3_transition_matrix_estimation.tex`
4. `tab4_statistical_tests.tex`
5. `tab5_hypothesis_srq_decisions.tex`

---

## 7. Compiling the Final IEEE Manuscript

To compile the manuscript and supplementary materials:
```bash
cd 08_PAPER/IEEE_FINAL
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

cd supplement
pdflatex -interaction=nonstopmode supplement.tex
pdflatex -interaction=nonstopmode supplement.tex
```

Verify zero compilation errors and zero overfull hbox warnings:
```bash
grep -i "error" 08_PAPER/IEEE_FINAL/main.log
grep -i "overfull" 08_PAPER/IEEE_FINAL/main.log
```
Expected output: No errors and no overfull warnings.

---

## 8. Data and Code Provenance Integrity

All data splits and random permutations are deterministically generated from fixed random seeds. Checksum verification hashes for the frozen processed data and script outputs are stored in `05_RESULTS/processed/provenance_manifest.json`.
