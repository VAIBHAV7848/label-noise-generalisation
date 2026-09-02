# Experimental Datasets Specification

To guarantee empirical breadth, rigorous model-capacity analysis, and real-world transferability, the experimental suite encompasses six benchmark datasets spanning vision, tabular, synthetic, and real crowdsourced human noise.

---

## 1. Dataset Matrix

| Dataset | Modality | Classes ($K$) | Training Samples | Test Samples | Dimensionality | Noise Regime | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MNIST** | Vision (Grayscale) | 10 | 60,000 | 10,000 | $28 \times 28 \times 1$ | Synthetic (Sym/Asym) | Low-complexity sanity checking; shallow model capacity benchmarking |
| **CIFAR-10** | Vision (RGB) | 10 | 50,000 | 10,000 | $32 \times 32 \times 3$ | Synthetic (Sym/Asym) | Core standard benchmark for loss correction & robust loss evaluation |
| **CIFAR-100** | Vision (RGB) | 100 | 50,000 | 10,000 | $32 \times 32 \times 3$ | Synthetic (Sym/Asym) | High-class-count benchmark ($K=100$) evaluating transition matrix scalability |
| **CIFAR-10N** | Vision (RGB) | 10 | 50,000 | 10,000 | $32 \times 32 \times 3$ | Real Human Annotations | Real-world crowdsourced human error (Clean, Aggregate, Random, Worst) |
| **Animal-10N** | Vision (RGB) | 10 | 50,000 | 5,000 | $64 \times 64 \times 3$ | Real Web Noise ($\approx 8.4\%$) | Naturally noisy web-scraped visual confusions |
| **UCI Adult** | Tabular | 2 | 32,561 | 16,281 | 14 features | Synthetic (Sym/Asym) | Non-vision tabular baseline for Logistic Regression & Decision Trees |

---

## 2. Train / Validation / Test Splits

- **Synthetic Datasets (CIFAR-10, CIFAR-100, MNIST)**:
  - 45,000 training instances (injected with synthetic noise $T$).
  - 5,000 validation instances (reserved for model selection, early stopping, and hyperparameter tuning).
  - 10,000 test instances (strictly clean ground truth, evaluated only for final performance reporting).
- **CIFAR-10N**:
  - Uses exact official splits from Wei et al. (ICLR 2022) with original 10,000 clean test images.
- **Data Preprocessing & Augmentations**:
  - Vision benchmarks use standard random cropping ($32 \times 32$ with 4-pixel padding), random horizontal flipping, and per-channel normalization (mean/std). No heavy mixup/CutMix is applied during base baseline evaluation to prevent confounding noise mitigation with data augmentation.
