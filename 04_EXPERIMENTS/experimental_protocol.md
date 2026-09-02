# Experimental Protocol & Evaluation Standards

To enforce publication-grade empirical integrity, all experimental workflows adhere to the following deterministic protocol.

---

## 1. Multi-Seed Protocol & Statistical Rigor

- **Random Seeds**: Every experiment is executed across **5 independent random seeds**:
  $$\text{Seeds} = \{42, 1337, 2024, 7, 999\}$$
- **Reported Statistics**:
  - All tabular and graphical results report the sample mean $\mu$ and sample standard deviation $\sigma$:
    $$\mu \pm \sigma$$
  - Two-tailed paired $t$-tests and Wilcoxon signed-rank tests are computed between proposed methods and baseline methods. Results with $p \ge 0.05$ are explicitly marked as statistically non-significant.
  - Confidence intervals are computed at the 95% level via non-parametric bootstrap ($B = 10,000$ resamples).

---

## 2. Model Architectures & Training Configurations

| Model Family | Representative Architecture | Optimizer | Initial LR | LR Schedule | Weight Decay | Epochs | Batch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear / Convex** | Multi-Class Logistic Regression | L-BFGS / SGD | 0.05 | Cosine Anneal | $10^{-4}$ | 100 | 256 |
| **Non-Parametric** | Decision Tree / Random Forest (100 trees) | Gini / Entropy | N/A | N/A | Max Depth=15 | N/A | N/A |
| **Shallow Neural** | 2-Layer MLP (Hidden dim 512, ReLU, Dropout 0.2) | AdamW / SGD | 0.001 / 0.05 | MultiStepLR | $10^{-4}$ | 100 | 128 |
| **Deep Neural** | PreAct-ResNet18 / ResNet-50 | SGD (Momentum 0.9) | 0.1 | Cosine Annealing | $5 \times 10^{-4}$ | 120 | 128 |

---

## 3. Evaluation Metrics

### 3.1 Classification Performance
- **Clean Test Accuracy (%)**: Top-1 and Top-5 accuracy on the uncorrupted test partition.
- **Noisy Train Accuracy (%) vs. Clean Test Accuracy (%)**: Tracking generalization gap across training epochs to diagnose memorisation onset.

### 3.2 Calibration & Reliability Metrics
- **Expected Calibration Error (ECE)**:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  computed with $M = 15$ equally spaced confidence bins.
- **Adaptive Expected Calibration Error (AdaECE)**: $M = 15$ equal-frequency (adaptive) bins.
- **Brier Score**:
  $$\text{Brier} = \frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K (f_k(x_n) - \mathbf{1}(y_n = k))^2$$
- **Negative Log-Likelihood (NLL)**: Evaluated on the clean test set.

### 3.3 Noise Estimation Quality
- **Frobenius Norm Error**: $\|\hat{T} - T\|_F$.
- **Spectral Error**: $\|\hat{T} - T\|_2$.
