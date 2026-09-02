# Transition-Matrix Estimators Audit

This document compares each estimator in `src/estimators/` with its authoritative publication and details required corrections.

---

## 1. Audit of Anchor-Point Estimator (`src/estimators/anchor_point.py`)

- **Primary Reference**: Patrini et al., "Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach", CVPR 2017, Section 4.1.
- **Original Algorithm**:
  1. Base model is trained with standard cross-entropy on noisy dataset $\tilde{\mathcal{S}}$, yielding predicted noisy posteriors $\hat{p}(\tilde{Y} = k \mid x)$.
  2. For each class $i \in \{1, \dots, K\}$, anchor points are identified by finding instances that maximize $\hat{p}(\tilde{Y}=i \mid x)$ **across the entire dataset** $\mathcal{S}$ (or the top 97th percentile of $\hat{p}_i$):
     $$\bar{x}^i = \arg\max_{x \in \mathcal{S}} \hat{p}(\tilde{Y} = i \mid x)$$
  3. The row $i$ of $\hat{T}$ is estimated as:
     $$\hat{T}_{ij} = \hat{p}(\tilde{Y} = j \mid \bar{x}^i)$$
- **Discrepancy Found in Code**:
  - The Phase 1 code filtered samples by `noisy_labels == i` before taking the percentile.
  - *Why this is a bug*: If an anchor point for class $i$ had its label corrupted to $j \ne i$, searching only within `noisy_labels == i` misses the true anchor point. Furthermore, searching within `noisy_labels == i` biases the selected instances towards non-corrupted samples rather than true extreme feature representations.
- **Required Fix**:
  - Search top percentile across all instances in the dataset, or allow an explicit `global_search=True` flag adhering strictly to Patrini et al. (2017).

---

## 2. Audit of Dual-T Estimator (`src/estimators/dual_t.py`)

- **Primary Reference**: Xia et al., "Are Anchor Points Really Indispensable in Label-Noise Learning?", NeurIPS 2019.
- **Original Algorithm**:
  - Decomposes transition matrix into an intermediate transition matrix $T_e$ and utilizes slack variables $\Delta$ to solve:
    $$\min_{T, \Delta} \|T_e - T \Delta\|_F^2 \quad \text{s.t.} \quad T_{ij} \ge 0, \sum_j T_{ij} = 1, \Delta \ge 0$$
- **Discrepancy Found in Code**:
  - Our Phase 1 code applied a top-5 average with a heuristic `slack_scale = 0.95`.
  - *Status*: The Phase 1 code is a simplified heuristic approximation, not the full constrained convex optimization of Xia et al. (2019).
- **Required Fix**:
  - Clearly document in code docstrings and research reports that this is a lightweight heuristic approximation, or integrate the full matrix factorization solver.

---

## 3. Audit of Confident Learning Estimator (`src/estimators/confident_learning.py`)

- **Primary Reference**: Northcutt et al., "Confident Learning: Estimating Uncertainty in Dataset Labels", JAIR 2021, Vol. 70.
- **Original Algorithm**:
  1. Requires **out-of-sample** predicted probabilities $\hat{P}(\tilde{Y} = k \mid x)$ computed via 5-fold cross validation.
  2. Evaluates per-class self-confidence threshold $t_j = \frac{1}{|X_{\tilde{y}=j}|} \sum_{x \in X_{\tilde{y}=j}} \hat{P}(\tilde{Y} = j \mid x)$.
  3. Constructs unnormalized matrix $C_{\tilde{y}, y^*}$ where each sample $x \in X_{\tilde{y}=j}$ is assigned to latent true class $k^* = \arg\max_{k \in \{1..K\} : \hat{p}_k(x) \ge t_k} (\hat{p}_k(x) - t_k)$.
- **Discrepancy Found in Code**:
  - Phase 1 code allowed a sample to increment multiple entries $C_{j, k}$ if $\hat{p}_k \ge t_k$ for multiple $k$.
- **Required Fix**:
  - Implement the single-assignment max-margin rule $k^* = \arg\max_{k: \hat{p}_k \ge t_k} \hat{p}_k(x)$ so sample count is strictly preserved.
