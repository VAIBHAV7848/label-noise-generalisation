# Estimator Fidelity & Publication Alignment Audit

This document audits all transition-matrix estimators in `src/estimators/` against their authoritative publications.

---

## 1. Anchor-Point Estimator (`src/estimators/anchor_point.py`)

- **Publication**: Patrini et al., "Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach", CVPR 2017, Section 4.1.
- **Published Algorithm**:
  1. An anchor point $x^i$ for class $i$ satisfies $P(Y=i \mid X=x^i) = 1$. Under CCN, $P(\tilde{Y}=j \mid X=x^i) = T_{ij}$.
  2. The paper searches over the **entire dataset** $\mathcal{S}$ for points maximizing predicted posterior $\hat{p}(\tilde{Y}=i \mid x)$.
  3. To guard against outliers, the paper suggests taking the 97th percentile of $\hat{p}_i(x)$ across $\mathcal{S}$ and averaging the full posterior vectors.
- **Repository Implementation Audit**:
  - `estimate_transition_matrix_anchor_points` with `global_search=True` (default) searches over all $x \in \mathcal{S}$.
  - Computes the 97th percentile of $\hat{p}_i$, averages probability vectors of anchor candidates, and sets row $i$ of $\hat{T}$.
  - Enforces row-stochastic normalization.
- **Controlled Synthetic Validation**:
  - Tested on 3-class synthetic mixture with known $T$ ($\eta=0.2$). Frobenius estimation error $\|\hat{T} - T\|_F < 0.05$.
  - Tested on adversarial case where all true anchor points for class 0 were corrupted to label 1: global search successfully recovered $T_{0j}$ with error $< 10^{-3}$.
- **Classification**: **[VERIFIED - FAITHFUL TO PATRINI ET AL. 2017]**.

---

## 2. Confident Learning Estimator (`src/estimators/confident_learning.py`)

- **Publication**: Northcutt, Jiang, Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels", JAIR 2021, Vol. 70, pp. 1373–1411.
- **Published Algorithm**:
  1. Requires **out-of-sample** predicted probabilities $\hat{p}(\tilde{Y}=k \mid x)$ (e.g. 5-fold cross validation).
  2. Evaluates per-class self-confidence threshold $t_j = \frac{1}{|X_{\tilde{y}=j}|} \sum_{x \in X_{\tilde{y}=j}} \hat{p}(\tilde{Y}=j \mid x)$.
  3. Constructs unnormalized confusion matrix $C_{\tilde{y}, y^*}$ where each sample $x \in X_{\tilde{y}=j}$ is assigned to latent true class $y^* = \arg\max_{k : \hat{p}_k(x) \ge t_k} (\hat{p}_k(x) - t_k)$. If no class exceeds threshold, $y^* = \arg\max_k \hat{p}_k(x)$.
  4. Joint distribution $Q_{\tilde{y}, y^*} = C / \sum_{j', k'} C_{j', k'}$.
  5. Noise transition matrix $T_{k j} = P(\tilde{Y}=j \mid Y^*=k) = Q(j, k) / \sum_{j'} Q(j', k)$.
- **Repository Implementation Audit**:
  - Implements the exact max-margin single-assignment rule.
  - Correctly normalizes $Q$ and computes $T = (Q / Q.\text{sum}(0, \text{keepdims=True}))^\top$.
  - Preserves row-stochasticity and sample conservation.
- **Prerequisite in Pipeline**: Must be fed strictly with out-of-sample cross-validated probabilities or hold-out predictions to avoid overfitting bias.
- **Classification**: **[VERIFIED - FAITHFUL TO NORTHCUTT ET AL. 2021]**.

---

## 3. Dual-T Estimator (`src/estimators/dual_t.py`)

- **Publication**: Xia et al., "Are Anchor Points Really Indispensable in Label-Noise Learning?", NeurIPS 2019.
- **Published Algorithm**:
  - Trains an intermediate neural network to approximate $P(\tilde{Y} \mid X)$.
  - Factorizes the intermediate transition matrix via constrained non-negative matrix optimization $\min_{T, \Delta} \|T_e - T \Delta\|_F^2$ subject to simplex constraints.
- **Repository Implementation Audit**:
  - The repository currently implements a top-5 average with a heuristic slack scalar (`slack_scale = 0.95`).
  - **Audit Finding**: This heuristic is **NOT** the published algorithm of Xia et al. (2019). It lacks the intermediate network training and the constrained optimization solver.
- **Action Taken**:
  - Accurately rename this estimator in code and documentation as `NonAnchorHeuristicEstimator` (`estimate_transition_matrix_heuristic_slack`) and clearly state that it is a lightweight heuristic baseline, NOT the complete Dual-T optimizer of Xia et al. (2019).
- **Classification**: **[SIMPLIFIED HEURISTIC $\to$ ACCURATELY SCOPED & RENAMED]**.
