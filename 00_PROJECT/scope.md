# Project Scope & Boundaries

## 1. In-Scope Areas

### 1.1 Problem Formulations
- **Supervised Multi-Class Classification**: Discrete output space $\mathcal{Y} = \{1, \dots, K\}$ with continuous input feature space $\mathcal{X} \subseteq \mathbb{R}^d$.
- **Noise Types Covered**:
  1. *Symmetric (Uniform) Label Noise*: Every true label $Y=i$ flips to any other class $j \ne i$ with uniform probability $\eta / (K-1)$.
  2. *Asymmetric / Class-Conditional Label Noise*: Flipping probabilities depend on the true class $Y=i$, parameterized by transition matrix $T_{ij} = P(\tilde{Y}=j \mid Y=i)$ (e.g., pairs like TRUCK $\leftrightarrow$ AUTOMOBILE, BIRD $\rightarrow$ AIRPLANE, DEER $\rightarrow$ HORSE).
  3. *Real-World Human Annotator Label Noise*: Real crowdsourced human errors (CIFAR-10N, Animal-10N).

### 1.2 Theoretical Scope
- Expected clean risk $R(f) = \mathbb{E}_{(X,Y)\sim \mathcal{D}}[\ell(f(X), Y)]$ vs corrupted risk $R_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{(X,\tilde{Y})\sim \tilde{\mathcal{D}}}[\ell(f(X), \tilde{Y})]$.
- Unbiased risk estimators via backward loss correction $\ell_{\text{backward}} = T^{-1} \vec{\ell}(f(x))$.
- Consistency, excess risk bounds, and Rademacher complexity under label noise.
- Classification calibration and Expected Calibration Error (ECE) dynamics under posterior corruption.

### 1.3 Model Scope
- Linear/Convex: Multi-Class Logistic Regression.
- Non-parametric: Decision Tree / Random Forest.
- Deep Neural Networks: 2-layer MLP (for controlled capacity analysis), PreAct-ResNet18 / ResNet-50 (for modern benchmark representation).

### 1.4 Datasets
- Synthetic benchmarks: CIFAR-10, CIFAR-100, MNIST, SVHN.
- Tabular benchmark: UCI Adult (for low-dimensional tabular baseline verification).
- Real-world human noisy benchmarks: CIFAR-10N (Clean, Aggregate, Random, Worst), Animal-10N.

---

## 2. Explicit Out-of-Scope Areas

- **Unsupervised / Self-Supervised Pretraining Solely**: We do not study pure self-supervised contrastive learning without fine-tuning on noisy downstream tasks.
- **Continuous Regression Problems**: Label noise on continuous targets (e.g. noisy regression, additive Gaussian target noise) is excluded to keep the focus on discrete classification margins and transition matrices.
- **Adversarial Perturbations ($L_\infty$ attacks)**: Adversarial attacks on input features $X + \delta$ are distinct from stochastic label corruptions $\tilde{Y}$ and are outside this project's core scope.
- **Hardware-Specific Acceleration / Low-Bit Quantization**: Focus is on mathematical algorithms and generalisation dynamics, not GPU kernel optimization.

---

## 3. Scope Evolution & Justification

| Topic | Initial Consideration | Final Scope Status | Rationale |
| :--- | :--- | :--- | :--- |
| Only 2-layer MLP on MNIST | Initial idea | **Expanded** to include ResNet & CIFAR-10/100/CIFAR-10N | Saturated; testing solely on MNIST with shallow MLP lacks modern empirical validity. |
| Only synthetic symmetric noise | Initial idea | **Expanded** to Asymmetric & Human noise | Symmetrical noise is oversimplified; real-world noise is rarely uniform across classes. |
| Pure accuracy metric | Initial idea | **Expanded** to Calibration (ECE, Brier) & $\| \hat{T} - T \|_F$ | Miscalibration under noise is a critical, underexplored safety property. |
