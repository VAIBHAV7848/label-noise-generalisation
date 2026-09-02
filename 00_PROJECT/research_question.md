# Research Questions & Evaluation

## 1. Context & Motivation

Label noise is ubiquitous in modern supervised learning, arising from human annotator error, automated web scraping, sensor inaccuracies, and crowdsourcing ambiguities. While standard Empirical Risk Minimization (ERM) with Cross-Entropy Loss achieves strong generalisation on clean benchmarks, label noise significantly degrades classification accuracy, corrupts decision boundaries, inflates confidence miscalibration (Expected Calibration Error), and triggers rapid capacity-dependent memorisation.

A major pitfall in academic label-noise research is formulating questions that are already fully solved (e.g., "Does symmetric noise degrade test accuracy of standard neural networks?" or "Is MAE robust to symmetric noise?"). To make a meaningful, scientifically defensible contribution, we must formulate research questions that rigorously bridge theoretical risk-estimator bounds, calibration degradation dynamics, transition matrix identifiability, and practical robustness under realistic (instance-dependent and human) noise.

---

## 2. Candidate Research Questions Analysis & Ranking

### Candidate RQ 1 (Naive / Saturated — REJECTED)
> *"How does varying symmetric label noise from 0% to 50% affect the test accuracy of Logistic Regression, Decision Trees, and 2-layer MLPs on MNIST and CIFAR-10?"*
- **Why it matters**: Basic sensitivity baseline.
- **Existing Literature**: Thoroughly investigated since Angluin & Laird (1988), Natarajan et al. (2013), and Arpit et al. (2017).
- **Novelty Rating**: **None (Saturated)**.
- **Verdict**: Inadequate for peer-reviewed publication; trivial demonstration of known phenomena.

---

### Candidate RQ 2 (Robust Losses vs. Architecture Capacity — PARTIAL)
> *"What is the exact trade-off between loss function robustness (e.g., GCE, SCE, MAE) and model capacity in preventing noisy label memorisation across convex vs non-convex models?"*
- **Why it matters**: Explores whether symmetric losses penalize optimization speed more severely in shallow vs deep models.
- **Existing Literature**: Zhang & Sabuncu (NeurIPS 2018), Charoenphakdee et al. (ICML 2019), Ma et al. (ICML 2020).
- **Novelty Rating**: **Medium-Low**.
- **Limitation**: While interesting, the optimization slowdown of symmetric losses (e.g. MAE) is well-known and largely characterized by vanishing gradients on hard examples.

---

### Candidate RQ 3 (Calibration & Decision Margin Degradation under Label Noise — HIGH)
> *"How does label noise structurally alter the confidence calibration (ECE, Brier score) and feature representation geometry of classifiers under symmetric vs class-conditional noise, and do loss correction techniques restore calibration or merely classification accuracy?"*
- **Why it matters**: Almost all prior works optimize strictly for top-1 accuracy under label noise; however, in safety-critical domains (medical imaging, autonomous driving), miscalibrated overconfident predictions on corrupted data pose severe hazards.
- **Existing Literature**: Guo et al. (ICML 2017), Thulasidasan et al. (NeurIPS 2019), Cheng et al. (NeurIPS 2020), Wei et al. (ICLR 2022).
- **Theoretical Opportunity**: Derivation of calibration bounds under noisy posterior distributions; analyzing whether forward/backward loss corrections introduce probability distribution distortion.
- **Empirical Opportunity**: Systematic benchmark measuring Top-1 Acc, ECE, AdaECE, and Brier Score across standard CE, GCE, SCE, Forward/Backward Correction, and DivideMix on synthetic (CIFAR-10/100) and human noise (CIFAR-10N).
- **Novelty Rating**: **High (Underexplored intersection)**.

---

### Candidate RQ 4 (Comprehensive Primary Recommended Research Question — RECOMMENDED)
> **Primary RQ:**
> *"Under what exact theoretical and empirical conditions do loss-correction methods, robust loss functions, and sample selection algorithms maintain both generalisation accuracy and probability calibration when subjected to symmetric, class-conditional, and realistic instance-dependent label noise across varying model capacities?"*
>
> **Sub-Questions (SRQs):**
> - **SRQ 1 (Theoretical Risk & Identifiability):** Under what bounds does backward loss correction $\ell_{\text{backward}} = T^{-1}\vec{\ell}$ maintain unbiased risk estimation when the noise transition matrix $T$ is imperfectly estimated ($\| \hat{T} - T \|_F > 0$), and how does transition matrix misspecification propagate to excess risk?
> - **SRQ 2 (Capacity & Memorisation Dynamics):** How does the transition from underparameterized models (Logistic Regression) to overparameterized models (MLP, ResNet) alter the critical noise threshold $\eta_{\text{crit}}$ at which empirical risk minimization fails to converge to the optimal Bayes decision rule?
> - **SRQ 3 (Calibration vs. Accuracy Trade-off):** Do robust loss functions (e.g., GCE, SCE) and loss correction methods preserve or degrade predictive probability calibration (Expected Calibration Error) compared to standard Cross-Entropy, and can post-hoc calibration methods (Temperature Scaling) repair noise-induced miscalibration?
> - **SRQ 4 (Synthetic vs. Realistic Noise Discrepancy):** To what extent do theoretical guarantees and empirical rankings established on synthetic class-conditional noise transfer to real-world human annotator noise (e.g., CIFAR-10N, Animal-10N)?

---

## 3. Evaluation & Justification Matrix

| RQ Candidate | Theoretical Depth | Empirical Relevance | Novelty Gap | Risk Level | Target Venue Alignment | Overall Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RQ 1** (Naive sensitivity) | Low | Low | Zero | High (desk reject) | None | Rejected |
| **RQ 2** (Loss trade-offs) | Medium | Medium | Low-Medium | Medium | Workshops | Secondary Ablation |
| **RQ 3** (Calibration under noise) | High | High | High | Low-Medium | NeurIPS/ICML/AISTATS | Core Pillar |
| **RQ 4 (Primary Composite)** | **High** | **Very High** | **High** | **Low (Defensible)** | **Tier-1 (ICML/NeurIPS/JMLR/TMLR)** | **Primary Recommended** |

---

## 4. Defensibility Assessment

The primary research question (RQ 4) is scientifically defensible because:
1. It does not re-invent already solved proofs (e.g. Natarajan 2013 backward unbiasedness is taken as the baseline and analyzed under *perturbation/misspecification error*).
2. It expands beyond accuracy into **calibration reliability** (ECE) and **representation stability**, which are critical unmet needs in literature.
3. It directly evaluates the domain transfer gap between synthetic transition matrices and real-world human noise (CIFAR-10N).
