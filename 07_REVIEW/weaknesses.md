# Potential Reviewer Criticisms & Defense Strategies

This document simulates adversarial peer review from top-tier venues (NeurIPS / ICML / ICLR / AISTATS), listing expected criticisms, vulnerability severity, and explicit defenses.

---

## 1. Reviewer Attack Surface Analysis

### Criticism 1: "The study mostly evaluates existing loss functions and baselines rather than introducing a completely new standalone learning algorithm."
- **Severity**: **Medium-High**.
- **Reviewer Argument**: "The paper provides thorough analysis and bounds under perturbation, but does it offer a new SOTA algorithm that beats DivideMix?"
- **Defense / Mitigation**:
  1. Position the paper as a rigorous **analytical and empirical diagnostic study** (similar to Arpit et al. 2017, Wei et al. 2022, or Charoenphakdee et al. 2019) that uncovers structural trade-offs between generalisation accuracy, model capacity, and confidence calibration.
  2. Highlight the novel theoretical contribution (Proposition 2 excess risk bounds under matrix misspecification) and the first unified calibration benchmark under noisy labels.
  3. Tier-1 conferences explicitly welcome foundational benchmark and analytical studies (e.g. NeurIPS Datasets & Benchmarks Track, ICML evaluation papers).

---

### Criticism 2: "Synthetic class-conditional noise is artificial; real noise is instance-dependent."
- **Severity**: **High (if unaddressed)**.
- **Reviewer Argument**: "Why should we care about transition matrix $T$ bounds when real-world noise depends on $X$?"
- **Defense / Mitigation**:
  1. We explicitly include CIFAR-10N (real human annotator noise) and Animal-10N (real web-scraped noise) in the primary evaluation matrix.
  2. We dedicate Section 5/6 to analyzing the exact empirical breakdown of class-conditional methods when applied to instance-dependent human noise, directly answering the reviewer's concern with quantitative data.

---

### Criticism 3: "Are the classical baselines (Logistic Regression, Decision Tree) necessary in a modern deep learning paper?"
- **Severity**: **Low-Medium**.
- **Reviewer Argument**: "Nobody uses Logistic Regression on CIFAR-10 in practice."
- **Defense / Mitigation**:
  1. Clarify that Logistic Regression and Decision Trees are evaluated on UCI Adult (tabular) and downsampled representations to provide a continuous theoretical capacity continuum from underparameterized convex models ($p \ll N$) to overparameterized deep networks ($p \gg N$).
  2. This isolates whether early-learning and memorisation inflection points are universal statistical properties or specific to overparameterized gradient-based deep architectures.
