# Hypothesis $\to$ Experiment Traceability Matrix

This document establishes the end-to-end provenance and falsification chain connecting research questions, theoretical gaps, hypotheses, observable predictions, experimental configurations, metrics, and statistical tests.

---

## 1. Traceability Chains

### Hypothesis 1 (H1: Asymmetry-Driven Decision Boundary Displacement)
```
Research Question (Primary RQ & SRQ 1)
  ↓
Research Gap: Distinguishing spectral conditioning κ(T) vs directional asymmetry ||T - T^T||_F in decision boundary displacement
  ↓
Hypothesis Statement: Angular displacement θ(w*, w_clean*) is strictly bounded by ||T - T^T||_F / min_i T_ii and clean margin γ, with zero asymptotic shift under symmetric noise and balanced priors
  ↓
Observable Prediction: Linear classifiers under symmetric noise exhibit 0.0° boundary shift; asymmetric noise produces boundary shift monotonic in ||T - T^T||_F
  ↓
Experiment: Synthetic 2-class & 10-class Gaussian linear classification across 5 noise regimes (Clean, Symmetric η ∈ {0.2, 0.4}, Asymmetric η ∈ {0.2, 0.4})
  ↓
Metric: Hyperplane angle θ = arccos((w_noisy^T w_clean) / (||w_noisy|| ||w_clean||)) and Clean 0-1 Test Error
  ↓
Statistical Test: Spearman rank correlation r_s(θ, ||T - T^T||_F) and one-sample Wilcoxon signed-rank test against θ_0 = 0.0°
  ↓
Falsification Condition: Observed angle θ > 1.0° under symmetric noise with balanced priors (p < 0.01), or lack of statistically significant positive correlation under asymmetric noise
  ↓
Alignment Verdict: [ALIGNED]
```

---

### Hypothesis 2 (H2: Parametric Generalisation Window & Capacity Scaling)
```
Research Question (Primary RQ & SRQ 3)
  ↓
Research Gap: Quantifying how overparameterization ratio p/N contracts the clean generalisation window Δτ and whether loss correction arrests gradient residuals
  ↓
Hypothesis Statement: Clean generalisation window Δτ = τ_memorize - τ_learn contracts inversely with p/N; backward and forward loss correction preserve test accuracy by arresting gradient norms on mislabeled instances
  ↓
Observable Prediction: Increasing hidden width d_h from 64 to 2048 contracts Δτ; loss correction exhibits lower gradient norm on corrupted samples than uncorrected CE
  ↓
Experiment: 2-layer MLP width scaling (d_h ∈ {64, 256, 1024, 4096}) and PreAct-ResNet18 on CIFAR-10 with 40% asymmetric noise
  ↓
Metric: Clean window duration Δτ = epoch(peak val acc) - epoch(train loss < 0.01) and Mean per-sample gradient L_2 norm E_{i ∈ corrupted}[||∇_θ l_i||]
  ↓
Statistical Test: Linear regression log(Δτ) ~ β log(p/N) + ε (testing β < 0) and paired t-test on corrupted gradient norms
  ↓
Falsification Condition: Slope β ≥ 0 (window does not contract), or loss-corrected models exhibit gradient norms on corrupted samples ≥ standard CE
  ↓
Alignment Verdict: [ALIGNED]
```

---

### Hypothesis 3 (H3: Divergent Miscalibration Profiles & Corrupted Validation Recovery)
```
Research Question (Primary RQ & SRQ 2)
  ↓
Research Gap: Gap 2 (Calibration degradation, over/underconfidence profiles, and post-hoc Temperature Scaling on corrupted validation sets)
  ↓
Hypothesis Statement: CE produces overconfident errors; bounded robust losses (GCE, SCE) produce underconfidence; Temperature Scaling fitted on corrupted validation sets achieves sub-optimal calibration recovery compared to clean-tuned TS or in-training robust losses
  ↓
Observable Prediction: Uncalibrated CE has average confidence > accuracy; GCE has average confidence < accuracy on hard classes; TS tuned on noisy validation sets has significantly higher test ECE than TS tuned on clean validation sets
  ↓
Experiment: PreAct-ResNet18 on CIFAR-10 under Symmetric (η=0.5) and Asymmetric (η=0.4) noise, evaluating CE, GCE, SCE, Forward, Backward. Temperature Scaling fitted on 100% clean vs noise-corrupted validation sets
  ↓
Metric: 15-bin ECE, Adaptive ECE (AdaECE), Brier Score, and Overconfidence Bias E[P_hat - Acc | Y_hat ≠ Y]
  ↓
Statistical Test: Paired Wilcoxon signed-rank test comparing Test ECE(TS_corrupted) vs Test ECE(TS_clean) across 5 paired seeds with Holm-Bonferroni correction
  ↓
Falsification Condition: Test ECE(TS_corrupted) ≤ Test ECE(TS_clean) (p > 0.05), or CE produces underconfidence while bounded losses produce overconfidence
  ↓
Alignment Verdict: [ALIGNED]
```

---

### Hypothesis 4 (H4: Controlled Confounder Audit on Human Noise)
```
Research Question (Primary RQ & SRQ 4)
  ↓
Research Gap: Transfer gap from synthetic class-conditional noise to real human noise when controlling for data augmentation (MixUp)
  ↓
Hypothesis Statement: Controlled for identical MixUp data augmentation and backbone capacity, class-conditional transition matrix methods experience larger excess generalisation degradation on CIFAR-10N (Worst) than feature-cluster filtering (Confident Learning)
  ↓
Observable Prediction: On CIFAR-10N Worst noise, Confident Learning + MixUp achieves statistically higher clean test accuracy than Forward Correction + MixUp
  ↓
Experiment: PreAct-ResNet18 on CIFAR-10N (Clean, Aggregate, Worst) comparing CE, Forward Correction, Confident Learning, Forward+MixUp, and ConfidentLearning+MixUp
  ↓
Metric: Top-1 Clean Test Accuracy (%) and 15-bin ECE
  ↓
Statistical Test: Paired Wilcoxon signed-rank test on Test Accuracy (ConfidentLearning+MixUp vs Forward+MixUp) across 5 paired seeds
  ↓
Falsification Condition: Forward Correction + MixUp achieves accuracy ≥ Confident Learning + MixUp on CIFAR-10N Worst (p > 0.05, Hedges' g < 0.2)
  ↓
Alignment Verdict: [ALIGNED]
```

---

## 2. Summary Alignment Table

| Hypothesis | Theoretical Scope | Experimental Operationalization | Metric & Statistical Test | Alignment Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **H1 (Asymmetry)** | Linear surrogate risk boundary shift | Gaussian 2-class / 10-class linear classification | Hyperplane angle $\theta$, Spearman $r_s$, Wilcoxon | **[ALIGNED]** |
| **H2 (Capacity)** | Overparameterization & gradient dynamics | MLP width scaling & PreAct-ResNet18 | $\Delta \tau$ epoch duration, corrupted gradient norm | **[ALIGNED]** |
| **H3 (Calibration)** | Bidirectional miscalibration & validation split | PreAct-ResNet18 clean vs corrupted TS tuning | 15-bin ECE, AdaECE, Brier, Overconfidence bias | **[ALIGNED]** |
| **H4 (Human Noise)** | Confounder-controlled CIFAR-10N transfer | PreAct-ResNet18 on CIFAR-10N with MixUp | Top-1 Test Accuracy, paired Wilcoxon, Hedges' $g$ | **[ALIGNED]** |
