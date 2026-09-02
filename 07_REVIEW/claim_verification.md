# Claim Verification Checklist (Quality Gate)

Before any empirical or theoretical claim is incorporated into the research manuscript, it must pass this 8-point verification checklist.

---

## 1. Verification Checklist

- [ ] **Check 1 (Mathematical Rigor)**: All proofs are verified line-by-line without hand-waving steps or unstated assumptions.
- [ ] **Check 2 (Attribution Integrity)**: Prior theoretical results are cited with original author, venue, and year; no established theorem is claimed as our own.
- [ ] **Check 3 (Statistical Significance)**: All empirical claims are backed by 5 random seeds with reported mean $\pm$ standard deviation and $p < 0.05$ on paired statistical tests.
- [ ] **Check 4 (Ablation Completeness)**: Every proposed algorithmic module has an ablation demonstrating that removing it degrades performance.
- [ ] **Check 5 (Baseline Fairness)**: All baselines use identical backbone architectures, optimizer budgets, and data augmentations.
- [ ] **Check 6 (Real-World Validation)**: Claims of practical utility are validated on real crowdsourced human noise (CIFAR-10N / Animal-10N), not solely synthetic uniform noise.
- [ ] **Check 7 (Calibration Reporting)**: Every test accuracy table is accompanied by corresponding Expected Calibration Error (ECE) and Brier score metrics.
- [ ] **Check 8 (Reproducibility)**: Exact random seeds, environment lockfiles, and configuration YAMLs are committed and runnable out-of-the-box.
