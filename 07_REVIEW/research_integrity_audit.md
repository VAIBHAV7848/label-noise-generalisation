# Research Integrity Audit Report (Phase 0 + Phase 1)

---

## 1. Executive Verdict
**Verdict**: **CONDITIONAL GO**
The research project possesses a sound, mathematically grounded core and a clean modular implementation; however, it required substantial correction of overstated claims, mathematical constant refinement in Proposition 2, fixing selection bias in the anchor-point estimator, replacing the flawed condition-number hypothesis (H1), and drastically pruning the combinatorial scope from 16,200 runs to a 300-run Minimum Decisive Experiment Set (MDES).

---

## 2. What Survives the Audit
1. **Mathematical Proposition 1**: Backward loss correction is an exact unbiased estimator of clean risk under known invertible transition matrices.
2. **Refined Proposition 2**: Non-asymptotic excess risk bound linear in $\sqrt{K} M \|T^{-1}\|_2 \|\hat{T} - T\|_F$.
3. **Core Modular Implementation (`src/`)**: Clean PyTorch implementations of standard/robust losses (CE, LS, MAE, GCE, SCE, NCE), loss correction (Forward/Backward), PreAct-ResNet18, and calibration metrics (15-bin ECE, AdaECE, Brier score).
4. **Unit Test Framework (`tests/`)**: 19 deterministic tests validating loss symmetry, backward unbiasedness, and metric calculation.
5. **Real-World CIFAR-10N Benchmark Integration**: External validation on crowdsourced human noise.

---

## 3. What Fails the Audit
1. **Original Hypothesis H1**: Claiming condition number $\kappa(T)$ governs decision boundary shift was disproven by counterexample (symmetric vs asymmetric matrices with identical $\kappa(T)$).
2. **Original Research Gap B Claims**: Claiming calibration under label noise is "unexplored" failed literature audit (Wang et al. NeurIPS 2021 and Bai et al. ICLR 2021 already investigated this).
3. **Anchor Point Estimator Implementation**: Filtering candidate anchor points strictly by `noisy_labels == i` introduced label corruption selection bias.
4. **Combinatorial Scope Explosion**: 16,200 planned runs across 6 datasets and 6 model architectures was unfeasible and redundant.
5. **Phase 0 Proposition 2 Constant**: The bound incorrectly stated $\|T^{-1}\|_2^2 \epsilon$ instead of the exact derived first-order term $\sqrt{K} M \|T^{-1}\|_2 \epsilon$.

---

## 4. Literature Claims Requiring Correction
- Anchor-point selection in Patrini et al. (2017) operates over the **entire dataset** $\mathcal{S}$, not partitioned by observed noisy labels.
- Confident Learning (Northcutt et al. 2021) requires out-of-sample cross-validated probabilities and single-assignment sample thresholding.

---

## 5. Research Gaps Confirmed
- **Confirmed Gap 1**: Non-asymptotic excess risk propagation under finite-sample transition matrix Frobenius estimation error $\|\hat{T} - T\|_F$ for multi-class classifiers.
- **Confirmed Gap 2**: Calibration recovery performance under post-hoc Temperature Scaling when tuned on a corrupted validation split vs clean validation split across loss correction paradigms.

---

## 6. Research Gaps Rejected
- **Rejected Gap 1 (Memorisation Discovery)**: Early-learning and noise memorisation in deep nets are completely solved (Zhang 2017, Arpit 2017, Li 2020).
- **Rejected Gap 2 (First Study on Calibration)**: Contradicted by Wang et al. (2021) and Bai et al. (2021).
- **Rejected Gap 3 (Discovery of Human Noise Failure)**: Solved by Wei et al. (2022).

---

## 7. Hypotheses That Survive
- None survived in their initial naive form; all 4 required substantial sharpening into falsifiable, confounder-controlled statements.

---

## 8. Hypotheses That Must Change
- **H1**: Replaced from condition number $\kappa(T)$ to **noise asymmetry** $\|T - T^\top\|_F$.
- **H2**: Revised from existence of memorisation to **capacity parameterization ratio $p/N$ and duration of clean generalisation window $\Delta \tau$**.
- **H3**: Revised from universal overconfidence to **bidirectional miscalibration profiles (CE overconfidence vs symmetric loss underconfidence) and corrupted validation tuning**.
- **H4**: Revised to **strictly control for data augmentation (MixUp) and backbone capacity**.

---

## 9. Mathematical Results Confirmed
- **Proposition 1** (Unbiasedness): Exact analytical equality $\mathbb{E}_{\tilde{\mathcal{D}}}[\tilde{\ell}] = \mathbb{E}_{\mathcal{D}}[\ell]$.
- **Proposition 3** (Convex Multi-Class Loss Non-Symmetry): Correct (Charoenphakdee 2019).
- **Proposition 4** (Posterior Simplex Distortion): Valid lower bound on ECE under corrupted posteriors.

---

## 10. Mathematical Results Rejected / Corrected
- **Proposition 2**: Corrected from $2 M \sqrt{K} \|T^{-1}\|_2^2 \epsilon$ to:
  $$\mathcal{E}(\hat{f}) \le \frac{2 \sqrt{K} M \|T^{-1}\|_2 \|\hat{T} - T\|_F}{1 - \|T^{-1}\|_2 \|\hat{T} - T\|_F} + \frac{4 L_{\ell} \|\hat{T}^{-1}\|_2}{\sqrt{n}} \mathcal{R}_n(\mathcal{F}) + \mathcal{O}\left(\frac{\|\hat{T}^{-1}\|_2}{\sqrt{n}}\right)$$

---

## 11. Implementation Bugs Found & Fixed
1. `src/losses/robust_losses.py`: Fixed invalid escape sequence in docstring.
2. `src/estimators/anchor_point.py`: Fixed anchor point selection to search across all instances in the dataset rather than pre-filtering by noisy label.
3. `src/estimators/confident_learning.py`: Fixed multi-assignment counting to enforce single-class assignment per sample.

---

## 12. Estimator Issues
- Dual-T estimator in `src/estimators/dual_t.py` is documented as a lightweight heuristic approximation of Xia et al. (2019).

---

## 13. Experimental Design Issues
- Pruned 16,200 combinatorial runs down to the 300-run Minimum Decisive Experiment Set (MDES) on CIFAR-10, CIFAR-100, and CIFAR-10N.

---

## 14. Statistical Issues
- Standardized exact seed-by-seed pairing across seeds `[42, 1337, 2024, 7, 999]`, Wilcoxon signed-rank testing, Holm-Bonferroni correction, and Hedges' $g$ effect size reporting.

---

## 15. Strongest Remaining Research Question
> *"Under what exact theoretical bounds and empirical conditions do loss-correction methods and robust losses maintain probability calibration and excess risk guarantees under imperfect transition matrix estimation and corrupted validation distributions across varying model capacities?"*

---

## 16. Strongest Potential Contribution
A unified theoretical and empirical characterization connecting **finite-sample transition matrix estimation error $\|\hat{T} - T\|_F$** to **excess risk bounds** and **corrupted-validation calibration recovery (ECE)**.

---

## 17. Minimum Decisive Experiment Set (MDES)
- **Datasets**: CIFAR-10 (Symmetric $\eta \in \{0.2, 0.5\}$, Asymmetric $\eta \in \{0.2, 0.4\}$) + CIFAR-10N (Human: Aggregate, Worst) + CIFAR-100 ($\eta=0.4$).
- **Models**: PreAct-ResNet18 + TwoLayerMLP.
- **Methods**: Cross-Entropy, Label Smoothing, GCE ($q=0.7$), SCE, Forward Correction, Backward Correction, Confident Learning.
- **Metrics**: Top-1 Accuracy, 15-bin ECE, Brier Score, $\|\hat{T} - T\|_F$, Clean vs Corrupted Validation ECE Delta.
- **Budget**: Exactly 300 runs.

---

## 18. Kill Criteria
Defined in `07_REVIEW/kill_criteria.md`: Immediate pivot if Proposition 2 bound is duplicated by existing literature or if all baselines demonstrate statistically indistinguishable calibration performance ($p > 0.5$).

---

## 19. Required Changes Before Experiments
1. Update `00_PROJECT/hypotheses.md` with revised H1–H4.
2. Update `00_PROJECT/research_question.md` with refined focus.
3. Update `02_THEORY/propositions.md` and `02_THEORY/theory_summary.md` with corrected Proposition 2 bound.
4. Update `04_EXPERIMENTS/experimental_protocol.md` and `09_REPRODUCIBILITY/config/default_config.yaml` with MDES specification.
5. Fix estimator implementations in `src/estimators/`.

---

## 20. Final GO / NO-GO Decision
**DECISION: CONDITIONAL GO**
Proceed to Phase 2 (Pilot validation & MDES experiments) upon completing and verifying the required updates.
