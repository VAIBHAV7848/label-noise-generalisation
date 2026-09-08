# ULTIMATE ADVERSARIAL RED-TEAM REVIEW
**Target**: *Excess Risk Bounds, Inversion Conditioning, and Calibration Distortion under Class-Conditional Label Noise*
**Verdict**: SCIENTIFICALLY_BLOCKED (Requires Major Revision before Submission)

## 1. Top 5 Strengths
1. **Pre-Registration Discipline**: The rigid adherence to a pre-registered 84-run grid prevents post-hoc cherry-picking.
2. **Condition Number Integration**: Highlighting $\kappa(T)$ as a dominant factor in excess risk successfully explains the failure of estimation methods like Confident Learning in asymmetric regimes.
3. **Identification of Corrupted Validation Trap**: Explicitly measuring the $\Delta \text{ECE}$ penalty of tuning calibrators on noisy labels provides a strong practical warning to the community.
4. **Rigorous Loss Taxonomy**: Acknowledging that standard unclipped CE does not satisfy the Rademacher complexity bounds without numerical clamping is theoretically mature.
5. **Divergent Calibration Analysis**: Showing that SCE becomes pathologically underconfident while CE becomes overconfident adds nuance to the robustness discussion.

## 2. Top 10 Weaknesses
1. **Statistically Meaningless Inference**: $N=3$ seeds ($\text{df}=2$) is insufficient for claiming statistical significance on differences like $+6.19\%$, especially in deep learning where initialization variance is high.
2. **The Oracle Paradox**: The unbiased oracle (True $T$) is beaten by a biased heuristic (GCE) by $>3\%$ in Symmetric 50%. This invalidates the premise that unbiasedness is the primary driver of performance.
3. **Theory-Practice Disconnect**: Proposition 2B assumes independent sample splitting for $\hat{T}$, but the code implements same-sample cross-validation (Confident Learning).
4. **Vacuous Complexity Bounds**: The Rademacher complexity of PreActResNet-18 is massive, rendering the finite-sample bound in Prop 2B numerically vacuous for the tested architecture.
5. **Unrealistic Perfect Calibration Assumption**: Theorem 4A relies on the base model being perfectly calibrated on clean data, which never holds in practice.
6. **Narrow Empirical Scope**: Claims about "generalisation" are based entirely on one dataset (CIFAR-10) and one architecture (PreActResNet-18).
7. **Lack of Real-World Noise**: The study exclusively uses synthetic noise, ignoring the instance-dependent nature of real human noise (e.g., CIFAR-10N).
8. **Strawman Calibration Setup**: Standard Temperature Scaling optimizes CE; tuning it on corrupted labels obviously fails. The paper doesn't test noise-robust temperature tuning.
9. **Overclaiming Novelty**: The abstract ambiguously groups prior theorems (convex symmetry barrier) with novel contributions.
10. **Circular Practical Advice**: Recommending practitioners to "Diagnose Noise Symmetry" is unhelpful if diagnosing symmetry requires the very transition matrix estimation that the practitioner is trying to avoid.

## 3. Top 5 Rejection Risks
1. **EXPERIMENTAL RISK (9/10)**: Reviewers will immediately attack the $N=3$ seed count and the restriction to a single architecture/dataset.
2. **THEORY RISK (8/10)**: Theory reviewers will reject the paper because the empirical estimation of $\hat{T}$ violates the independence assumption of Proposition 2B.
3. **LOGICAL RISK (8/10)**: The failure of True $T$ to beat GCE on symmetric noise undermines the core argument that unbiased loss correction is superior.
4. **NOVELTY RISK (6/10)**: Lack of comparison against modern (2022+) label noise baselines (e.g., contrastive learning, advanced sample selection) makes the paper feel dated.
5. **CALIBRATION RISK (5/10)**: The "corrupted validation trap" may be dismissed as a trivial consequence of applying a CE-based calibrator to noisy targets.

## 4. Required Experiments
- **Multi-Dataset Expansion**: Must execute on CIFAR-100 and CIFAR-10N (Human Noise).
- **Multi-Architecture Grid**: Must execute on varying capacities (e.g., Logistic Regression, simple CNN) to validate the capacity scaling claims in H2.
- **Increase N**: Must increase seeds to at least $N=5$ (preferably $N=10$) for reliable inference.
- **Robust Calibration Baseline**: Test Temperature Scaling fitted with a noise-robust objective (e.g., TS-GCE) on the corrupted validation set.

## 5. Required Mathematical Fixes
- **Sample-Splitting Caveat**: Explicitly state in the main text of Proposition 2B that it does not cover the Confident Learning implementation used in the paper, OR derive a uniform stability bound that does.
- **Explain True $T$ Failure**: Provide a formal mathematical hypothesis for why GCE outperforms True $T$ under symmetric noise (e.g., variance of the inverted loss estimator).

## 6. Required Writing Fixes
- Change the title. "Generalisation" is too broad for a CIFAR-10 study. Suggestion: *Empirical Risk Bounds and Calibration Distortion under Class-Conditional Label Noise*.
- Rewrite the abstract to explicitly state that the convex symmetry barrier is a prior result.
- Soften all language claiming "decisive" results based on $N=3$ statistics.

## 7. Required Citation Fixes
- Ensure Charoenphakdee et al. (2019) is explicitly cited *inside* the abstract or intro where the symmetry barrier is mentioned.
- Add comparisons/citations to modern (2022-2024) label noise methods in Related Work to show awareness of the current State-of-the-Art, even if they aren't benchmarked.

## 8. Final Recommendation
**DO NOT SUBMIT IN CURRENT STATE.** The manuscript has a strong core but contains fatal experimental limitations and theoretical disconnects that will trigger automatic rejection at top-tier venues (NeurIPS/ICLR/ICML). Execute Phase 2 (CIFAR-10N, larger N, multi-architecture) before submission.
