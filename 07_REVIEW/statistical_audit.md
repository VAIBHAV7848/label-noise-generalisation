# Statistical Methodology Audit

This document audits the statistical validity of the experimental evaluation protocol and specifies rigorous standards.

---

## 1. Audit of the "5-Seed Multi-Run Protocol"

- **Critical Question**: Does running 5 random seeds automatically guarantee statistical significance?
- **Hostile Reviewer Verdict**: **NO**.
  1. With sample size $N=5$, standard parametric $t$-tests have low statistical power unless the effect size is massive (Cohen's $d > 1.5$).
  2. Normality assumptions cannot be verified with only 5 samples.
  3. Paired testing requires explicit, justified pairing: pairing is only valid if both methods are evaluated on the exact same dataset split, identical batch ordering, and identical noise corruption instance across each seed.

---

## 2. Statistical Protocol Requirements

To ensure statistical rigor:

1. **Exact Seed-by-Seed Pairing**:
   - For seed $s \in \{42, 1337, 2024, 7, 999\}$, generate the exact same corrupted dataset $\tilde{S}_s$ and evaluate all baselines on this identical dataset.
   - Compute the paired differences $\Delta_s = \text{Metric}_{\text{method}}(s) - \text{Metric}_{\text{baseline}}(s)$.
2. **Confidence Intervals**:
   - Report 95% Studentized or BCa Bootstrap Confidence Intervals for the mean difference $\mathbb{E}[\Delta_s]$.
3. **Non-Parametric Significance Testing**:
   - Use the **Wilcoxon Signed-Rank Test** for paired comparisons where normality is unverified.
4. **Multiple Comparison Correction**:
   - When comparing multiple baselines against the control method, apply the **Holm-Bonferroni correction** to control the Family-Wise Error Rate (FWER) at $\alpha = 0.05$.
5. **Effect Size**:
   - Report **Hedges' $g$** (bias-corrected Cohen's $d$ for small sample sizes $N=5$):
     $$g = \frac{\bar{x}_1 - \bar{x}_2}{s^*} \cdot \left( 1 - \frac{3}{4(n_1 + n_2) - 9} \right)$$
