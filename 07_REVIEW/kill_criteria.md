# Kill Criteria & Project Pivot Triggers

To prevent sunk-cost fallacy, this document defines explicit, non-negotiable conditions under which specific research branches or the entire project must be paused, pivoted, or terminated.

---

## 1. Project-Level Kill Criteria (Immediate Pivot to Alternative Topic)

1. **Direct Proof Duplication**:
   - *Condition*: If a pre-existing published paper (e.g. from 2020–2026) already proved the exact operator-norm excess risk bound $\mathcal{E}(\hat{f}) \le \mathcal{O}(\sqrt{K} M \|T^{-1}\|_2 \|\hat{T} - T\|_F + \mathcal{R}_n(\mathcal{F}) / \sqrt{n})$ for multi-class loss correction under identical assumptions.
   - *Action*: Immediately drop the theoretical bound as a primary contribution and pivot to purely empirical calibration and validation-split repair.
2. **Benchmark Degeneracy**:
   - *Condition*: If all 10 baseline methods achieve statistically indistinguishable performance ($p > 0.5$, effect size $g < 0.2$) across both accuracy and calibration metrics on synthetic and human noise benchmarks.
   - *Action*: Terminate the benchmark study.

---

## 2. Hypothesis-Level Kill & Revision Criteria

1. **Kill Criterion for H1 (Condition Number $\kappa(T)$)**:
   - *Trigger*: Proven mathematically flawed via the symmetric matrix counterexample ($\kappa(T_{\text{sym}}) = \kappa(T_{\text{asym}})$ with 0 shift vs large shift).
   - *Action*: **KILLED AND REPLACED** with asymmetry-driven boundary shift formulation in `00_PROJECT/hypotheses.md`.
2. **Kill Criterion for H2 (Naive Early Learning Discovery)**:
   - *Trigger*: Identified as pre-existing established art (Arpit et al. 2017, Li et al. 2020).
   - *Action*: **REVISED** to focus on parameterization ratio $p/N$ and duration of clean plateau $\Delta \tau$.
3. **Kill Criterion for H3 (Universal Miscalibration Inflation)**:
   - *Trigger*: Contradicted by literature showing robust losses often underfit rather than overfit.
   - *Action*: **REVISED** to measure bidirectional miscalibration (overconfidence in CE vs underconfidence in bounded losses) and corrupted-validation calibration.
