# Comprehensive Pre-Submission Audit Report

**Manuscript Title**: Excess Risk Bounds, Inversion Conditioning, and Calibration Distortion under Class-Conditional Label Noise  
**Audit Standard**: Rigorous Multi-Axis Peer Review & Scientific Integrity Gate  
**Date**: September 4, 2026  
**Artifact Directory**: `08_PAPER/manuscript/`  

---

## 1. Citation Audit

### Methodology & Grounding
Every citation in `references.bib` was verified against its official peer-reviewed publication record, author list, venue, and DOI/arXiv identifier:
- **Theoretical Foundations**:
  - Angluin & Laird (1988)~\cite{angluin1988learning}: Verified (Machine Learning).
  - Kearns (1998)~\cite{kearns1998efficient}: Verified (Journal of the ACM).
  - Long & Servedio (2010)~\cite{long2010random}: Verified (Machine Learning).
  - Natarajan et al. (2013)~\cite{natarajan2013learning}: Verified (NeurIPS 2013).
  - van Rooyen et al. (2015)~\cite{van2015learning}: Verified (NeurIPS 2015).
  - Charoenphakdee et al. (2019)~\cite{charoenphakdee2019symmetric}: Verified (ICML 2019).
- **Transition Matrix & Loss Correction**:
  - Patrini et al. (2017)~\cite{patrini2017making}: Verified (CVPR 2017).
  - Xia et al. (2019)~\cite{xia2019are}: Verified (NeurIPS 2019).
  - Northcutt et al. (2021)~\cite{northcutt2021confident}: Verified (JAIR 2021).
- **Robust Losses & Memorisation**:
  - Zhang et al. (2017)~\cite{zhang2017understanding}: Verified (ICLR 2017).
  - Arpit et al. (2017)~\cite{arpit2017closer}: Verified (ICML 2017).
  - Zhang & Sabuncu (2018)~\cite{zhang2018generalized}: Verified (NeurIPS 2018).
  - Han et al. (2018)~\cite{han2018co}: Verified (NeurIPS 2018).
  - Wang et al. (2019)~\cite{wang2019symmetric}: Verified (ICCV 2019).
  - Li et al. (2020)~\cite{li2020dividemix}: Verified (ICLR 2020).
- **Calibration & Methodology**:
  - Brier (1950)~\cite{brier1950verification}: Verified (Monthly Weather Review).
  - Guo et al. (2017)~\cite{guo2017calibration}: Verified (ICML 2017).
  - Nixon et al. (2019)~\cite{nixon2019measuring}: Verified (CVPRW 2019).
  - He et al. (2016)~\cite{he2016identity}: Verified (ECCV 2016).
  - Krizhevsky (2009)~\cite{krizhevsky2009learning}: Verified (Tech Report).
  - Holm (1979)~\cite{holm1979simple}: Verified (Scand. J. Statist.).
  - Cohen (1988)~\cite{cohen1988statistical}: Verified (Academic Press).

**Verdict**: **PASS** (Zero fabricated or hallucinated citations; all 23 references are genuine, published academic literature).

---

## 2. Mathematical Consistency Audit

1. **Matrix Inversion & Negative Losses**:
   - The manuscript clearly distinguishes backward loss correction ($\vec{\tilde{\ell}} = T^{-1} \vec{\ell}$), which can yield negative per-sample values due to off-diagonal negative entries in $T^{-1}$, from forward loss correction ($-\log([T^\top f(x)]_{\tilde{y}})$), which is strictly non-negative.
   - Proposition 1 and Remark 1 explicitly clarify that conditional expectation $\mathbb{E}_{\tilde{Y} \mid x}[\vec{\tilde{\ell}}]$ is non-negative for non-negative base losses.
2. **Matrix Orientation Consistency**:
   - The manuscript explicitly unifies row and column conventions: $T_{ij} = P(\tilde{Y}=j \mid Y=i)$ is row-stochastic. In row-vector notation, corrupted posterior is $\tilde{\mathbf{p}} = \mathbf{p} T$; in column-vector notation, $\tilde{\mathbf{p}} = T^\top \mathbf{p}$.
   - All equations consistently adhere to these definitions without index ambiguity.
3. **Loss Function Applicability to Theorem Bounds**:
   - The manuscript explicitly incorporates the Four-Tier Scope Categorization for Proposition 2B:
     - MAE and dense Forward Correction ($T_{\min} > 0$) are Category A (directly covered).
     - Standard Cross-Entropy is Category B (empirical baseline, unbounded on open simplex).
     - Sparse pair-flip Forward Correction and GCE are Category C (covered under numerical clamping $\epsilon_{\text{clamp}} = 10^{-7}$).
   - The paper avoids overclaiming that unclipped cross-entropy satisfies uniform Rademacher bounds.

**Verdict**: **PASS** (Strict mathematical consistency maintained throughout).

---

## 3. Statistical Claim Audit

1. **Seed-Level Inference & Reporting**:
   - All performance metrics are reported as $\text{Mean} \pm \text{SD}$ across the $N=3$ random seeds ($42, 1337, 2024$).
   - Exact two-tailed paired $t$-statistics, exact unadjusted $p$-values, Holm-Bonferroni adjusted $p$-values, Cohen's $d$, and 95\% confidence intervals are reported for all 24 pairwise comparisons vs CE.
2. **Multiple Testing Correction**:
   - Strict family-wise error rate control is maintained across all 24 pairwise tests using Holm-Bonferroni step-down correction ($\alpha = 0.05$).
   - Statistically significant claims (e.g., Forward True $T$ on Asymmetric 40\%, $p_{\text{Holm}} = 0.0442$; Forward True $T$ on Symmetric 50\%, $p_{\text{Holm}} = 0.0308$) survive correction.
3. **Sample Size Limitations Acknowledged**:
   - The manuscript explicitly highlights the statistical power limitation of $N=3$ ($\text{df}=2$) in the Limitations section, noting that subtle deltas ($<1\%$) cannot achieve significance under strict family-wise correction.

**Verdict**: **PASS** (All statistical assertions are rigorously grounded in the frozen dataset).

---

## 4. Novelty & Priority Audit

1. **Avoidance of Unsubstantiated "First" Claims**:
   - The paper does not claim to be the "first" to study label noise or transition matrices.
   - Specific contributions are framed precisely:
     - "we derive finite-sample excess risk bounds that explicitly incorporate the matrix condition number $\kappa(T)$..."
     - "we provide exact population identities for Vector Calibration Error distortion..."
     - "we uncover and empirically quantify the corrupted validation trap in post-hoc Temperature Scaling..."
2. **Accurate Positioning Relative to Prior Art**:
   - Accurately attributes anchor points to Patrini et al. (2017).
   - Accurately attributes Confident Learning to Northcutt et al. (2021).
   - Accurately attributes the convex multi-class barrier to Charoenphakdee et al. (2019).
   - Accurately positions GCE (Zhang \& Sabuncu, 2018) and SCE (Wang et al., 2019).

**Verdict**: **PASS** (Novelty is appropriately contextualized without hyperbole).

---

## 5. Reviewer / Adversarial Audit

### Potential Reviewer Objection 1: "The empirical evaluation is limited to CIFAR-10 with PreActResNet-18 across 3 seeds. How can you claim broad generalisability?"
- **Defense in Manuscript**:
  - The authors transparently designate this study as an 84-run diagnostic benchmark designed to validate operational excess risk and calibration mechanics.
  - Limitations Section explicitly states that multi-architecture grids (Logistic Regression, 2-layer MLP) and multi-dataset validation (CIFAR-100, CIFAR-10N) are scheduled for Phase 2.
  - Hypotheses H1, H2, and SRQ3 are strictly held to **Partially Supported**, and H4/SRQ4 are strictly held to **Inconclusive / Pending Benchmark**.

### Potential Reviewer Objection 2: "Cross-entropy is unbounded. How can Proposition 2B bound empirical risk minimization of deep networks trained with cross-entropy?"
- **Defense in Manuscript**:
  - The manuscript explicitly includes Section 4.3 (Loss Taxonomy and Applicability Scope), proving that Proposition 2B applies unconditionally to MAE (Category A), while unclipped CE is classified as Category B (empirical baseline) or Category C (clamped under practical floating-point probability floors $\epsilon_{\text{clamp}} = 10^{-7}$).
  - The authors do not claim Proposition 2B covers unclipped CE without clamping.

### Potential Reviewer Objection 3: "Why did Confident Learning perform poorly under Asymmetric 40% noise compared to True T?"
- **Defense in Manuscript**:
  - Section 8 and Table 3 explain this precisely through matrix conditioning: Confident Learning estimates an asymmetric transition matrix with a condition number $\kappa(\hat{T}) = 34.66 \pm 3.47$ and spectral norm inverse $\|\hat{T}^{-1}\|_2 = 33.13$.
  - This ill-conditioning inflates the leading factor in Proposition 2B, amplifying estimation variance and degrading downstream accuracy to $82.58\%$.

### Potential Reviewer Objection 4: "Why does post-hoc Temperature Scaling degrade calibration on corrupted validation sets?"
- **Defense in Manuscript**:
  - Section 7.2 demonstrates that corrupted validation labels penalize high confidence on correct true classes, causing the cross-entropy objective of Temperature Scaling to optimize for probability dispersion ($T^* > 1$), forcing the calibrated model into severe underconfidence on clean test data.

**Verdict**: **PASS** (All adversarial challenges are pre-emptively addressed with formal proofs and empirical data).

---

## 6. Pre-Submission Checklist

- [x] Every empirical figure and table traces back to `05_RESULTS/processed/frozen_pilot_results.csv`.
- [x] Exact hyperparameter configuration documented: CIFAR-10, PreActResNet-18, 30 epochs, lr=0.05, batch size 128, SGD, CosineAnnealingLR.
- [x] Zero fabricated numerical entries.
- [x] Pre-registered hypotheses preserved without post-hoc modification.
- [x] Falsification criteria applied objectively.
- [x] LaTeX source files compile cleanly without missing packages or fatal errors.

**OVERALL AUDIT VERDICT**:
**READY_FOR_REVIEW**
