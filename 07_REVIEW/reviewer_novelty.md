# REVIEWER 4: NOVELTY & GENERAL MACHINE LEARNING
**Confidence**: 4/5 (General ML reviewer, broad knowledge of literature)
**Overall Score**: 4/10 (Weak Reject)

## Summary
The manuscript presents a unified theoretical and empirical study of empirical risk minimization under label noise, focusing on transition matrix ill-conditioning, excess risk bounds, and calibration. While the paper touches on several interesting axes, it struggles to articulate a singular, groundbreaking contribution, instead reading like a synthesis or extensive replication study of existing ideas (Patrini et al. 2017, Northcutt et al. 2021, Zhang & Sabuncu 2018).

## Strengths
1. **Holistic Synthesis**: Bringing together risk bounds, memorization dynamics, and calibration into a single framework is ambitious and provides a nice tutorial-like overview of the field's current state.
2. **Writing Quality**: The paper is well-written, logically structured, and easy to follow.

## Fatal Concerns (Must Fix)
1. **Incremental Novelty**: 
   - Unbiased loss correction (Prop 1) is Patrini 2017.
   - The convex symmetry barrier (Prop 3) is Charoenphakdee 2019.
   - Memorization dynamics (H2) is Arpit 2017.
   - What is strictly new? The inclusion of $\kappa(T)$ in the finite-sample bound (Prop 2) and the exact VCE identity (Theorem 4). However, adding matrix perturbation bounds to existing Rademacher complexity proofs is a standard exercise in statistical learning. The novelty threshold for a top-tier ML venue is not met.
2. **Overclaiming Contributions**: The abstract claims to establish the "fundamental noise tolerance barrier of convex multi-class surrogates." This is highly misleading, as Proposition 3 explicitly cites Charoenphakdee et al. (2019). The authors must clearly separate what they derived versus what they are citing.

## Major Weaknesses
1. **Missing Recent Literature**: The label noise literature has moved rapidly since 2021. The paper relies heavily on baselines from 2017-2019 (GCE, SCE, Forward Correction). Where are the comparisons to recent contrastive learning approaches for label noise (e.g., Sel-CL, 2022), or advanced sample selection methods like DivideMix (2020) and UNICON (2022)? Comparing only against early transition-matrix and robust-loss methods makes the empirical study feel dated.
2. **Actionable Guidelines are Trivial**: The discussion recommends "Diagnose Noise Symmetry Before Selecting Mitigation." How exactly is a practitioner supposed to diagnose noise asymmetry *without* first perfectly estimating the transition matrix? If they can estimate the matrix perfectly to diagnose asymmetry, they wouldn't need the heuristic. This guideline is practically circular.

## Recommendation
**WEAK REJECT**. The paper is a solid, rigorous replication and synthesis study, but it lacks the theoretical or empirical novelty required for a major conference. The overclaiming of prior theorems (e.g., Prop 3) and the lack of modern baselines (>2021) severely hurt its competitiveness.
