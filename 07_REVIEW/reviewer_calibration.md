# REVIEWER 3: CALIBRATION & LABEL NOISE
**Confidence**: 4/5 (Expert in uncertainty quantification and confidence calibration)
**Overall Score**: 6/10 (Borderline)

## Summary
This paper investigates the impact of class-conditional label noise on posterior probability calibration. The authors establish theoretical bounds on Vector Calibration Error (VCE) and empirically demonstrate that post-hoc Temperature Scaling is severely compromised when tuned on a corrupted validation set. The findings regarding the "corrupted validation trap" are the strongest and most novel aspect of the manuscript.

## Strengths
1. **Identifying the Corrupted Validation Trap**: Highlighting the $\Delta \text{ECE}$ penalty when calibrators are tuned on noisy labels is an excellent, highly practical contribution. Many papers assume access to a clean validation set, which is unrealistic.
2. **Divergent Calibration Profiles**: The analysis of how SCE becomes severely underconfident while CE becomes severely overconfident under noise provides useful intuition for practitioners.

## Fatal Concerns (Must Fix)
1. **Strawman Argument for Corrupted TS**: Tuning Temperature Scaling on a validation set with 50% symmetric noise is a known anti-pattern. The cross-entropy objective of TS explicitly penalizes the model for predicting the true class confidently if the validation label is flipped. The fact that this degrades calibration is mathematically trivial. The paper treats this as a "discovery," but it is arguably a strawman. A more robust comparison would be tuning TS using a robust loss (e.g., TS with GCE objective) on the corrupted validation set.
2. **Binning Artifacts in ECE**: The paper relies heavily on raw ECE (15 equal-width bins). ECE is notoriously sensitive to binning schemes and can be artificially minimized by underconfident models. While AdaECE is mentioned in the protocol, the main text leans heavily on raw ECE. Are the conclusions robust across binning strategies? The severe underconfidence of SCE might simply be exploiting the fixed binning structure differently than CE.

## Major Weaknesses
1. **Theorem 4B (Top-Label ECE Upper Bound)**: The exact equality for ECE distortion is only proven for symmetric noise with flip probability $\eta < (K-1)/K$. It does not generalize to asymmetric noise. The paper's claim that "label noise inherently induces probability miscalibration" is technically only strictly proven for the symmetric case.
2. **GCE's "Intrinsic Robustness"**: The paper praises GCE for maintaining low ECE without TS. However, this might be an artifact of GCE naturally acting as a regularizer that prevents logit explosion, rather than a fundamental property of noise robustness. Does GCE maintain good calibration on *clean* data? If GCE simply flattens logits universally, its "robustness" is just a side-effect of its functional form.

## Recommendation
**BORDERLINE**. The calibration analysis is the best part of the paper, but the reliance on a naive application of TS to corrupted data weakens the impact. Expanding the analysis to include robust validation tuning techniques would elevate the paper to an Accept.
