# REVIEWER 2: EXPERIMENTS & METHODOLOGY
**Confidence**: 5/5 (Expert in empirical methodology and robust deep learning)
**Overall Score**: 3/10 (Reject)

## Summary
The paper conducts an 84-run pilot study on CIFAR-10 using PreActResNet-18 to evaluate various robust losses and loss correction techniques across different label noise regimes. The authors emphasize the "corrupted validation trap" and the necessity of matrix inversion for asymmetric noise. Unfortunately, the empirical design is fatally flawed by severe statistical under-powering (N=3 seeds), narrow scope (one dataset, one architecture), and contradictory empirical results that undermine the theoretical claims.

## Strengths
1. **Pre-Registration**: The strict pre-registration of the 84-run grid and adherence to the protocol is highly commendable and rare in deep learning.
2. **Corrupted Validation Trap**: The empirical demonstration of how corrupted validation sets ruin Temperature Scaling is practically valuable and well-executed.

## Fatal Concerns (Must Fix)
1. **Catastrophic Statistical Under-Powering**: All inferential claims (e.g., "$+6.19\%$ gain, $p = 0.0019$") are based on $N=3$ random seeds. A paired $t$-test with $\text{df} = 2$ is virtually meaningless for establishing scientific truth in deep learning, where seed variance can be massive. The paper claims "decisive" results, but a single lucky seed could drive this significance. Claims of statistical significance must be stripped or the N must be increased to at least 10.
2. **The "True T" Paradox**: In Table 1, under Symmetric 50% noise, the ground-truth Forward Correction (True $T$) achieves $79.17\%$, while Generalized Cross-Entropy (GCE) achieves $82.40\%$. **This is a massive red flag.** True $T$ represents the theoretically optimal, unbiased risk estimator with perfect oracle knowledge of the noise. If an empirical heuristic like GCE beats the mathematical oracle by over $3\%$, it suggests either (a) the unbiasedness property (Proposition 1) is insufficient for generalization in deep networks, or (b) the implementation of Forward Correction is suboptimal (e.g., learning rate/weight decay not tuned for the corrected loss). The authors gloss over this fatal contradiction.
3. **Overclaimed "Generalisation"**: The title and abstract promise insights into "Generalisation." Yet the paper tests exactly ONE dataset (CIFAR-10) and ONE architecture (PreActResNet-18). This is not a study of generalization; it is a case study on CIFAR-10.
4. **No Real-World Noise (CIFAR-10N)**: The paper restricts itself entirely to synthetic, synthetic class-conditional noise. Real-world label noise is instance-dependent. The failure to include CIFAR-10N makes the claims about practical applicability highly suspect.

## Major Weaknesses
1. **Unfair Baseline Tuning**: The hyperparameters (30 epochs, SGD, lr=0.05, cosine annealing) appear to be a one-size-fits-all setup. It is highly likely that SCE fails catastrophically ($51\%$) simply because it requires a different learning rate or optimizer, not because the loss function is inherently broken.
2. **Peak vs Final Accuracy**: The paper makes a big deal about "memorization decay" (Acc_peak - Acc_final). However, in practice, no one deploys the final model if validation accuracy is dropping; they use early stopping. If early stopping via a clean validation set solves the CE memorization problem, the practical value of robust losses is diminished.

## Recommendation
**REJECT**. The empirical foundation ($N=3$, 1 dataset, 1 architecture) is far too weak to support the broad, decisive claims made in the manuscript. Furthermore, the underperformance of the oracle True $T$ baseline compared to GCE in symmetric noise reveals a fundamental gap in the narrative that is left unexplained.
