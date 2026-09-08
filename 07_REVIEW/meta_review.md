# AREA CHAIR META-REVIEW

## Overall Recommendation: REQUIRES MAJOR REVISION (Weak Reject / Borderline)

## Meta-Review Summary
This manuscript tackles an important intersection of problems in learning with noisy labels: the coupling of transition matrix estimation error, matrix conditioning, early memorization, and confidence calibration. The authors present a highly disciplined, pre-registered 84-run empirical pilot and accompanying theoretical bounds. 

However, the reviewers unanimously agree that while the writing and synthesis are excellent, the paper suffers from critical disconnects between its theoretical claims and empirical execution, alongside severe statistical limitations. The current manuscript cannot be accepted in its present form.

## Key Strengths to Preserve
1. **The "Corrupted Validation Trap"**: The empirical demonstration of how Temperature Scaling collapses when tuned on noisy validation sets is practically highly valuable.
2. **Focus on Matrix Conditioning ($\kappa(T)$)**: Formalizing why methods like Confident Learning fail under asymmetric noise due to matrix ill-conditioning is an insightful contribution.
3. **Rigorous Taxonomy**: The four-tier loss taxonomy categorizing applicability to uniform bounds is theoretically sound and honest.

## Fatal Flaws (Must be addressed for acceptance)
1. **Empirical Underpowering ($N=3$ / 1 Dataset / 1 Architecture)**: The claim of understanding "generalisation" cannot rest on a single CIFAR-10 / ResNet-18 benchmark with 3 seeds. Seed variance makes the p-values highly brittle. (Reviewer 2)
2. **Theory/Practice Disconnect on Sample Splitting**: Proposition 2B strictly assumes independent estimation of $\hat{T}$. The empirical implementation uses same-sample estimation (cross-validation on the training set). The theory does not cover the experiments. (Reviewer 1)
3. **The True $T$ Oracle Paradox**: The ground-truth oracle (True $T$) underperforms the heuristic GCE on Symmetric 50\% noise ($79.17\%$ vs $82.40\%$). If True $T$ is the theoretically optimal unbiased estimator, this massive performance gap fundamentally contradicts the theoretical narrative and is left unexplained. (Reviewer 2)
4. **Novelty Overclaiming**: The abstract and intro imply the derivation of the convex symmetry barrier, which is a prior result (Charoenphakdee 2019). The boundary between novel derivations and literature review must be clarified. (Reviewer 4)
5. **Strawman Calibration Baseline**: Applying standard Cross-Entropy Temperature Scaling to a 50% noisy validation set is a known failure mode. The authors must test a robust validation tuning method (e.g., tuning TS with a noise-robust objective) to prove the trap is inescapable. (Reviewer 3)

## Minimum Revision Requirements for Next Submission
- **Expand Empirical Scope**: Execute the planned Phase 2 experiments (CIFAR-100, CIFAR-10N, varying capacities, $N \ge 5$ seeds).
- **Resolve True $T$ Paradox**: Provide a concrete mathematical or empirical explanation for why GCE outperforms the exact True $T$ oracle under symmetric noise (e.g., hyperparameter sub-optimality or optimization difficulties of the inverted loss landscape).
- **Tone Down "Firsts"**: Strip out language implying theoretical derivations of prior work.
- **Bridge the Theory Gap**: Either implement true sample-splitting in the empirical pipeline (e.g., estimate $\hat{T}$ on a held-out noisy set) or extend Proposition 2B to cover same-sample algorithmic stability.
