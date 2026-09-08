# REJECTION RISK SCORING

## Component Scores

| Axis | Score (out of 10) | Severity | Rationale |
| :--- | :---: | :--- | :--- |
| **THEORY RISK** | 8/10 | HIGH | Reviewers will spot that Proposition 2B assumes independent sample-splitting for matrix estimation, while the actual code (Confident Learning) uses cross-validation on the training set. This invalidates the application of the bound to the experiment. |
| **STATISTICAL RISK** | 9/10 | CRITICAL | $N=3$ random seeds is an automatic rejection at top venues for deep learning benchmark papers. The $p$-values derived from $df=2$ are highly brittle. |
| **EXPERIMENTAL RISK** | 9/10 | CRITICAL | Claims about "generalisation" are tested on a single dataset (CIFAR-10) and a single architecture. The "True $T$" oracle paradox (GCE beating True $T$) is left unexplained. |
| **NOVELTY RISK** | 6/10 | MODERATE | The paper feels like an excellent synthesis of 2017-2020 papers. It lacks comparison to modern (2022+) label noise literature. Overclaiming Proposition 3 (prior art) will irritate reviewers. |
| **WRITING RISK** | 3/10 | LOW | The manuscript is exceptionally well-written, cleanly formatted, and easy to follow. The writing itself is a major asset. |
| **REPRODUCIBILITY RISK** | 2/10 | LOW | The pre-registered manifest, rigorous artifact tracking, and clear parameter declarations make this highly reproducible. |

## OVERALL REJECTION RISK: 8.5 / 10
*(Very likely to be rejected in current state)*

## Top 3 Fatal Causes for Rejection:
1. **The $N=3$ Seed Limit**: A skeptical reviewer will instantly reject claims of "+6.19% gain" if the standard deviation across only 3 seeds could easily overlap with a different initialization.
2. **The Oracle Paradox**: A theory reviewer will demand to know why the mathematically unbiased oracle (True $T$) performs worse than GCE. If the theory says True $T$ recovers clean risk, the empirical failure of True $T$ invalidates the practical utility of the theory.
3. **The Sample-Splitting Violation**: A statistical learning theorist will reject the use of Proposition 2B to explain the Confident Learning collapse, as the estimation of $\hat{T}$ is coupled to the empirical risk minimization data.
