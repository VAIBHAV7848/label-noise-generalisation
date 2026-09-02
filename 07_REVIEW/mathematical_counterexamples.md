# Mathematical Counterexamples Catalog

This document records the exact mathematical counterexamples constructed to audit and disprove incorrect theoretical propositions.

---

## Counterexample 1: Condition Number Does NOT Control Decision Boundary Displacement

- **Target Claim (Phase 0 H1)**: Decision boundary angular shift is strictly bounded by condition number $\kappa(T)$.
- **Counterexample Construction**:
  Let $K=2$ with balanced class priors $P(Y=1) = P(Y=2) = 0.5$.
  - **Matrix A (Symmetric)**: $T_{\text{sym}} = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}$.  
    Eigenvalues: $\lambda_1 = 1.0, \lambda_2 = 0.4$.  
    Condition Number: $\kappa(T_{\text{sym}}) = 1.0 / 0.4 = \mathbf{2.5}$.  
    Decision boundary: $P(\tilde{Y}=1 \mid x) = P(\tilde{Y}=2 \mid x) \iff 0.7 P(Y=1 \mid x) + 0.3 P(Y=2 \mid x) = 0.3 P(Y=1 \mid x) + 0.7 P(Y=2 \mid x) \iff P(Y=1 \mid x) = P(Y=2 \mid x)$.  
    **Observed Decision Boundary Shift**: **EXACTLY $0.0^\circ$ (Zero Displacement)**.
  
  - **Matrix B (Asymmetric)**: $T_{\text{asym}} = \begin{bmatrix} 1.0 & 0.0 \\ 0.6 & 0.4 \end{bmatrix}$.  
    Eigenvalues: $\lambda_1 = 1.0, \lambda_2 = 0.4$.  
    Condition Number: $\kappa(T_{\text{asym}}) = 1.0 / 0.4 = \mathbf{2.5}$.  
    Decision boundary: $P(\tilde{Y}=1 \mid x) = P(\tilde{Y}=2 \mid x) \iff 1.0 P(Y=1 \mid x) + 0.6 P(Y=2 \mid x) = 0.4 P(Y=2 \mid x) \iff 1.0 P(Y=1 \mid x) + 0.2 P(Y=2 \mid x) = 0$.  
    Since $P(Y \ge 0)$, this shifts the entire boundary to the extreme manifold boundary.  
    **Observed Decision Boundary Shift**: **MAXIMAL DISPLACEMENT**.
- **Conclusion**: Condition number $\kappa(T)$ is completely orthogonal to decision boundary displacement. Displacement is driven by matrix asymmetry $\|T - T^\top\|_F$.

---

## Counterexample 2: Proposition 4 ECE Lower Bound Disproven by Jensen Cancellation

- **Target Claim (Phase 0 Proposition 4)**: $\text{ECE}_{\tilde{\mathcal{D}}}(f) \ge \mathbb{E}_{\mathbf{p}}[|\max_k p_k - \max_k [T^\top \mathbf{p}]_k|]$.
- **Counterexample Construction**:
  Let $K=2$, $T = \begin{bmatrix} 0.9 & 0.1 \\ 0.8 & 0.2 \end{bmatrix}$.  
  Feature space has two points $x_1, x_2$ with $P(x_1) = P(x_2) = 0.5$:
  - $f(x_1) = \mathbf{p}_1 = [0.7, 0.3]^\top \implies \hat{Y}=0, \hat{P}=0.7$.  
    $P(\tilde{Y}=0 \mid \mathbf{p}_1) = 0.7(0.9) + 0.3(0.8) = 0.87$.  
    Error on $x_1$: $+0.17$.
  - $f(x_2) = \mathbf{p}_2 = [0.3, 0.7]^\top \implies \hat{Y}=1, \hat{P}=0.7$.  
    $P(\tilde{Y}=1 \mid \mathbf{p}_2) = 0.3(0.1) + 0.7(0.2) = 0.17$.  
    Error on $x_2$: $-0.53$.
- **Calculations**:
  - True Expected Accuracy given $\hat{P}=0.7$: $P(\tilde{Y}=\hat{Y} \mid \hat{P}=0.7) = 0.5(0.87) + 0.5(0.17) = 0.52$.
  - **True ECE**: $|0.52 - 0.70| = \mathbf{0.18}$.
  - **Proposed Lower Bound**: $0.5 |+0.17| + 0.5 |-0.53| = \mathbf{0.35}$.
- **Result**: $0.18 < 0.35$. The inequality $\text{ECE} \ge \mathbb{E}[|\dots|]$ is **FALSE**.
- **Cause**: Jensen's inequality for $|\mathbb{E}[Z]| \le \mathbb{E}[|Z|]$ causes errors of opposite signs for classes sharing the same confidence to cancel out.

---

## Counterexample 3: Determinant Collapse at Noise Thresholds

- **Target Assumption**: Unconstrained invertibility of transition matrices for arbitrary noise rates $\eta \in [0, 1)$.
- **Counterexample**:
  - For $K=10$ symmetric noise at $\eta = 0.90 = \frac{K-1}{K}$:
    $$T = \begin{bmatrix} 0.1 & 0.1 & \dots & 0.1 \\ 0.1 & 0.1 & \dots & 0.1 \\ \vdots & \vdots & \ddots & \vdots \\ 0.1 & 0.1 & \dots & 0.1 \end{bmatrix} \implies \text{Rank}(T) = 1, \quad \det(T) = 0$$
  - Inversion fails; $T^{-1}$ does not exist.
  - For $K=10$ asymmetric pair-flip noise at $\eta = 0.50$:
    $$T_{ii} = 0.50, \quad T_{i, (i+1)\bmod K} = 0.50 \implies \det(T) = 0$$
- **Rule Enforced**: Mathematical bounds and experiments must enforce $\eta < \frac{K-1}{K}$ and $\eta < 0.50$.
