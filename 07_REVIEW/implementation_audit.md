# Implementation & Code Integrity Audit

This document audits all source files in `src/` against exact mathematical definitions.

---

## 1. Audit of `src/losses/loss_correction.py`

### 1.1 Backward Loss Correction
- **Mathematical Specification**: $\ell_{\text{backward}}(f(x), j) = \sum_{k=1}^K [T^{-1}]_{jk} \ell(f(x), k)$.
- **Code Check**:
  ```python
  ell_vec = -log_probs  # (Batch, NumClasses)
  ell_corrected = torch.matmul(ell_vec, self.T_inv.t())
  loss_per_sample = ell_corrected.gather(1, targets.unsqueeze(1)).squeeze(1)
  ```
- **Verification**:
  Let $\mathbf{L} \in \mathbb{R}^{B \times K}$ with $L_{b, k} = \ell(f(x_b), k)$.
  $(\mathbf{L} (T^{-1})^\top)_{b, j} = \sum_{k=1}^K L_{b, k} [(T^{-1})^\top]_{kj} = \sum_{k=1}^K L_{b, k} [T^{-1}]_{jk}$.
  `gather(1, targets)` picks entry $j = \text{targets}_b$.
  Matches exact mathematical formula.
- **Numerical Stability**:
  When $T$ has condition number $\kappa(T) \gg 1$, $T^{-1}$ has large negative entries. $\ell_{\text{backward}}$ can produce negative numbers, which is mathematically expected for unbiased estimators (Natarajan et al. 2013).

### 1.2 Forward Loss Correction
- **Mathematical Specification**: $\ell_{\text{forward}}(f(x), j) = -\log \left( \sum_{i=1}^K T_{ij} f_i(x) \right) = -\log ([T^\top f(x)]_j)$.
- **Code Check**:
  ```python
  probs = F.softmax(logits, dim=-1)  # (Batch, K)
  p_corrupted = torch.matmul(probs, self.T)  # (Batch, K)
  log_p_corrupted = torch.log(p_corrupted)
  loss_per_sample = -log_p_corrupted.gather(1, targets.unsqueeze(1)).squeeze(1)
  ```
- **Verification**:
  `probs @ T`: $(B \times K) \times (K \times K)$ entry $j = \sum_{i=1}^K \text{probs}_i T_{ij} = [T^\top \text{probs}]_j$.
  Matches exact mathematical formula.

---

## 2. Audit of `src/losses/robust_losses.py`

1. **MeanAbsoluteErrorLoss**:
   - $\ell = 2.0 - 2.0 \cdot p_y$. Matches theory.
2. **GeneralizedCrossEntropyLoss**:
   - $\ell = (1 - p_y^q) / q$. Clamped with $\epsilon = 10^{-7}$ to avoid NaN gradients. Matches theory.
3. **SymmetricCrossEntropyLoss**:
   - $\ell = \alpha \ell_{\text{CE}} + \beta \ell_{\text{RCE}}$. Bounded by $\epsilon$ in log calculation. Matches theory.
4. **NormalizedCrossEntropyLoss**:
   - $\ell = \text{CE}_y / \sum_k \text{CE}_k$. Correct implementation of Ma et al. (2020).

---

## 3. Identified Implementation Flaws

1. **Estimator Discrepancy in `anchor_point.py`**:
   - The anchor search was partitioned by `noisy_labels == i`, which introduces bias if the anchor points were themselves noisy.
2. **Confident Learning Over-Counting**:
   - Multi-threshold counting in `confident_learning.py` allowed non-exclusive sample assignment.
