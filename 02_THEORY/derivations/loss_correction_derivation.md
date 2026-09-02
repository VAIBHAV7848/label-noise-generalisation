# Derivation: Forward & Backward Loss Correction

Here we derive the mathematical mechanics of Backward Loss Correction (Natarajan et al., 2013) and Forward Loss Correction (Patrini et al., 2017).

---

## 1. Backward Loss Correction (Unbiased Estimator)

### Objective
Construct a corrected surrogate loss vector $\vec{\tilde{\ell}}(f(x)) \in \mathbb{R}^K$ such that the expected noisy risk under $\vec{\tilde{\ell}}$ exactly equals the clean risk under the original loss $\vec{\ell}$:
$$\mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\tilde{\ell}(f(X), \tilde{Y})] = \mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)]$$

### Derivation
From the risk expansion derived in `risk_corruption_derivation.md`:
$$\mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\tilde{\ell}(f(X), \tilde{Y})] = \mathbb{E}_X [\vec{\eta}(X)^\top T \vec{\tilde{\ell}}(f(X))]$$
The clean risk is:
$$\mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)] = \mathbb{E}_X [\vec{\eta}(X)^\top \vec{\ell}(f(X))]$$

For this equality to hold for any distribution $\vec{\eta}(X)$, we require:
$$T \vec{\tilde{\ell}}(f(x)) = \vec{\ell}(f(x))$$

Assuming $T$ is invertible ($\det(T) \ne 0$), we left-multiply by $T^{-1}$:
$$\vec{\tilde{\ell}}(f(x)) = T^{-1} \vec{\ell}(f(x))$$

For a specific observed noisy label $\tilde{Y} = j$:
$$\ell_{\text{backward}}(f(x), j) = [\vec{\tilde{\ell}}(f(x))]_j = \sum_{k=1}^K [T^{-1}]_{jk} \ell(f(x), k)$$

---

## 2. Forward Loss Correction

### Motivation
While Backward Correction is theoretically unbiased, the matrix inverse $T^{-1}$ contains negative off-diagonal entries. This causes $\ell_{\text{backward}}(f(x), \tilde{y})$ to become negative, leading to non-convex, unstable gradient dynamics during deep network training.

### Derivation
Forward correction instead modifies the predicted probability distribution of the model $f(x) \in \Delta^{K-1}$.
Recall that the corrupted posterior probability is related to the true posterior by:
$$\vec{\tilde{\eta}}(x) = T^\top \vec{\eta}(x)$$

If the neural network outputs predicted class probabilities $f(x) \approx \vec{\eta}(x)$, the predicted noisy class probabilities are:
$$\tilde{f}(x) = T^\top f(x)$$

The **Forward Corrected Loss** computes the standard cross-entropy with respect to the noise-corrupted model output $\tilde{f}(x)$:
$$\ell_{\text{forward}}(f(x), \tilde{y}) = \ell_{\text{CE}}(T^\top f(x), \tilde{y}) = -\log \left( [T^\top f(x)]_{\tilde{y}} \right) = -\log \left( \sum_{i=1}^K T_{i, \tilde{y}} f_i(x) \right)$$

### Properties of Forward Correction
1. **Non-negativity**: Since $T_{i, \tilde{y}} \ge 0$ and $f_i(x) \ge 0$ with $\sum_i T_{i, \tilde{y}} f_i(x) \le 1$, $\ell_{\text{forward}} \ge 0$ always.
2. **Smooth Gradients**: Backpropagation flows naturally through the linear transformation layer $T^\top$.
