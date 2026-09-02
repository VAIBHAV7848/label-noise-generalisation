# Numerical Stability & Ill-Conditioning Audit

This document quantifies the numerical behavior of transition matrix inversion, condition numbers, and loss magnitudes across noise regimes.

---

## 1. Condition Number Scaling $\kappa(T)$ and Singular Thresholds

For a transition matrix $T \in [0, 1]^{K \times K}$:
$$\kappa(T) = \|T\|_2 \|T^{-1}\|_2 = \frac{\sigma_{\max}(T)}{\sigma_{\min}(T)}$$

### 1.1 Symmetric Noise Scaling ($K=10$)
Under symmetric noise with flip rate $\eta$:
$$T = (1-\eta) I + \frac{\eta}{K-1} (\mathbf{1}\mathbf{1}^\top - I)$$
The eigenvalues are:
- $\lambda_1 = 1.0$ (multiplicity 1, eigenvector $\mathbf{1}$)
- $\lambda_2 = \dots = \lambda_K = 1 - \frac{K}{K-1} \eta$ (multiplicity $K-1$)

Singular Threshold: $\lambda_2 = 0 \iff \eta = \frac{K-1}{K} = 0.90$.

| Noise Rate $\eta$ | Condition Number $\kappa(T)$ | Inverse Norm $\|T^{-1}\|_2$ | Minimum Entry of $T^{-1}$ |
| :--- | :--- | :--- | :--- |
| $\eta = 0.10$ | 1.13 | 1.13 | -0.01 |
| $\eta = 0.30$ | 1.50 | 1.50 | -0.05 |
| $\eta = 0.50$ | 2.25 | 2.25 | -0.13 |
| $\eta = 0.70$ | 4.50 | 4.50 | -0.35 |
| $\eta = 0.80$ | 9.00 | 9.00 | -0.80 |
| $\eta = 0.89$ | 81.82 | 81.82 | -8.08 |
| $\eta = 0.90$ | $\mathbf{\infty}$ **(Singular)** | $\mathbf{\infty}$ | $\mathbf{-\infty}$ |

### 1.2 Asymmetric Noise Scaling (CIFAR-10 Pair Flips)
Under asymmetric pair flips ($i \to (i+1) \bmod K$) with rate $\eta$:
Singular Threshold: $\eta = 0.50$ (where class $i$ and class $i+1$ become mutually indistinguishable).

| Noise Rate $\eta$ | Condition Number $\kappa(T)$ | Inverse Norm $\|T^{-1}\|_2$ | Minimum Entry of $T^{-1}$ |
| :--- | :--- | :--- | :--- |
| $\eta = 0.10$ | 1.28 | 1.25 | -0.12 |
| $\eta = 0.20$ | 1.74 | 1.67 | -0.33 |
| $\eta = 0.30$ | 2.69 | 2.50 | -0.75 |
| $\eta = 0.40$ | 5.54 | 5.00 | -2.00 |
| $\eta = 0.48$ | 28.18 | 25.00 | -12.00 |
| $\eta = 0.50$ | $\mathbf{\infty}$ **(Singular)** | $\mathbf{\infty}$ | $\mathbf{-\infty}$ |

---

## 2. Gradient Explosion under Ill-Conditioned Inversion

The gradient of the backward corrected loss with respect to logits $\mathbf{z}$ is:
$$\nabla_{\mathbf{z}} \ell_{\text{backward}} = \sum_{k=1}^K [T^{-1}]_{\tilde{y}, k} \nabla_{\mathbf{z}} \ell(\mathbf{p}, k)$$
When $\kappa(T) \gg 10^3$, $|[T^{-1}]_{\tilde{y}, k}| > 10^3$, which scales the gradient magnitude by orders of magnitude, destabilizing SGD with standard learning rates.

---

## 3. Mandatory Experimental Constraints

1. **Explicit Invertibility Gate**: Reject any estimated matrix $\hat{T}$ with $\kappa(\hat{T}) > 10^4$ or $\det(\hat{T}) \le 10^{-6}$.
2. **Noise Rate Ceiling**:
   - Synthetic Symmetric: $\eta \le 0.70 < \frac{K-1}{K}$.
   - Synthetic Asymmetric: $\eta \le 0.40 < 0.50$.
3. **Loss Clipping**: In Backward Loss, clamp large negative loss values to $-10.0$ to prevent floating-point underflow/overflow during backpropagation.
