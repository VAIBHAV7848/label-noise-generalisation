# Theoretical and Methodological Limitations

In accordance with academic rigor, all known limitations of the proposed theoretical derivations, empirical protocols, and scope are cataloged here.

---

## 1. Theoretical Limitations

1. **Class-Conditional Noise Assumption**:
   - The exact unbiasedness proof for backward loss correction ($\ell_{\text{backward}} = T^{-1} \vec{\ell}$) strictly relies on the conditional independence $P(\tilde{Y} \mid X, Y) = P(\tilde{Y} \mid Y)$. In settings where noise is strongly instance-dependent ($P(\tilde{Y} \mid X, Y)$), $T$ becomes a point-wise function of $x$, rendering a global $K \times K$ transition matrix an approximation.

2. **Anchor Point Identifiability**:
   - The theoretical identifiability of $T$ requires the existence of pure anchor points where $P(Y=i \mid X=x^i) = 1$. In datasets with severe class overlap or heavy label ambiguity, pure anchor points may not exist, introducing a non-vanishing lower bound on estimation error $\|\hat{T} - T\|_F$.

3. **Invertibility & Noise Rate Upper Bounds**:
   - Backward correction fails when $\det(T) = 0$ or when the condition number $\kappa(T)$ diverges. In symmetric noise, this occurs when $\eta \to \frac{K-1}{K}$; in pair-flip noise, this occurs when $\eta \ge 0.5$.

---

## 2. Empirical Limitations

1. **Vision Dominance in Benchmarks**:
   - While UCI Adult is included for tabular validation, the primary high-complexity benchmarks (CIFAR-10, CIFAR-100, CIFAR-10N, Animal-10N) are 2D image classification tasks. Findings may require further validation in natural language processing (e.g. LLM fine-tuning on noisy web scrapes) or graph learning.

2. **Computational Budget & Scaling Constraints**:
   - While ResNet-18 and ResNet-50 are evaluated across 5 random seeds, trillion-parameter Foundation Models / Vision Transformers (ViT-H) are beyond the current single-workstation compute budget.

3. **Single Human Noise Domain**:
   - CIFAR-10N represents crowdsourced image annotations from Amazon Mechanical Turk. While widely adopted, human noise patterns in professional domains (e.g. expert medical consensus in pathology or radiology) may exhibit different label error distributions.
