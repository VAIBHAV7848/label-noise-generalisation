# Noise Transition Matrix Estimation & Error Bounds

This document formalizes the algorithms, mathematical guarantees, and error metrics for estimating the noise transition matrix $T \in [0, 1]^{K \times K}$.

---

## 1. Transition Matrix Definition & Role

The transition matrix $T$ encapsulates the class-conditional corruption distribution:
$$T_{ij} = P(\tilde{Y}=j \mid Y=i)$$
where $\sum_{j=1}^K T_{ij} = 1$ for all $i \in \{1, \dots, K\}$.

Estimating $T$ accurately is the primary prerequisite for:
1. **Backward Loss Correction**: $\ell_{\text{backward}} = T^{-1} \vec{\ell}(f(x))$
2. **Forward Loss Correction**: $\ell_{\text{forward}} = -\log ([T^\top f(x)]_{\tilde{y}})$

---

## 2. Estimation Paradigms

### 2.1 Anchor Point Estimation (Patrini et al., 2017)
Given a base classifier $\hat{p}(\tilde{Y}=j \mid x)$ trained on noisy data:
1. For each class $i \in \{1, \dots, K\}$, identify the candidate anchor point:
   $$x^i = \arg\max_{x \in \tilde{S}} \hat{p}(\tilde{Y}=i \mid x)$$
2. Estimate the $i$-th row of $T$ by reading off the full predicted posterior at $x^i$:
   $$\hat{T}_{ij} = \hat{p}(\tilde{Y}=j \mid x^i)$$
3. Normalize each row to ensure $\sum_{j=1}^K \hat{T}_{ij} = 1$:
   $$\hat{T}_{ij} \leftarrow \frac{\hat{T}_{ij}}{\sum_{k=1}^K \hat{T}_{ik}}$$

### 2.2 Dual-T Estimator without Anchor Points (Xia et al., 2019)
When perfect anchor points do not exist, the maximum posterior is upper-bounded by $T_{ii}$. Xia et al. estimate an intermediate transition matrix $T_e$ and solve a convex optimization problem with slack variables to recover $T$ directly from deep representations.

### 2.3 Confident Learning Joint Matrix (Northcutt et al., 2021)
Computes out-of-sample predicted probabilities via 5-fold cross-validation, evaluates per-class self-confidence thresholds $t_i$, computes the unnormalized confusion matrix $C_{\tilde{Y}, Y^*}$, and estimates $T$ via normalized Bayesian inversion.

---

## 3. Transition Matrix Evaluation Metrics

To strictly quantify the quality of estimated transition matrices, we compute:
1. **Frobenius Norm Error**:
   $$\mathcal{E}_{\text{Frob}}(\hat{T}, T) = \|\hat{T} - T\|_F = \sqrt{\sum_{i=1}^K \sum_{j=1}^K (\hat{T}_{ij} - T_{ij})^2}$$
2. **Spectral Norm Error**:
   $$\mathcal{E}_{\text{Spec}}(\hat{T}, T) = \|\hat{T} - T\|_2 = \sigma_{\max}(\hat{T} - T)$$
3. **Total Variation Distance (Row-wise mean)**:
   $$\mathcal{E}_{\text{TV}}(\hat{T}, T) = \frac{1}{2K} \sum_{i=1}^K \sum_{j=1}^K |\hat{T}_{ij} - T_{ij}|$$
