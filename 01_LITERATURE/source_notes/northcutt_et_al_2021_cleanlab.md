# Source Note: Northcutt et al. (JAIR 2021)

- **Title**: Confident Learning: Estimating Uncertainty in Dataset Labels
- **Authors**: Curtis G. Northcutt, Lu Jiang, Isaac L. Chuang
- **Venue**: Journal of Artificial Intelligence Research (JAIR 2021), Vol. 70, pp. 1373–1411
- **DOI / URL**: https://doi.org/10.1613/jair.1.12125

---

## 1. Problem Studied
Finding and cleaning label errors in real-world training datasets by estimating the joint distribution of observed noisy labels $\tilde{Y}$ and latent true labels $Y^*$, denoted $P(\tilde{Y}=j, Y^*=i)$, using out-of-sample predicted probabilities.

---

## 2. Key Contributions

### 2.1 Confident Joint Distribution Matrix $C_{\tilde{Y}, Y^*}$
Confident Learning uses cross-validated out-of-sample predicted probabilities $\hat{P}(\tilde{Y}=k \mid X=x)$ to construct an unnormalized counting matrix $C_{j, i}$:
$$C_{j, i} = |\{ x \in X_{\tilde{y}=j} : \hat{P}(\tilde{Y}=i \mid X=x) \ge t_i \}|$$
where $t_i$ is the per-class threshold defined as the expected predicted self-confidence:
$$t_i = \frac{1}{|X_{\tilde{y}=i}|} \sum_{x \in X_{\tilde{y}=i}} \hat{P}(\tilde{Y}=i \mid X=x)$$

### 2.2 Calibration & Transition Matrix Derivation
From the normalized joint distribution $\hat{Q}_{\tilde{Y}, Y^*}$, the noise transition matrix $T$ is recovered via Bayes rule:
$$\hat{T}_{ij} = \hat{P}(\tilde{Y}=j \mid Y^*=i) = \frac{\hat{Q}_{\tilde{Y}=j, Y^*=i}}{\sum_{k=1}^K \hat{Q}_{\tilde{Y}=k, Y^*=i}}$$

### 2.3 Noise-Pruned Learning
Identified label errors are pruned or re-weighted, yielding high performance on benchmarks like ImageNet, CIFAR, and Amazon Reviews (embodied in the Cleanlab package).

---

## 3. Relevance to Our Project
Confident Learning represents the primary modern alternative to anchor-point transition matrix estimation and is included as a core baseline in `04_EXPERIMENTS/`.
