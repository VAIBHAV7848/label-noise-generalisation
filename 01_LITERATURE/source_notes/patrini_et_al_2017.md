# Source Note: Patrini et al. (CVPR 2017)

- **Title**: Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach
- **Authors**: Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, Lizhen Qu
- **Venue**: IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017), pp. 1944–1952
- **DOI / URL**: https://doi.org/10.1109/CVPR.2017.210

---

## 1. Problem Studied
Training deep neural networks under class-conditional label noise $T_{ij} = P(\tilde{Y}=j \mid Y=i)$ where the noise transition matrix $T$ is unknown and must be estimated from noisy training data.

---

## 2. Key Contributions

### 2.1 Forward and Backward Loss Correction for Deep Networks
- **Backward Correction**:
  $$\ell_{\text{backward}}(f(x), \tilde{y}) = T^{-1} \vec{\ell}(f(x))$$
- **Forward Correction**:
  $$\ell_{\text{forward}}(f(x), \tilde{y}) = \ell(T^\top f(x), \tilde{y}) = -\log \left( [T^\top f(x)]_{\tilde{y}} \right) = -\log \left( \sum_{i=1}^K T_{i, \tilde{y}} f_i(x) \right)$$
  Forward correction alters the predicted softmax probability distribution by multiplying by $T^\top$, avoiding negative losses and maintaining numerical stability in SGD.

### 2.2 Transition Matrix Estimation via Anchor Points
Under the **Anchor Point Assumption**, for each class $i \in \{1, \dots, K\}$, there exists an instance $x^i \in \mathcal{X}$ such that $P(Y=i \mid X=x^i) = 1$. Then:
$$P(\tilde{Y}=j \mid X=x^i) = \sum_{k=1}^K P(\tilde{Y}=j \mid Y=k) P(Y=k \mid X=x^i) = T_{ij}$$
Therefore, given a deep network trained on noisy data producing predicted posterior $\hat{p}(\tilde{Y}=j \mid x)$:
$$\hat{T}_{ij} = \hat{p}(\tilde{Y}=j \mid x^i) = \arg\max_{x \in \mathcal{D}} \hat{p}(\tilde{Y}=i \mid x)$$

---

## 3. Assumptions & Limitations
- **Anchor Point Availability**: In high-dimensional complex datasets (e.g. CIFAR-100), pure anchor points where $P(Y=i \mid X=x)=1$ rarely exist or are difficult to identify due to overlapping class distributions.
- **Overfitting Prior to Estimation**: The network used to estimate $T$ may have already memorized noisy labels, distorting $\hat{T}$.

---

## 4. Relevance to Our Project
Patrini et al. provides the standard forward/backward correction baseline and anchor-point transition matrix estimator that we analyze, implement, and benchmark against non-anchor estimators (Xia et al. 2019) and confident learning (Northcutt et al. 2021).
