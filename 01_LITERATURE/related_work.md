# Related Work & Thematic Synthesis

The literature on learning under label noise spans four decades and can be structured into five fundamental thematic pillars:

---

## 1. Classical Statistical Learning Theory & Noise Tolerance

The theoretical foundations of label noise were established in early computational learning theory.
- **PAC Learning under Classification Noise**: Angluin & Laird (1988) first formalized the Random Classification Noise (RCN) model for binary concepts, proving that an algorithm can PAC-learn under noise rate $\eta < 1/2$ with sample complexity scaling as $\mathcal{O}\left(\frac{1}{(1-2\eta)^2}\right)$.
- **Statistical Query (SQ) Model**: Kearns (1998) established that any concept class learnable in the SQ model is PAC-learnable in the presence of classification noise, providing a general framework for noise-tolerant estimation.
- **Convexity Barriers**: Long & Servedio (2010) proved a foundational negative result: any boosting algorithm using a convex potential loss function fails under random classification noise on simple linearly separable data. This demonstrated that convex surrogate losses are fundamentally vulnerable to label noise without explicit correction.
- **Symmetric Loss Robustness**: Manwani & Sastry (2013), van Rooyen et al. (2015), and Charoenphakdee et al. (2019) analyzed conditions under which loss functions are inherently noise-tolerant. Specifically, a loss $\ell$ is symmetric if $\sum_{k=1}^K \ell(f(x), k) = C$ for all $x$. Under symmetric noise, minimizing empirical risk under a symmetric loss yields the Bayes optimal classifier. However, as Charoenphakdee et al. (2019) demonstrated, symmetric losses alone do not guarantee robustness under asymmetric / class-conditional noise.

---

## 2. Loss Correction & Noise Transition Matrix Estimation

When noise is class-conditional ($P(\tilde{Y}=j \mid Y=i) = T_{ij}$), loss functions can be corrected to restore risk unbiasedness.
- **Backward Loss Correction (Unbiased Estimator)**: Natarajan et al. (NeurIPS 2013) proved that if the transition matrix $T$ is invertible, the corrected loss:
  $$\ell_{\text{backward}}(f(x), \tilde{y}) = T^{-1} \vec{\ell}(f(x))$$
  satisfies $\mathbb{E}_{\tilde{Y} \mid Y} [\ell_{\text{backward}}(f(x), \tilde{Y})] = \ell(f(x), Y)$.
  Thus, empirical risk on corrupted labels is an unbiased estimator of true clean risk:
  $$\mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\ell_{\text{backward}}(f(X), \tilde{Y})] = \mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)]$$
- **Forward Loss Correction**: Patrini et al. (CVPR 2017) introduced forward loss correction for deep networks:
  $$\ell_{\text{forward}}(f(x), \tilde{y}) = \ell(T^\top f(x), \tilde{y})$$
  which avoids negative loss values and stabilizes SGD optimization.
- **Transition Matrix Estimation**:
  - *Anchor Points*: Patrini et al. (2017) estimated $T$ using "anchor points"—instances $x$ for which $P(Y=i \mid X=x) = 1$, allowing $T_{ij} \approx P(\tilde{Y}=j \mid X=x)$.
  - *Non-Anchor Estimators*: Xia et al. (NeurIPS 2019) relaxed the anchor point condition using slack variables and dual T-estimators. Northcutt et al. (JAIR 2021) formulated **Confident Learning**, directly estimating the joint distribution $P(\tilde{Y}, Y^*)$ using out-of-sample predicted probabilities.

---

## 3. Robust Loss Functions for Deep Networks

Because pure symmetric losses like Mean Absolute Error (MAE) suffer from vanishing gradients and severe underfitting during early training, hybrid robust losses were developed:
- **Generalized Cross Entropy (GCE)**: Zhang & Sabuncu (NeurIPS 2018) designed the $L_q$ loss:
  $$\ell_{\text{GCE}}(f(x), y) = \frac{1 - f_y(x)^q}{q}, \quad q \in (0, 1]$$
  interpolating between Cross Entropy ($q \to 0$) and MAE ($q = 1$).
- **Symmetric Cross Entropy (SCE)**: Wang et al. (ICML 2019) combined Cross Entropy with Reverse Cross Entropy (RCE):
  $$\ell_{\text{SCE}}(f(x), y) = \alpha \ell_{\text{CE}}(f(x), y) + \beta \ell_{\text{RCE}}(f(x), y)$$
  ensuring bounded gradients while maintaining robust optimization.
- **Active-Passive Loss (APL)**: Ma et al. (ICML 2020) proved that normalizing any loss makes it robust to symmetric noise and combined active and passive loss terms.

---

## 4. Memorisation Dynamics & Early Learning

Understanding how deep networks fit noisy labels revealed key structural phenomena:
- **Empirical Capacity & Random Labels**: Zhang et al. (ICLR 2017) demonstrated that overparameterized deep networks have sufficient capacity to fit 100% random labels with zero training error.
- **Early-Learning Phenomenon**: Arpit et al. (ICML 2017) showed that neural networks learn simple, generalizable patterns in early epochs before memorizing noisy labels in later epochs.
- **Sample Selection & Peer Filtering**:
  - *Co-teaching*: Han et al. (NeurIPS 2018) exploited early learning by training two networks simultaneously, with each network selecting its small-loss instances to update the other.
  - *DivideMix*: Li et al. (ICLR 2020) combined 2-component Gaussian Mixture Models (GMM) fitted on per-sample loss distributions with MixMatch semi-supervised learning.

---

## 5. Model Calibration & Real-World Noisy Benchmarks

- **Calibration in Deep Learning**: Guo et al. (ICML 2017) established that modern deep networks are severely miscalibrated, producing overconfident probabilities.
- **Calibration under Noise**: Thulasidasan et al. (NeurIPS 2019) and Cheng et al. (NeurIPS 2020) demonstrated that label noise exacerbates miscalibration, and standard regularizers like Mixup partially mitigate this.
- **Real-World Human Noise**: Wei et al. (ICLR 2022) introduced **CIFAR-10N** and **CIFAR-100N**, showing that human annotator noise contains instance-dependent corruptions that severely degrade algorithms optimized purely for synthetic class-conditional matrices.
