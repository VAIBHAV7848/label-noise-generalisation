# Source Note: Zhang & Sabuncu (NeurIPS 2018)

- **Title**: Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels
- **Authors**: Zhilu Zhang, Mert R. Sabuncu
- **Venue**: Advances in Neural Information Processing Systems (NeurIPS 2018), Vol. 31
- **DOI / URL**: https://proceedings.neurips.cc/paper_files/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf

---

## 1. Problem Studied
The trade-off between Mean Absolute Error (MAE), which is theoretically robust to symmetric label noise but difficult to optimize with gradient descent, and Categorical Cross Entropy (CCE), which optimizes easily but memorizes noisy labels rapidly.

---

## 2. Key Contributions

### 2.1 Generalized Cross Entropy ($L_q$ Loss)
Defined via the negative Box-Cox transformation:
$$\mathcal{L}_q(f(x), y) = \frac{1 - f_y(x)^q}{q}, \quad q \in (0, 1]$$

- As $q \to 0$: $\lim_{q \to 0} \frac{1 - f_y(x)^q}{q} = -\log f_y(x) = \mathcal{L}_{\text{CE}}(f(x), y)$.
- At $q = 1$: $\mathcal{L}_1(f(x), y) = 1 - f_y(x) = \frac{1}{2} \mathcal{L}_{\text{MAE}}(f(x), y)$.

### 2.2 Gradient Dynamics & Truncation
The gradient with respect to model parameters $\theta$ is:
$$\frac{\partial \mathcal{L}_q(f(x), y)}{\partial \theta} = f_y(x)^q \frac{\partial \mathcal{L}_{\text{CE}}(f(x), y)}{\partial \theta}$$
The term $f_y(x)^q$ acts as an automatic sample weighting mechanism: when the model predicts low confidence on a noisy sample ($f_y(x) \approx 0$), its gradient is heavily downweighted by $f_y(x)^q$, preventing noisy label memorisation.

For extreme noise, Zhang & Sabuncu also introduced **Truncated GCE ($L_{\text{trunc}}$)** to zero out gradients for samples where $f_y(x) \le k$.

---

## 3. Limitations & Relevance
- **Hyperparameter Sensitivity**: The tuning of $q$ (typically $q=0.7$) depends strongly on the dataset and noise rate.
- **Calibration Impact**: By downweighting low-confidence samples, GCE distorts probability calibration (studied in our project as a primary metric).
