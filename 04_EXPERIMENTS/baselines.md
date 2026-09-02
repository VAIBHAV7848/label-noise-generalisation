# Baseline Methods & Categorization

To guarantee fair, comprehensive comparison, we benchmark against 10 established methods spanning four distinct methodological paradigms:

---

## 1. Uncorrected & Regularization Baselines

1. **Standard Cross-Entropy (CE)**:
   - Baseline empirical risk minimization without noise correction:
     $$\ell_{\text{CE}}(f(x), y) = -\log f_y(x)$$
   - *Role*: Measures baseline vulnerability to memorisation and miscalibration.

2. **Label Smoothing (LS)**:
   - Regularized cross-entropy with uniform label target mixture:
     $$\mathbf{y}_{\text{smooth}} = (1 - \alpha) \mathbf{y} + \frac{\alpha}{K} \mathbf{1}$$
   - *Role*: Tests whether simple confidence regularization mitigates label noise and restores calibration.

---

## 2. Robust Loss Function Baselines

3. **Mean Absolute Error (MAE / $L_1$ Loss)**:
   - Symmetric loss: $\ell_{\text{MAE}}(f(x), y) = 2 - 2 f_y(x)$.
   - *Role*: Theoretical noise-tolerance baseline under symmetric noise.

4. **Generalized Cross Entropy (GCE)** (Zhang & Sabuncu, NeurIPS 2018):
   - $L_q$ loss: $\ell_{\text{GCE}}(f(x), y) = \frac{1 - f_y(x)^q}{q}$ with $q=0.7$.
   - *Role*: State-of-the-art single-loss interpolation between MAE and CE.

5. **Symmetric Cross Entropy (SCE)** (Wang et al., ICML 2019):
   - Combines active cross entropy with passive reverse cross entropy: $\ell_{\text{SCE}} = \alpha \ell_{\text{CE}} + \beta \ell_{\text{RCE}}$ with $\alpha=0.1, \beta=1.0$.
   - *Role*: Gradient-balanced robust loss baseline.

---

## 3. Loss Correction & Matrix Estimators

6. **Backward Loss Correction** (Natarajan et al., NeurIPS 2013):
   - $\ell_{\text{backward}} = \hat{T}^{-1} \vec{\ell}(f(x))$.
   - *Role*: Unbiased risk estimation baseline.

7. **Forward Loss Correction** (Patrini et al., CVPR 2017):
   - $\ell_{\text{forward}} = -\log ([\hat{T}^\top f(x)]_{\tilde{y}})$.
   - *Role*: Standard deep-learning loss correction baseline with anchor-point estimation.

8. **Dual T-Estimator** (Xia et al., NeurIPS 2019):
   - Loss correction without anchor points via intermediate slack transition matrix estimation.
   - *Role*: State-of-the-art non-anchor transition matrix correction baseline.

---

## 4. Sample Selection & Semi-Supervised Learning

9. **Confident Learning (Cleanlab)** (Northcutt et al., JAIR 2021):
   - Estimates $P(\tilde{Y}, Y^*)$ via out-of-sample predicted probabilities, prunes noisy samples, and retrains model on clean subset.
   - *Role*: Modern industry standard for dataset cleaning and sample selection.

10. **DivideMix** (Li et al., ICLR 2020):
    - Fits 2-component Gaussian Mixture Model (GMM) on per-sample training losses to split dataset dynamically into clean (labeled) and noisy (unlabeled) sets, trained with MixMatch semi-supervised learning.
    - *Role*: Upper-bound SOTA semi-supervised baseline under extreme noise.
