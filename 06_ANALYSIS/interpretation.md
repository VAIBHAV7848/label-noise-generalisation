# Scientific Interpretation & Mechanism Analysis

This document outlines the mechanistic principles that guide the interpretation of results across theory and experiments.

---

## 1. Why Do Overparameterized Classifiers Memorize Corrupted Labels?
- **Effective Capacity vs. Optimization Dynamics**: As shown by Zhang et al. (2017) and Arpit et al. (2017), deep networks possess sufficient Rademacher complexity to interpolate arbitrary random labels. However, because gradient descent on natural data prioritizes low-frequency spectral components first (the "spectral bias" of deep nets), simple, clean decision boundaries are learned rapidly during early epochs.
- **Gradient Norm & Residuals**: Noisy samples generate large persistent gradient residuals $-\nabla_{\theta} \log f_{\tilde{y}}(x)$. In later epochs, as clean samples achieve near-zero loss, the optimizer's remaining capacity is directed toward fitting the high-loss noisy anomalies.

---

## 2. Why Does Label Noise Distort Confidence Calibration?
- **Posterior Probability Warping**: Under class-conditional noise $T$, the training objective forces the network's softmax output $f(x)$ to approximate the corrupted posterior $\tilde{\eta}(x) = T^\top \vec{\eta}(x)$ rather than the true clean posterior $\vec{\eta}(x)$.
- **Confidence Over-Estimation**: Standard cross-entropy minimization pushes logits toward extreme values to drive training loss to zero, resulting in overconfident predictions even on ambiguous or mislabeled samples. When evaluated on clean ground-truth distributions, the model's confidence distribution is systematically unaligned with empirical accuracy, inflating ECE.

---

## 3. Why Does Synthetic Noise Fail to Model Real Human Error?
- **Instance Dependence ($P(\tilde{Y} \mid X, Y)$)**: In real-world annotations (CIFAR-10N), mislabeling probability is heavily concentrated on low-contrast, occluded, or boundary images. Synthetic class-conditional models assume every sample in class $i$ has identical corruption probability, ignoring feature geometry.
