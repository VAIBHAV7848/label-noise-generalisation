# Catalog of Rejected Research Ideas & Dead Ends

To prevent circular research and avoid pursuing redundant or disproven directions, all rejected ideas, their initial appeal, and the literature evidence against them are formally recorded.

---

### Rejected Idea 1: Claiming Backward Loss Correction Unbiasedness as a Novel Contribution
- **Initial Appeal**: Clean, elegant mathematical derivation proving that noisy risk under $\tilde{\ell} = T^{-1} \vec{\ell}$ matches clean risk.
- **Why Rejected**: Already proven in full generality by Natarajan et al. (NeurIPS 2013). Presenting this as novel would constitute academic misconduct or fatal ignorance of literature.
- **Literature Evidence**: Natarajan, Dhillon, Ravikumar, Tewari, "Learning with Noisy Labels", NeurIPS 2013.
- **Corrected Direction**: Treat Proposition 1 as established baseline theorem; focus novel theoretical contribution on Proposition 2 (excess risk under estimation error $\|\hat{T} - T\|_F$).

---

### Rejected Idea 2: Proposing Pure Mean Absolute Error (MAE) as a Superior Loss Function
- **Initial Appeal**: MAE satisfies the symmetric loss condition $\sum_k \ell(f(x), k) = C$ and is theoretically 100% noise-tolerant under uniform symmetric noise.
- **Why Rejected**: MAE suffers from severe gradient vanishing ($|\partial \ell / \partial z| \to 0$ as $f_y(x) \to 1$), leading to catastrophic underfitting and inability to converge on datasets with more than 10 classes (e.g. CIFAR-100).
- **Literature Evidence**: Zhang & Sabuncu (NeurIPS 2018), Charoenphakdee et al. (ICML 2019).
- **Corrected Direction**: Benchmark MAE purely as a theoretical reference point; use GCE ($L_q$) and SCE to demonstrate practical gradient-balanced robustness.

---

### Rejected Idea 3: Restricting Empirical Scope to 2-Layer MLP on MNIST
- **Initial Appeal**: Low computational cost, fast execution, simple visualization of decision boundaries.
- **Why Rejected**: Highly saturated, toy-scale benchmark that yields no generalizable conclusions for modern computer vision, deep architectures, or real-world noise.
- **Literature Evidence**: Hundreds of papers since 1990 have tested shallow MLPs on MNIST noise.
- **Corrected Direction**: Include MNIST and 2-layer MLP solely as a low-capacity diagnostic sanity check, while grounding core conclusions in CIFAR-10, CIFAR-100, and CIFAR-10N using PreAct-ResNet18.

---

### Rejected Idea 4: Evaluating Exclusively on Top-1 Test Accuracy
- **Initial Appeal**: Standard, widely reported metric that allows direct copying of numbers from prior papers.
- **Why Rejected**: Top-1 accuracy masks dangerous overconfident miscalibration. A model with 85% accuracy and 35% ECE is unsafe for deployment in automated medical or safety systems.
- **Literature Evidence**: Guo et al. (ICML 2017), Thulasidasan et al. (NeurIPS 2019).
- **Corrected Direction**: Elevate Expected Calibration Error (ECE), Adaptive ECE, and Brier Score to primary evaluation metrics alongside accuracy.
