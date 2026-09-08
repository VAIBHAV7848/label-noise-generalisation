# Table 3: Transition Matrix Estimation Quality and Inversion Condition Numbers

| Noise Regime | Estimation Method | Frobenius Error ||T_hat - T||_F | Condition Number kappa(T) | ||T_hat^-1||_2 | Test Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Clean | Forward (True T) | 0.0000 ± 0.0000 | 1.00 ± 0.00 | 1.00 | 91.64 $\pm$ 0.12 |
| Clean | Forward (Anchor T) | 0.1952 ± 0.0733 | 1.23 ± 0.11 | 1.22 | 91.62 $\pm$ 0.15 |
| Clean | Forward (ConfLearning T) | 1.0047 ± 0.0203 | 2.26 ± 0.12 | 2.24 | 91.19 $\pm$ 0.17 |
| Clean | Forward (Perturbed T) | 1.5000 ± 0.0000 | 2.00 ± 0.00 | 2.00 | 90.92 $\pm$ 0.29 |
| Symmetric 20% | Forward (True T) | 0.0000 ± 0.0000 | 1.29 ± 0.00 | 1.29 | 87.37 $\pm$ 0.31 |
| Symmetric 20% | Forward (Anchor T) | 0.3435 ± 0.0205 | 1.95 ± 0.12 | 1.93 | 86.61 $\pm$ 0.48 |
| Symmetric 20% | Forward (ConfLearning T) | 0.9626 ± 0.0249 | 3.50 ± 0.23 | 3.48 | 87.91 $\pm$ 0.17 |
| Symmetric 20% | Forward (Perturbed T) | 1.1667 ± 0.0000 | 2.57 ± 0.00 | 2.57 | 88.63 $\pm$ 0.07 |
| Symmetric 50% | Forward (True T) | 0.0000 ± 0.0000 | 2.25 ± 0.00 | 2.25 | 79.17 $\pm$ 0.72 |
| Symmetric 50% | Forward (Anchor T) | 0.5033 ± 0.0478 | 8.51 ± 3.00 | 8.42 | 78.33 $\pm$ 0.84 |
| Symmetric 50% | Forward (ConfLearning T) | 0.7297 ± 0.0143 | 14.23 ± 1.67 | 14.22 | 79.97 $\pm$ 0.10 |
| Symmetric 50% | Forward (Perturbed T) | 0.6667 ± 0.0000 | 4.50 ± 0.00 | 4.50 | 82.00 $\pm$ 0.76 |
| Asymmetric 40% | Forward (True T) | 0.0000 ± 0.0000 | 5.54 ± 0.00 | 5.00 | 87.80 $\pm$ 0.25 |
| Asymmetric 40% | Forward (Anchor T) | 0.2960 ± 0.0711 | 9.39 ± 4.62 | 8.51 | 83.63 $\pm$ 2.74 |
| Asymmetric 40% | Forward (ConfLearning T) | 0.8390 ± 0.0072 | 34.66 ± 3.47 | 33.13 | 82.58 $\pm$ 2.14 |
| Asymmetric 40% | Forward (Perturbed T) | 1.2845 ± 0.0000 | 10.13 ± 0.00 | 10.00 | 87.04 $\pm$ 0.15 |
