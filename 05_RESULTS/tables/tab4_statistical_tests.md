# Table 4: Statistical Significance of Performance Deltas vs CE Baseline

| Noise Regime | Track vs CE | Mean Delta Acc (%) | t-statistic | Exact p-value | Holm-Bonferroni p | Cohen's d | 95% Confidence Interval |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean | Generalized CE (GCE) | -1.80% | -22.44 | 1.9802e-03 | 4.4226e-02 | -12.96 | [-2.14, -1.45] |
| Clean | Symmetric CE (SCE) | -10.80% | -14.31 | 4.8490e-03 | 8.8485e-02 | -8.26 | [-14.05, -7.55] |
| Clean | Forward (True T) | -0.18% | -3.93 | 5.9107e-02 | 2.9554e-01 | -2.27 | [-0.38, +0.02] |
| Clean | Forward (Anchor T) | -0.20% | -5.01 | 3.7548e-02 | 2.5993e-01 | -2.89 | [-0.38, -0.03] |
| Clean | Forward (ConfLearning T) | -0.63% | -3.78 | 6.3414e-02 | 2.9554e-01 | -2.18 | [-1.35, +0.09] |
| Clean | Forward (Perturbed T) | -0.91% | -3.77 | 6.3678e-02 | 2.9554e-01 | -2.18 | [-1.95, +0.13] |
| Symmetric 20% | Generalized CE (GCE) | +3.15% | 11.81 | 7.0894e-03 | 1.0906e-01 | 6.82 | [+2.00, +4.30] |
| Symmetric 20% | Symmetric CE (SCE) | -6.84% | -9.67 | 1.0530e-02 | 1.2635e-01 | -5.58 | [-9.89, -3.80] |
| Symmetric 20% | Forward (True T) | +2.35% | 22.60 | 1.9524e-03 | 4.4226e-02 | 13.05 | [+1.91, +2.80] |
| Symmetric 20% | Forward (Anchor T) | +1.59% | 8.62 | 1.3184e-02 | 1.4031e-01 | 4.98 | [+0.80, +2.39] |
| Symmetric 20% | Forward (ConfLearning T) | +2.89% | 10.83 | 8.4228e-03 | 1.0950e-01 | 6.25 | [+1.74, +4.04] |
| Symmetric 20% | Forward (Perturbed T) | +3.62% | 14.60 | 4.6571e-03 | 8.8485e-02 | 8.43 | [+2.55, +4.68] |
| Symmetric 50% | Generalized CE (GCE) | +7.95% | 11.39 | 7.6136e-03 | 1.0906e-01 | 6.58 | [+4.95, +10.95] |
| Symmetric 50% | Symmetric CE (SCE) | -23.42% | -17.69 | 3.1796e-03 | 6.3593e-02 | -10.21 | [-29.12, -17.73] |
| Symmetric 50% | Forward (True T) | +4.72% | 27.89 | 1.2828e-03 | 3.0787e-02 | 16.10 | [+3.99, +5.45] |
| Symmetric 50% | Forward (Anchor T) | +3.88% | 5.04 | 3.7133e-02 | 2.5993e-01 | 2.91 | [+0.57, +7.18] |
| Symmetric 50% | Forward (ConfLearning T) | +5.52% | 8.77 | 1.2755e-02 | 1.4031e-01 | 5.06 | [+2.81, +8.22] |
| Symmetric 50% | Forward (Perturbed T) | +7.55% | 14.18 | 4.9384e-03 | 8.8485e-02 | 8.19 | [+5.26, +9.84] |
| Asymmetric 40% | Generalized CE (GCE) | -2.88% | -8.67 | 1.3053e-02 | 1.4031e-01 | -5.00 | [-4.31, -1.45] |
| Asymmetric 40% | Symmetric CE (SCE) | -18.47% | -8.77 | 1.2760e-02 | 1.4031e-01 | -5.06 | [-27.54, -9.41] |
| Asymmetric 40% | Forward (True T) | +6.19% | 22.77 | 1.9229e-03 | 4.4226e-02 | 13.15 | [+5.02, +7.36] |
| Asymmetric 40% | Forward (Anchor T) | +2.01% | 1.35 | 3.0868e-01 | 6.1735e-01 | 0.78 | [-4.38, +8.40] |
| Asymmetric 40% | Forward (ConfLearning T) | +0.97% | 0.99 | 4.2625e-01 | 6.1735e-01 | 0.57 | [-3.23, +5.16] |
| Asymmetric 40% | Forward (Perturbed T) | +5.42% | 12.05 | 6.8162e-03 | 1.0906e-01 | 6.96 | [+3.48, +7.36] |
