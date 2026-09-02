# Hypothesis $\leftrightarrow$ Pilot Alignment & Traceability Matrix

**Document Type**: Pre-Pilot Scientific Scope & Hypothesis Mapping  
**Framework**: Academic Research Skills (ARS) Hypothesis Traceability  
**Single Source of Truth**: [`VAIBHAV7848/label-noise-generalisation`](https://github.com/VAIBHAV7848/label-noise-generalisation)

---

## 1. Traceability Matrix: Hypotheses vs. Pilot Capabilities

The 84-run pilot grid ($4 \text{ Noise Regimes} \times 7 \text{ Diagnostic Tracks} \times 3 \text{ Seeds}$ on PreAct-ResNet18) is an empirical diagnostic validation tool. It does **not** test all hypotheses equally.

| Hypothesis | Tested by 84-Run Pilot? | What the Pilot Actually Establishes | What is Deferred to Phase 3 (300-Run MDES)? |
| :--- | :---: | :--- | :--- |
| **Hypothesis 1 (H1: Noise Asymmetry & Decision Boundary Displacement)** | **Partially (Diagnostic)** | Evaluates whether asymmetric pair-flip noise ($\eta=0.4, \|T-T^\top\|_F > 0$) induces greater generalisation loss on uncorrected Cross-Entropy than symmetric noise ($\eta=0.5, \|T-T^\top\|_F = 0$) under balanced priors on PreAct-ResNet18. | Explicit hyperplane angle measurements $\theta(\mathbf{w}^*_{\text{clean}}, \mathbf{w}^*_{\text{noisy}})$, margin distribution shifts $\gamma$, and linear model capacity sweeps. |
| **Hypothesis 2 (H2: Parametric Generalisation Window & Capacity)** | **Partially (Tracking)** | Records epoch-by-epoch training loss, training accuracy, and clean validation accuracy across 30 epochs on PreAct-ResNet18 to observe the onset of memorization $\tau_{\text{memorize}}$. | Sweeping parameter count $p \in [10^3, 10^7]$ across 10 diverse architectures (MLP, ConvNet, ResNets) to verify inverse window scaling $\Delta \tau \propto N/p$. |
| **Hypothesis 3 (H3: Miscalibration Profiles & Corrupted Val Recovery)** | **Fully (Direct Test)** | **Primary Pilot Focus**: Directly compares 15-bin ECE, AdaECE, and Brier score across uncorrected CE, bounded GCE ($q=0.7$), symmetric SCE ($\alpha=0.1, \beta=1.0$), and Forward correction, while quantifying validation calibration degradation $\Delta \text{ECE}_{\text{val}} = \text{ECE}(\text{TS}_{\text{corrupted}}) - \text{ECE}(\text{TS}_{\text{clean}})$. | Multi-architecture statistical power evaluation (5 seeds per condition) and reliability diagram clustering. |
| **Hypothesis 4 (H4: Controlled Human Noise Gap on CIFAR-10N)** | **NO (0%)** | **None**: The pilot is strictly restricted to CIFAR-10 synthetic noise regimes to establish baseline experimental control. | Controlled head-to-head comparison on CIFAR-10N (Clean, Aggregate, Worst) comparing Transition Matrix Forward Correction + MixUp vs. Confident Learning + MixUp. |

---

## 2. Methodological Scope Summary

1. **Pilot Scope**: Diagnostic verification of end-to-end pipeline mechanics, validation calibration transfer degradation ($\Delta \text{ECE}_{\text{val}}$), loss correction numerical behavior under True $T$ vs. Estimated $\hat{T}$ vs. Bad $\hat{T}_{\text{bad}}$, and baseline empirical effect sizes.
2. **Phase 3 MDES Scope**: Full multi-architecture capacity scaling ($p/N$), human noise CIFAR-10N transfer evaluation, and inferential statistical testing (Wilcoxon signed-rank tests with Holm-Bonferroni correction across 5 seeds).
