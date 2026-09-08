# REQUIRED REVISION PLAN

## GROUP 1: MUST FIX BEFORE SUBMISSION (Blocking Issues)

| Exact File | Section | Problem | Why Reviewer Will Care | Recommended Fix | New Experiment Required? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `main.tex` | Sec 5.1 (Exp Setup) | $N=3$ Seeds | 3 seeds is statistically meaningless for DL generalization claims. | Increase to $N=5$ or $N=10$ seeds. | **YES** |
| `main.tex` | Sec 6.1 (Accuracy) | True $T$ Oracle Paradox | GCE beats the mathematically unbiased oracle (True $T$). Undermines theory. | Add a paragraph explaining *why* GCE wins (e.g., optimization landscape of inverted matrices vs gradient bounding). | No |
| `main.tex` | Sec 4.2 (Prop 2B) | Sample Splitting Disconnect | Prop 2B assumes independent $\hat{T}$. Code uses same-sample estimation. | Add a formal remark clarifying that Prop 2B is a theoretical upper bound that assumes splitting, and acknowledge the gap. | No |
| `main.tex` | Sec 5.2 / Sec 8 | Missing Real-World Noise | CIFAR-10 synthetic noise is not enough for modern papers. | Execute Phase 2 (CIFAR-10N benchmark) and include in the main text. | **YES** |
| `main.tex` | Abstract / Intro | Overclaiming Novelty | Abstract implies derivation of convex symmetry barrier (prior art). | Rewrite abstract to say "Building on the fundamental noise tolerance barrier..." | No |

---

## GROUP 2: SHOULD FIX (Major Improvements)

| Exact File | Section | Problem | Why Reviewer Will Care | Recommended Fix | New Experiment Required? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `main.tex` | Title | Scope Overreach | "Generalisation" implies a broad study across domains/architectures. | Change to: *Empirical Risk Bounds and Calibration Distortion...* | No |
| `main.tex` | Sec 7.2 | Strawman TS | Tuning standard TS on noisy labels is trivially bad. | Add a robust TS baseline (e.g., tune TS with GCE loss on noisy val). | **YES** |
| `main.tex` | Sec 2 (Related Work) | Dated Baselines | Literature stops around 2021. | Add a subsection discussing modern contrastive/sample-selection methods (2022-2024). | No |
| `main.tex` | Sec 4.5 (Theorem 4B) | ECE Bound Scope | Exact ECE equality is only proven for symmetric noise. | Clarify text to ensure readers know asymmetric noise lacks the exact lower bound. | No |

---

## GROUP 3: OPTIONAL POLISH

| Exact File | Section | Problem | Why Reviewer Will Care | Recommended Fix | New Experiment Required? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `main.tex` | Sec 8 (Discussion) | Circular Advice | "Diagnose noise symmetry" requires knowing the matrix you are trying to avoid estimating. | Suggest practical heuristics for diagnosing asymmetry (e.g., confusion matrix of early stopped model). | No |
| `main.tex` | Figures | ECE Binning | Raw ECE is bin-sensitive. | Ensure AdaECE is reported prominently alongside raw ECE to prove robustness to binning artifacts. | No |
| `main.tex` | Sec 6.2 | Capacity Scaling | H2 claims capacity affects memorization, but only tests one architecture. | Run a small grid of ResNet-34 and Logistic Regression to prove the scaling law. | **YES** (Optional) |
