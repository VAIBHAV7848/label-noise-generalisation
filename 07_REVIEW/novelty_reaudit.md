# Adversarial Novelty & Related Literature Re-Audit

**Document Type**: Literature Landscape & Novelty Boundary Re-Audit  
**Framework**: Academic Research Skills (ARS) Literature Audit Standards  
**Single Source of Truth**: [`VAIBHAV7848/label-noise-generalisation`](https://github.com/VAIBHAV7848/label-noise-generalisation)

---

## 1. Literature Re-Audit & Taxonomy of Related Work

To prevent ungrounded claims of novelty (such as *"nobody has studied this"*), we comprehensively survey modern post-2020 and foundational literature addressing confidence calibration, loss correction, and label noise.

### 1.1 Detailed Related Work Matrix

| Paper & Venue | Core Contribution | Classification | Relationship to This Project |
| :--- | :--- | :---: | :--- |
| **Penso, Frenkel, & Goldberger** (*IEEE TMI 2024*) | "Confidence Calibration of a Medical Imaging Classification System That is Robust to Label Noise": uses $C_{\text{noisy}} = C_{\text{clean}} T$ to correct confidence calibration under noisy medical annotations. | **Substantial Partial Overlap / Domain Precedent** | *Precedent*: Established that transition matrix products apply to confusion matrices for calibration in medical imaging.<br>*Our Distinction*: We derive closed-form vector calibration identities (Theorems 4A & 4B), analyze non-medical multi-class vision benchmarks under controlled synthetic and human noise, and evaluate in-training robust losses (GCE, SCE) vs. post-hoc Temperature Scaling transfer degradation ($\Delta \text{ECE}_{\text{val}}$). |
| **Wu, Shi, Dong, & Zheng** (*Science China Information Sciences 2026*) | "TransTS: an adaptive post-hoc method for probability calibration under label noise": adaptively scales logits based on estimated noise levels. | **Substantial Partial Overlap / Direct Adjacency** | *Precedent*: Demonstrates that standard Temperature Scaling degrades under label noise and proposes adaptive scaling.<br>*Our Distinction*: We focus on the exact transfer penalty $\Delta \text{ECE}_{\text{val}} = \text{ECE}(\text{TS}_{\text{corrupted}}) - \text{ECE}(\text{TS}_{\text{clean}})$ under imperfect matrix estimation error $\|\hat{T}-T\|_F$, providing formal excess risk bounds (Proposition 2) and out-of-fold Confident Learning integration. |
| **Bai et al.** (*NeurIPS 2021*) | "Understanding and improving early stopping for learning with noisy labels": analyzes validation loss dynamics when validation sets contain noisy labels. | **Methodological Adjacency** | *Precedent*: Explores how noisy validation sets mislead early stopping.<br>*Our Distinction*: We focus on post-hoc calibration distortion and excess risk rather than early stopping heuristics. |
| **Wang et al.** (*ICML 2019*) | "Symmetric Cross Entropy for Robust Learning with Noisy Labels": combines Reverse Cross-Entropy with CE for noise robustness. | **Methodological Baseline** | *Precedent*: Establishes SCE loss.<br>*Our Project*: Includes SCE ($\alpha=0.1, \beta=1.0$) as an in-training robust loss track to benchmark its calibration profiles against loss-correction methods. |
| **Zhang & Sabuncu** (*NeurIPS 2018*) | "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels": $L_q$ bounded surrogate loss ($q=0.7$). | **Methodological Baseline** | *Precedent*: Establishes GCE loss.<br>*Our Project*: Includes GCE as an in-training robust loss track. |
| **Patrini et al.** (*CVPR 2017*) | "Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach": Forward and Backward loss correction via anchor points. | **Foundational Precedent** | *Precedent*: Establishes Forward/Backward correction.<br>*Our Project*: Extends with non-asymptotic operator-norm excess risk bounds under finite-sample matrix perturbation (Proposition 2) and out-of-fold joint estimation. |
| **Northcutt et al.** (*JAIR 2021*) | "Confident Learning: Estimating Uncertainty in Dataset Labels": joint distribution estimation with out-of-sample predicted probabilities. | **Foundational Estimator** | *Precedent*: Establishes Confident Learning.<br>*Our Project*: Integrates true 3-fold OOF cross-validation into the Forward correction pipeline. |
| **Natarajan et al.** (*NeurIPS 2013*) | "Learning with Noisy Labels": unbiased surrogate risk estimators for binary classification. | **Foundational Theory** | *Precedent*: Establishes unbiased risk minimization.<br>*Our Project*: Evaluates multi-class non-asymptotic bounds with operator norm conditioning. |

---

## 2. Refined & Defensible Research Gap Statement

We explicitly reject hyperbolic claims of absolute novelty. The research gap is rigorously framed as follows:

> **Defensible Research Gap:**  
> *"While recent studies have demonstrated that label noise degrades probability calibration (Guo et al., 2017; Wang et al., 2021) and proposed specialized post-hoc corrections (Penso et al., IEEE TMI 2024; Wu et al., SCIS 2026), existing literature lacks a unified theoretical and empirical treatment connecting finite-sample transition matrix perturbation bounds ($\|\hat{T}-T\|_F$) to excess risk guarantees and post-hoc calibration transfer degradation ($\Delta \text{ECE}_{\text{val}}$) across in-training robust losses (GCE, SCE) and loss-corrected deep architectures under controlled out-of-fold estimation."*
