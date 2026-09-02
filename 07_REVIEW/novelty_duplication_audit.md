# Novelty Duplication & Prior Art Search (2020–2026)

This document provides a targeted search for overlapping literature across key keyword intersections to ensure our work does not duplicate recent studies.

---

## 1. Topic: Label Noise + Model Calibration
- **Keywords**: `"label noise" AND "calibration"`, `"noisy labels" AND "ECE"`, `"confidence calibration" AND "loss correction"`
- **Discovered Papers**:
  - *Wang et al. (NeurIPS 2021)*: "On the Calibration of Noisy-Label Learning" — demonstrated that deep networks trained with noisy labels suffer from severe overconfidence on corrupted instances and that loss smoothing helps.
  - *Bai et al. (ICLR 2021)*: "How Does Mixup Help With Robustness and Calibration under Label Noise?" — analyzed Mixup's dual role in accuracy and ECE.
  - *Lukasik et al. (ICML 2020)*: "Does label smoothing mitigate label noise?" — showed that label smoothing acts as a noise regularizer and reduces calibration error.
- **Impact on Our Project**:
  - We **CANNOT** claim to be the first to evaluate ECE under label noise.
  - Our unique focus must be: *How does post-hoc calibration (Temperature Scaling) perform when tuned on a corrupted validation split vs clean validation split across loss correction and robust loss functions?*

---

## 2. Topic: Transition Matrix Estimation Error & Generalisation Bounds
- **Keywords**: `"transition matrix estimation error" AND "excess risk"`, `"perturbed noise matrix" AND "generalization bound"`
- **Discovered Papers**:
  - *Natarajan et al. (NeurIPS 2013)*: Asymptotic excess risk under binary noise estimation.
  - *Xia et al. (NeurIPS 2019)*: Estimation error bounds for Dual-T estimator.
  - *Cheng et al. (NeurIPS 2020)*: "Learning with Bounded Instance- and Label-Dependent Label Noise" — bounds under instance noise.
  - *Zhu et al. (TMLR 2023)*: Sample complexity under noise rate bounds.
- **Impact on Our Project**:
  - Our Proposition 2 (explicit linear excess risk bound in terms of $\sqrt{K} M \|T^{-1}\|_2 \|\hat{T} - T\|_F$) provides a clean, unified operator-norm bound connecting finite-sample estimation error directly to Rademacher generalization.

---

## 3. Topic: Synthetic vs. Real Human Noise (CIFAR-10N)
- **Keywords**: `"CIFAR-10N" AND "benchmark"`, `"human label noise" AND "loss correction"`
- **Discovered Papers**:
  - *Wei et al. (ICLR 2022)*: "Learning with Noisy Labels Revisited: A Study on ImageNet and CIFAR-10N" — established that class-conditional loss correction fails on CIFAR-10N because human noise is instance-dependent.
  - *Song et al. (IEEE TPAMI 2022)*: Re-affirmed instance-dependent failure modes.
- **Impact on Our Project**:
  - CIFAR-10N is a standard benchmark, not a novel gap of ours. We use it strictly to benchmark the transfer gap under controlled conditions.
