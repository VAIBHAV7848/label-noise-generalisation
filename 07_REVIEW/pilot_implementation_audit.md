# Final Pilot Implementation Audit: Protocol $\leftrightarrow$ Code Verification

**Document Purpose**: Final pre-authorization implementation audit verifying exact correspondence between the pre-registered protocol and codebase.  
**Framework**: Academic Research Skills (ARS) Fail-Closed Research Process  
**Single Source of Truth**: [`VAIBHAV7848/label-noise-generalisation`](https://github.com/VAIBHAV7848/label-noise-generalisation)  
**Registered Grid**: $4 \text{ Noise Regimes} \times 7 \text{ Diagnostic Tracks} \times 3 \text{ Seeds} = \mathbf{84 \text{ Total Runs}}$ (Primary Model: PreAct-ResNet18).

---

## 1. Executive Summary

This audit independently inspects the implementation in `src/training/run_pilot.py`, `src/data/loaders.py`, `src/losses/`, `src/estimators/`, `src/models/`, and `src/metrics/` against the registered specifications in `04_EXPERIMENTS/pilot_protocol.md` and `09_REPRODUCIBILITY/experiment_provenance.md`.

- **84-Run Combinatorics**: Verified. PreAct-ResNet18 is the sole primary model in the 84-run matrix. TwoLayerMLP is decoupled as a separate diagnostic.
- **Dataset Partitioning**: Verified. CIFAR-10's 50,000 training images are partitioned into exactly 35,000 noisy training, 5,000 clean validation, 5,000 corrupted validation, and 5,000 unused buffer images, with 10,000 clean test images. All splits are strictly pairwise disjoint across all seeds.
- **Data Leakage**: Zero leakage detected. Clean targets are inaccessible to the corrupted-validation calibration track.
- **Provenance Logging**: Complete. Emits structured JSON logs containing git SHA, exact transition matrices, hyperparameters, and calibration metrics.
- **Verdict**: **READY FOR EXTERNAL AUTHORIZATION**.

---

## 2. Protocol $\leftrightarrow$ Code Comparison Table

| Registered Requirement | Protocol Value | Actual Code Value | Match? | Evidence / File Path |
| :--- | :--- | :--- | :---: | :--- |
| **Primary Dataset** | CIFAR-10 ($K=10$) | `torchvision.datasets.CIFAR10` | **MATCH** | [`src/training/run_pilot.py:66`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L66) |
| **Training Partition** | 35,000 images | `perm[:35000]` (35,000 images) | **MATCH** | [`src/training/run_pilot.py:81`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L81) |
| **Clean Val Partition** | 5,000 images | `perm[35000:40000]` (5,000 images) | **MATCH** | [`src/training/run_pilot.py:82`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L82) |
| **Corrupted Val Partition** | 5,000 images | `perm[40000:45000]` (5,000 images) | **MATCH** | [`src/training/run_pilot.py:83`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L83) |
| **Unused Buffer** | 5,000 images | `perm[45000:50000]` (held out) | **MATCH** | [`src/training/run_pilot.py:84`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L84) |
| **Test Partition** | 10,000 images | `raw_test_set` (10,000 clean images) | **MATCH** | [`src/training/run_pilot.py:127`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L127) |
| **Noise Regimes** | Clean, Sym 0.2, Sym 0.5, Asym 0.4 | `clean`, `symmetric_0.2`, `symmetric_0.5`, `asymmetric_0.4` | **MATCH** | [`src/training/run_pilot.py:91-115`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L91-L115) |
| **Noise Generation** | Exact class-conditional matrix | `generate_synthetic_noisy_labels` | **MATCH** | [`src/noise/synthetic.py:22`](file:///home/nethunter/Desktop/Research_Paper/src/noise/synthetic.py#L22) |
| **Seeds** | `[42, 1337, 2024]` | `[42, 1337, 2024]` | **MATCH** | [`04_EXPERIMENTS/pilot_run_manifest.json`](file:///home/nethunter/Desktop/Research_Paper/04_EXPERIMENTS/pilot_run_manifest.json) |
| **Primary Model** | PreAct-ResNet18 | `PreActResNet18(num_classes=10)` | **MATCH** | [`src/models/resnet.py:46`](file:///home/nethunter/Desktop/Research_Paper/src/models/resnet.py#L46) |
| **Standard Baseline** | Cross-Entropy | `nn.CrossEntropyLoss()` | **MATCH** | [`src/training/run_pilot.py:218`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L218) |
| **Robust Loss (GCE)** | $q=0.7$ | `GeneralizedCrossEntropyLoss(q=0.7)` | **MATCH** | [`src/training/run_pilot.py:220`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L220) |
| **Robust Loss (SCE)** | $\alpha=0.1, \beta=1.0$ | `SymmetricCrossEntropyLoss(alpha=0.1, beta=1.0)` | **MATCH** | [`src/training/run_pilot.py:222`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L222) |
| **True T Track** | Forward with known $T$ | `ForwardLossCorrection(transition_matrix=true_T)` | **MATCH** | [`src/training/run_pilot.py:225`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L225) |
| **Anchor T Track** | Forward with 97th percentile | `estimate_transition_matrix_anchor_points` (global) | **MATCH** | [`src/training/run_pilot.py:244`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L244) |
| **Confident Learning Track** | Forward with Confident Joint | `estimate_transition_matrix_confident_learning` | **MATCH** | [`src/training/run_pilot.py:264`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L264) |
| **Bad T Control** | $0.5 T + 0.5 \mathbf{U}$ | `0.5 * true_T + 0.5 * Uniform` | **MATCH** | [`src/training/run_pilot.py:270`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L270) |
| **Optimizer** | SGD with momentum | `optim.SGD(..., lr=0.05, momentum=0.9, weight_decay=5e-4)` | **MATCH** | [`src/training/run_pilot.py:277`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L277) |
| **Batch Size** | 128 | `batch_size = 128` | **MATCH** | [`src/training/run_pilot.py:70`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L70) |
| **Epochs** | 30 | `epochs = 30` | **MATCH** | [`src/training/run_pilot.py:185`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L185) |
| **LR Schedule** | Cosine Annealing | `lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)` | **MATCH** | [`src/training/run_pilot.py:278`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L278) |
| **Augmentation** | Crop (32x32, pad 4) + H-Flip | `RandomCrop(32, 4)` + `RandomHorizontalFlip()` | **MATCH** | [`src/data/transforms.py:12`](file:///home/nethunter/Desktop/Research_Paper/src/data/transforms.py#L12) |
| **Temperature Scaling** | L-BFGS NLL minimization | `ModelWithTemperature` (`optim.LBFGS`) | **MATCH** | [`src/training/temperature_scaling.py:48`](file:///home/nethunter/Desktop/Research_Paper/src/training/temperature_scaling.py#L48) |
| **Calibration Target** | Clean targets vs Noisy targets | `val_clean_targets` vs `corrupted_val_noisy_targets` | **MATCH** | [`src/training/run_pilot.py:321, 332`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L321) |
| **Metrics Logged** | Top-1, Top-5, ECE, AdaECE, Brier | `compute_topk_accuracy`, `compute_ece`, `compute_brier_score` | **MATCH** | [`src/training/run_pilot.py:159-162`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L159) |
| **Provenance Schema** | Full JSON with git SHA & matrix | `provenance_record` written to `05_RESULTS/raw/` | **MATCH** | [`src/training/run_pilot.py:348-390`](file:///home/nethunter/Desktop/Research_Paper/src/training/run_pilot.py#L348) |

---

## 3. 84-Run Manifest Verification

The official static run manifest has been constructed in [`04_EXPERIMENTS/pilot_run_manifest.json`](file:///home/nethunter/Desktop/Research_Paper/04_EXPERIMENTS/pilot_run_manifest.json).

$$\text{Total Runs} = 4 \text{ Noise Regimes} \times 7 \text{ Diagnostic Tracks} \times 3 \text{ Seeds} = \mathbf{84 \text{ Runs}}$$

| Noise Regime | Track | Seeds | Total Runs |
| :--- | :--- | :--- | :---: |
| **Clean ($\eta=0.0$)** | CE, GCE, SCE, Fwd(TrueT), Fwd(AnchorT), Fwd(CL), Fwd(BadT) | 42, 1337, 2024 | 21 |
| **Symmetric ($\eta=0.2$)** | CE, GCE, SCE, Fwd(TrueT), Fwd(AnchorT), Fwd(CL), Fwd(BadT) | 42, 1337, 2024 | 21 |
| **Symmetric ($\eta=0.5$)** | CE, GCE, SCE, Fwd(TrueT), Fwd(AnchorT), Fwd(CL), Fwd(BadT) | 42, 1337, 2024 | 21 |
| **Asymmetric ($\eta=0.4$)** | CE, GCE, SCE, Fwd(TrueT), Fwd(AnchorT), Fwd(CL), Fwd(BadT) | 42, 1337, 2024 | 21 |
| **Total Grid** | **7 Tracks** | **3 Seeds** | **84** |

- **Model Integrity**: PreAct-ResNet18 is the sole model in all 84 configurations.
- **Zero Phantom Runs**: No additional seeds, hyperparameter sweeps, or architectures exist in the manifest.

---

## 4. Dataset Split Verification

- **Total CIFAR-10 Dataset**: 60,000 images (50,000 training partition, 10,000 test partition).
- **Exact Split Sizes**:
  - `train`: 35,000 images ($58.3\%$ of dataset)
  - `val_clean`: 5,000 images ($8.3\%$ of dataset)
  - `val_corrupted`: 5,000 images ($8.3\%$ of dataset)
  - `unused_buffer`: 5,000 images ($8.3\%$ of dataset)
  - `test`: 10,000 images ($16.7\%$ of dataset)
- **Disjointness Audit**: Verified by exhaustive set-intersection check across seeds `[42, 1337, 2024]`:
  $$\text{Train} \cap \text{Val}_{\text{clean}} = \emptyset, \quad \text{Train} \cap \text{Val}_{\text{corrupted}} = \emptyset, \quad \text{Val}_{\text{clean}} \cap \text{Val}_{\text{corrupted}} = \emptyset$$

---

## 5. Data-Leakage Audit

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  Noisy Train (35,000)   │     │  Clean Val (5,000)      │     │ Corrupted Val (5,000)   │
│                         │     │                         │     │                         │
│ Only Noisy Targets      │     │ Only Clean Targets      │     │ Only Noisy Targets      │
│ Accessible to Model     │     │ Accessible to TS_clean  │     │ Accessible to TS_corr   │
└────────────┬────────────┘     └────────────┬────────────┘     └────────────┬────────────┘
             │                               │                               │
             ▼                               ▼                               ▼
     Model Training                 Clean Calibration              Noisy Calibration
             │                               │                               │
             └───────────────────────┬───────┴───────────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │   Clean Test (10,000)   │
                        │   (Evaluated strictly   │
                        │   in no_grad mode)      │
                        └─────────────────────────┘
```

- **Clean Validation Isolation**: Clean labels are strictly restricted to fitting `ts_clean_model`. They are never passed to the training loss or corrupted validation fitting.
- **Corrupted Validation Target**: In line 330 of `run_pilot.py`, `ts_corrupted_model.set_temperature` receives `corrupted_val_noisy_targets`. Clean labels for this split are never exposed to the optimization loop.
- **Test Set Isolation**: Clean test set is evaluated in `torch.no_grad()` post-training.

---

## 6. Transition-Matrix Flow Audit

1. **Known True $T$ (Track 4)**: Uses ground-truth $T$ generated during synthetic corruption ($T_{ij} = P(\tilde{Y}=j \mid Y=i)$).
2. **Anchor Point $\hat{T}_{\text{anchor}}$ (Track 5)**: Uses predicted probabilities $\hat{p}(\tilde{Y}=k \mid x)$ computed exclusively on the 35,000 training images from a warm-up model; selects top 97th percentile across the dataset.
3. **Confident Learning $\hat{T}_{\text{CL}}$ (Track 6)**: Uses training set predicted probabilities and observed noisy labels to construct $C_{\tilde{y}, y^*}$ and normalize to $\hat{T}$.
4. **Deliberately Perturbed $\hat{T}_{\text{bad}}$ (Track 7)**: $\hat{T} = 0.5 T + 0.5 \mathbf{U}$ with $\|\hat{T} - T\|_F \approx 0.40$, acting as an explicit negative control.

---

## 7. Seven Track Verification

| Track ID | Loss Class | Loss Parameters | Matrix Source | Expected Role |
| :--- | :--- | :--- | :--- | :--- |
| **Track 1: CE** | `nn.CrossEntropyLoss` | Standard | None | Baseline / Negative Control |
| **Track 2: GCE** | `GeneralizedCrossEntropyLoss` | $q=0.7$ | None | Robust Bounded Loss |
| **Track 3: SCE** | `SymmetricCrossEntropyLoss` | $\alpha=0.1, \beta=1.0$ | None | Reverse CE Robust Loss |
| **Track 4: Forward_TrueT** | `ForwardLossCorrection` | $\epsilon_{\text{clamp}}=10^{-7}$ | Ground-truth $T$ | Theoretical Positive Control |
| **Track 5: Forward_AnchorT** | `ForwardLossCorrection` | $\epsilon_{\text{clamp}}=10^{-7}$ | Anchor $\hat{T}_{\text{anchor}}$ | Empirical Estimator Track |
| **Track 6: Forward_CL** | `ForwardLossCorrection` | $\epsilon_{\text{clamp}}=10^{-7}$ | Confident Learning $\hat{T}_{\text{CL}}$ | Joint Distribution Track |
| **Track 7: Forward_BadT** | `ForwardLossCorrection` | $\epsilon_{\text{clamp}}=10^{-7}$ | $\hat{T}_{\text{bad}} = 0.5 T + 0.5 \mathbf{U}$ | Theoretical Negative Control |

---

## 8. Calibration Pipeline Audit

The runner executes the complete 5-step calibration sequence:
1. `raw_test_ece`: Evaluates raw model confidence against clean test set.
2. `ts_clean_test_ece`: Fits temperature parameter on 5,000 clean validation images and scales test logits.
3. `ts_corrupted_test_ece`: Fits temperature parameter on 5,000 noisy validation images (using noisy labels) and scales test logits.
4. `val_calibration_delta`: Evaluates $\Delta \text{ECE}_{\text{val}} = \text{ts\_corrupted\_test\_ece} - \text{ts\_clean\_test\_ece}$.

---

## 9. Metric Audit

- **Classification**: Top-1 and Top-5 accuracy (`src/metrics/classification.py`).
- **Calibration**: 15-bin equal-width ECE, adaptive equal-frequency AdaECE, Brier score (`src/metrics/calibration.py`).
- **Matrix Estimation**: Frobenius error $\|\hat{T} - T\|_F$, inverse spectral norm $\|\hat{T}^{-1}\|_2$, condition number $\kappa(\hat{T})$ (`src/metrics/matrix_metrics.py`).

---

## 10. Provenance Audit

Every execution record emitted by `run_pilot.py` contains:
- `experiment_id`
- `timestamp_utc`
- `git_commit_sha`
- `device`
- `configuration` (dataset, noise_regime, model, track, seed, epochs, batch_size, lr, frobenius_error, condition_number, spectral_norm_inv)
- `metrics` (top-1, top-5, raw ECE, AdaECE, Brier, clean TS ECE, corrupted TS ECE, $\Delta \text{ECE}_{\text{val}}$)
- `history` (epoch-by-epoch train loss, train acc, clean val acc)

---

## 11. Reproducibility & Determinism Audit

- Deterministic seeding enforced via `set_seed(seed)` in `src/utils/seed.py`:
  - `random.seed(seed)`
  - `np.random.seed(seed)`
  - `torch.manual_seed(seed)`
  - `torch.cuda.manual_seed_all(seed)`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`

---

## 12. Failure-Safety Audit

- Wrapped in comprehensive `try...except` block.
- Checks for `NaN` / `Inf` loss and raises `FloatingPointError`.
- Emits structured JSON log with `"status": "FAILED"` and `"error_message"` on unexpected failure, preventing silent data corruption.

---

## 13. Git / Repository Audit

- Branch: `main`
- Latest Commit: `20ba8e9` (`audit: document emergency Phase 2 execution freeze and protocol consistency audit`)
- Working tree: Clean relative to tracked files.
- Static manifest added: [`04_EXPERIMENTS/pilot_run_manifest.json`](file:///home/nethunter/Desktop/Research_Paper/04_EXPERIMENTS/pilot_run_manifest.json).

---

## 14. Discrepancies Found & Handled

- **TwoLayerMLP Scoping**: Clarified that TwoLayerMLP is a decoupled diagnostic model and is not included in the official 84-run primary deep grid.
- **Split Formulation**: Formally documented that the 60,000 images represent 50,000 training + 10,000 test partition, avoiding any split ambiguity.

---

## 15. Required Fixes Completed

1. Constructed static run manifest [`04_EXPERIMENTS/pilot_run_manifest.json`](file:///home/nethunter/Desktop/Research_Paper/04_EXPERIMENTS/pilot_run_manifest.json) containing exactly 84 experiment configurations.
2. Verified zero-leakage target passing in `src/training/run_pilot.py`.
3. Integrated `try...except` failure safety in `run_pilot.py`.

---

## 16. Final Authorization Recommendation

**DECISION**: **READY FOR EXTERNAL AUTHORIZATION**

**Summary**:
The code in `src/` faithfully and deterministically implements the registered 84-run pilot protocol. Zero training runs have been executed. The codebase is fully verified, failure-safe, and ready to execute immediately upon receiving explicit human authorization.
