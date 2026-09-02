# Information-Access & Data-Leakage Audit Matrix

**Scope**: Data access permissions and label visibility across all 7 diagnostic tracks in Phase 2 Pilot.  
**Framework**: Academic Research Skills (ARS) Fail-Closed Research Process  
**Single Source of Truth**: [`VAIBHAV7848/label-noise-generalisation`](https://github.com/VAIBHAV7848/label-noise-generalisation)

---

## 1. Information-Access Matrix Across Diagnostic Tracks

| Diagnostic Track | Clean Train Labels? | Noisy Train Labels? | Clean Val Labels? | Corrupted Val Labels? | Test Labels? | Probability Source | OOF Required? |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Track 1: CE Baseline** | $\times$ | $\checkmark$ | $\times$ (eval only) | $\times$ | $\times$ | Model output | $\times$ |
| **Track 2: GCE ($q=0.7$)** | $\times$ | $\checkmark$ | $\times$ (eval only) | $\times$ | $\times$ | Model output | $\times$ |
| **Track 3: SCE ($\alpha=0.1, \beta=1.0$)** | $\times$ | $\checkmark$ | $\times$ (eval only) | $\times$ | $\times$ | Model output | $\times$ |
| **Track 4: Forward (True $T$)** | $\times$ | $\checkmark$ | $\times$ (eval only) | $\times$ | $\times$ | Model output + Known $T$ | $\times$ |
| **Track 5: Forward (Anchor $\hat{T}$)** | $\times$ | $\checkmark$ | $\times$ (eval only) | $\times$ | $\times$ | 5-epoch base warm-up model on $\tilde{S}$ | $\times$ (Base candidates) |
| **Track 6: Forward (Confident Learning $\hat{T}$)** | $\times$ | $\checkmark$ | $\times$ (eval only) | $\times$ | $\times$ | 3-Fold CV warm-up models (5 ep each) | $\checkmark$ (**3-Fold CV OOF**) |
| **Track 7: Forward (Bad $\hat{T}_{\text{bad}}$)** | $\times$ | $\checkmark$ | $\times$ (eval only) | $\times$ | $\times$ | Deliberately perturbed $0.5 T + 0.5 \mathbf{U}$ | $\times$ |

---

## 2. Partition-by-Partition Isolation Guarantees

```
┌───────────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐
│ Noisy Train (35,000)      │     │ Clean Val (5,000)         │     │ Corrupted Val (5,000)     │
│                           │     │                           │     │                           │
│ Accessible to:            │     │ Accessible to:            │     │ Accessible to:            │
│ • Model Trainer (noisy)   │     │ • Clean TS optimizer only │     │ • Corrupted TS optimizer  │
│ • Estimator warm-up (noisy│     │   (L-BFGS NLL with clean  │     │   (L-BFGS NLL with noisy  │
│ • OOF CV folds (noisy)    │     │    targets)               │     │    targets)               │
│                           │     │                           │     │                           │
│ CLEAN LABELS: FORBIDDEN   │     │ CORRUPTED LABELS: N/A     │     │ CLEAN LABELS: FORBIDDEN   │
└─────────────┬─────────────┘     └─────────────┬─────────────┘     └─────────────┬─────────────┘
              │                                 │                                 │
              └─────────────────────────┬───────┴─────────────────────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Clean Test (10,000)       │
                          │                           │
                          │ Accessible strictly       │
                          │ in torch.no_grad()        │
                          │ post-training evaluation. │
                          └───────────────────────────┘
```

1. **Model Training Isolation**: Model parameters are optimized using `noisy_targets` from the 35,000 noisy training split. Clean training labels are never supplied to `optimizer.step()`.
2. **Estimator Isolation**:
   - `estimate_transition_matrix_anchor_points` receives `train_noisy_labels` and posteriors from a 5-epoch base model trained on $\tilde{S}$.
   - `estimate_transition_matrix_confident_learning` receives `train_noisy_labels` and out-of-fold cross-validated probabilities from 3 disjoint fold models.
3. **Calibration Transfer Isolation**:
   - `ts_clean_model`: Tuned on `val_clean_eval["logits"]` against `val_clean_targets`.
   - `ts_corrupted_model`: Tuned on `val_corrupted_eval["logits"]` against `corrupted_val_noisy_targets`. Clean targets for this partition are inaccessible.
4. **Test Set Isolation**: Clean test set is evaluated in `torch.no_grad()` mode only after all model training and temperature optimization are complete.
