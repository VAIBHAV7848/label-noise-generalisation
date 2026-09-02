# Methodological Design Decisions

## 1. Modular Separation of Estimation vs. Loss Correction
- **Decision**: Separate the transition matrix estimation step $\hat{T}$ from the downstream training objective into independent, modular components.
- **Rationale**: Combining estimation and training in an end-to-end black box makes it impossible to isolate whether classification failure stems from poor matrix estimation or loss function deficiency. Decoupling allows exact sensitivity benchmarking under varying $\|\hat{T} - T\|_F$.

## 2. Standardized Optimizer & Hyperparameter Budgets
- **Decision**: Fix all architectural hyperparameters (learning rate schedule, weight decay, batch size, momentum) across baselines within each model family, adhering strictly to established benchmark standards (e.g. PreAct-ResNet18 trained with SGD, initial lr=0.1, cosine annealing, weight decay $5 \times 10^{-4}$ for 120 epochs).
- **Rationale**: Prevents unfair baseline comparisons where one method appears superior merely due to hyperparameter tuning disparities.

## 3. Post-Hoc Noise-Aware Calibration Protocol
- **Decision**: Evaluate Temperature Scaling (Guo et al., 2017) using both a clean validation split and a noisy validation split to evaluate calibration recoverability in scenarios where no clean verification data is accessible.
- **Rationale**: In real-world noisy label problems, practitioners rarely possess a clean validation set. Testing calibration repair on corrupted validation data addresses a key practical bottleneck.
