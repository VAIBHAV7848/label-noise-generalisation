# Transition Matrix Estimation Methodology & Out-Of-Fold (OOF) Protocols

**Scope**: Transition matrix estimators implemented in `src/estimators/`.  
**Framework**: Academic Research Skills (ARS) Protocol Specification  
**Single Source of Truth**: [`VAIBHAV7848/label-noise-generalisation`](https://github.com/VAIBHAV7848/label-noise-generalisation)

---

## 1. Literature Foundations & Requirements

### 1.1 Anchor Point Estimator (Patrini et al., CVPR 2017)
- **Theoretical Basis**: Under the anchor point condition, for every clean class $i$, there exists an instance $x \in \mathcal{X}$ such that $P(Y=i \mid x) = 1$. Under class-conditional noise:
  $$P(\tilde{Y}=j \mid x) = \sum_{k=1}^K T_{kj} P(Y=k \mid x) = T_{ij}$$
- **Algorithm**:
  1. Train a base model on the 35,000 noisy training instances $\tilde{S}$ for 5 epochs.
  2. Compute predicted probabilities $\hat{p}(\tilde{y} \mid x)$ across candidate instances.
  3. Select the top 97th percentile candidate instances maximizing $\hat{p}(\tilde{y}=i \mid x)$ globally across the dataset (`global_search=True`).
  4. Estimate row $i$ as $\hat{T}_{i, :} = \frac{1}{|\mathcal{A}_i|} \sum_{x \in \mathcal{A}_i} \hat{p}(\tilde{y} \mid x)$.

### 1.2 Confident Learning Estimator (Northcutt et al., JAIR 2021)
- **Theoretical Basis**: Confident Learning characterizes label noise by directly estimating the unnormalized joint distribution matrix $\mathbf{C}_{\tilde{y}, y^*}$.
- **Strict Literature Requirement**: To prevent deep models from memorizing corrupted training instances, predicted probabilities **must be out-of-fold (OOF)** computed via cross-validation.
- **Algorithm**:
  1. Partition the 35,000 noisy training instances into $K_{\text{cv}} = 3$ deterministic disjoint folds.
  2. For fold $k \in \{0, 1, 2\}$:
     - Train a model on folds $\{0, 1, 2\} \setminus \{k\}$ for 5 warm-up epochs.
     - Predict posterior probabilities on holdout fold $k$.
  3. Form full OOF probability matrix $\hat{\mathbf{P}}_{\text{oof}} \in [0, 1]^{35000 \times 10}$.
  4. Compute per-class self-confidence thresholds:
     $$t_j = \frac{1}{|X_{\tilde{y}=j}|} \sum_{x \in X_{\tilde{y}=j}} \hat{\mathbf{P}}_{\text{oof}}[x, j]$$
  5. Form counting matrix $\mathbf{C}_{\tilde{y}=j, y^*=k}$ by assigning each instance to the class maximizing margin $\hat{p}_k - t_k$ among candidates exceeding threshold.
  6. Derive $\hat{Q}_{\tilde{y}, y^*} = \mathbf{C} / \sum_{j, k} \mathbf{C}_{j, k}$ and obtain row-stochastic $\hat{T} = (\hat{Q} / \hat{p}(y^*))^\top$.

---

## 2. Information Constraints & Zero-Leakage Guarantees

1. **Only Noisy Training Labels Permitted**: All warm-up models and CV folds have access strictly to $\tilde{y}_i$.
2. **Clean Val & Corrupted Val Isolated**: Zero validation instances are present in training or estimation.
3. **Clean Test Isolated**: Evaluated strictly in `torch.no_grad()` mode post-training.
