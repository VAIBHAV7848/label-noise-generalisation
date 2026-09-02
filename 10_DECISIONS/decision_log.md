# Comprehensive Research Decision Log

Every major research decision, its motivation, evidence basis, rejected alternatives, and impact are formally documented here.

---

### Record DEC-001: Selection of Primary Research Question (Composite RQ 4)
- **Date**: 2026-09-02
- **Decision**: Adopt Candidate RQ 4 as the primary project research question: *"Under what exact theoretical and empirical conditions do loss-correction methods, robust loss functions, and sample selection algorithms maintain both generalisation accuracy and probability calibration when subjected to symmetric, class-conditional, and realistic instance-dependent label noise across varying model capacities?"*
- **Reason**: Balances mathematical depth (excess risk under matrix perturbation $\hat{T}$) with critical empirical gaps (calibration ECE and synthetic-to-human noise generalisation).
- **Evidence**: Literature matrix demonstrates that pure symmetric noise sensitivity is saturated, while calibration and human noise robustness remain major open concerns.
- **Alternatives Considered**: Candidate RQ 1 (Naive sensitivity on MNIST/CIFAR-10) and Candidate RQ 2 (Pure loss robustness trade-offs).
- **Why Rejected**: RQ 1 is fully solved and leads to immediate desk-rejection; RQ 2 lacks calibration and realistic noise scope.

---

### Record DEC-002: Inclusion of Deep Residual Networks alongside Classical Models
- **Date**: 2026-09-02
- **Decision**: Incorporate PreAct-ResNet18 and ResNet-50 alongside Logistic Regression, Decision Tree, and 2-layer MLP.
- **Reason**: Investigating capacity-dependent memorisation requires bridging underparameterized convex models ($p \ll N$) with modern overparameterized deep models ($p \gg N$).
- **Evidence**: Zhang et al. (2017) and Arpit et al. (2017) proved that overparameterization fundamentally alters the optimization trajectory under noise.
- **Alternatives Considered**: Restricting scope strictly to Logistic Regression, Decision Tree, and 2-layer MLP.
- **Why Rejected**: Restricting to shallow models makes results outdated and irreproducible for modern deep learning.

---

### Record DEC-003: Integration of Real Crowdsourced Human Noise (CIFAR-10N)
- **Date**: 2026-09-02
- **Decision**: Include CIFAR-10N as a mandatory primary benchmark.
- **Reason**: Synthetic class-conditional noise assumes conditional independence $P(\tilde{Y} \mid X, Y) = P(\tilde{Y} \mid Y)$, which fails in real human annotation settings.
- **Evidence**: Wei et al. (ICLR 2022) demonstrated that algorithms achieving SOTA on synthetic noise degrade substantially on human noise.
- **Alternatives Considered**: Evaluating exclusively synthetic uniform and pair-flip noise.
- **Why Rejected**: Saturated, artificial, and susceptible to heavy reviewer criticism.

---

### Record DEC-004: Standardizing 5-Seed Multi-Run Statistical Protocol
- **Date**: 2026-09-02
- **Decision**: Mandate 5 fixed random seeds (`[42, 1337, 2024, 7, 999]`) and report mean $\pm$ standard deviation with two-tailed paired t-tests ($p < 0.05$).
- **Reason**: Ensures statistical reproducibility and eliminates lucky-seed anomalies.
- **Alternatives Considered**: 3 seeds or single-seed runs.
- **Why Rejected**: 3 seeds lack sufficient statistical power for Wilcoxon rank tests; single runs violate empirical integrity.
