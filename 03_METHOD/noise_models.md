# Noise Models & Generation Protocols

This document defines the mathematical formulation and generation procedures for all synthetic and real-world label noise regimes used in this research.

---

## 1. Symmetric (Uniform) Label Noise

Under symmetric noise with overall corruption rate $\eta \in [0, 1)$, a sample's true label $Y \in \{1, \dots, K\}$ is flipped uniformly at random to any of the remaining $K-1$ classes with probability $\frac{\eta}{K-1}$.

### Formal Transition Matrix:
$$T_{\text{sym}}(\eta) = (1 - \eta) I_K + \frac{\eta}{K-1} (\mathbf{1}_K \mathbf{1}_K^\top - I_K)$$

$$\forall i, j \in \{1, \dots, K\}: \quad T_{ij} = \begin{cases} 1 - \eta & \text{if } i = j \\ \frac{\eta}{K-1} & \text{if } i \ne j \end{cases}$$

### Investigated Noise Rates:
$$\eta \in \{0.0, 0.2, 0.4, 0.6, 0.8\}$$

---

## 2. Asymmetric (Class-Conditional / Pair-Flip) Label Noise

Under asymmetric noise, label flips mimic realistic semantic confusions between visually or conceptually similar classes.

### CIFAR-10 Asymmetric Noise Mapping:
Following Patrini et al. (2017) and Zhang & Sabuncu (2018):
- $\text{TRUCK} \rightarrow \text{AUTOMOBILE}$ with probability $\eta$
- $\text{AUTOMOBILE} \rightarrow \text{TRUCK}$ with probability $\eta$
- $\text{BIRD} \rightarrow \text{AIRPLANE}$ with probability $\eta$
- $\text{DEER} \rightarrow \text{HORSE}$ with probability $\eta$
- $\text{CAT} \leftrightarrow \text{DOG}$ with probability $\eta$

### Formal Transition Matrix ($K=10$):
For classes with directed flips:
$$T_{ii} = 1 - \eta, \quad T_{i, \text{target}(i)} = \eta, \quad T_{ij} = 0 \quad (\forall j \notin \{i, \text{target}(i)\})$$

### Investigated Noise Rates:
$$\eta \in \{0.1, 0.2, 0.3, 0.4\}$$ *(Note: $\eta \ge 0.5$ in pair-flip makes classes indistinguishable without anchor points).*

---

## 3. Real-World Human Annotator Noise (CIFAR-10N & Animal-10N)

Rather than simulating synthetic transition matrices, real-world noise captures true human cognitive ambiguity, low-resolution artifacts, and label disagreement.

### CIFAR-10N Noise Regimes (Wei et al., ICLR 2022):
1. **Clean**: Original verified test/train ground truth labels.
2. **Aggregate Label**: Aggregated majority vote across 3 Amazon Mechanical Turk annotators ($\approx 9.03\%$ noise rate).
3. **Random-1 / Random-2 / Random-3**: Single independent human annotator ratings ($\approx 17.2\% - 18.1\%$ noise rate).
4. **Worst Label**: The label assigned by the most error-prone human annotator ($\approx 40.21\%$ noise rate).

### Animal-10N Noise (Song et al., ICML 2019):
- Real web-crawled dataset of 5 pairs of confusing animal classes with naturally occurring estimated human noise rate $\approx 8.4\%$.
