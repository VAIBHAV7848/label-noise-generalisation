# Proof: Unbiasedness of Backward Loss Correction & Forward Risk Relation

This document contains the step-by-step mathematical proof of Proposition 1 and formal analysis of Forward Loss Correction.

---

## 1. Theorem Statement (Backward Correction Unbiasedness)

Let $\mathcal{X}$ be the feature domain and $\mathcal{Y} = \{1, \dots, K\}$ be the label domain. Let $(X, Y) \sim \mathcal{D}$ denote clean random variables and $(X, \tilde{Y}) \sim \tilde{\mathcal{D}}$ denote corrupted random variables governed by transition matrix $T \in [0, 1]^{K \times K}$, where $T_{ij} = P(\tilde{Y}=j \mid Y=i)$. 
Assume $T$ is invertible. Let $\vec{\ell}(f(x)) = [\ell(f(x), 1), \dots, \ell(f(x), K)]^\top \in \mathbb{R}^K$.
Define the corrected loss vector:
$$\vec{\tilde{\ell}}(f(x)) = T^{-1} \vec{\ell}(f(x))$$
Then:
$$\mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\tilde{\ell}(f(X), \tilde{Y})] = \mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)]$$

---

## 2. Proof

**Step 1: Expand the conditional expectation of the noisy loss given $X=x$.**

$$\mathbb{E}_{\tilde{Y} \mid X=x} [\tilde{\ell}(f(x), \tilde{Y})] = \sum_{j=1}^K P(\tilde{Y} = j \mid X=x) \tilde{\ell}(f(x), j)$$

Using vector notation, let $\vec{\tilde{\eta}}(x) = [P(\tilde{Y}=1 \mid X=x), \dots, P(\tilde{Y}=K \mid X=x)]^\top \in \Delta^{K-1}$.
Then the sum is the inner product:
$$\mathbb{E}_{\tilde{Y} \mid X=x} [\tilde{\ell}(f(x), \tilde{Y})] = \vec{\tilde{\eta}}(x)^\top \vec{\tilde{\ell}}(f(x))$$

**Step 2: Relate corrupted posterior $\vec{\tilde{\eta}}(x)$ to true clean posterior $\vec{\eta}(x)$.**

By the law of total probability and the class-conditional noise assumption:
$$P(\tilde{Y} = j \mid X=x) = \sum_{i=1}^K P(\tilde{Y} = j \mid Y = i, X=x) P(Y = i \mid X=x) = \sum_{i=1}^K T_{ij} \eta_i(x)$$
In matrix form:
$$\vec{\tilde{\eta}}(x) = T^\top \vec{\eta}(x)$$

**Step 3: Substitute $\vec{\tilde{\eta}}(x)$ and $\vec{\tilde{\ell}}(f(x))$ into the inner product.**

$$\vec{\tilde{\eta}}(x)^\top \vec{\tilde{\ell}}(f(x)) = (T^\top \vec{\eta}(x))^\top (T^{-1} \vec{\ell}(f(x)))$$

Applying the transpose property $(A B)^\top = B^\top A^\top$:
$$(T^\top \vec{\eta}(x))^\top = \vec{\eta}(x)^\top T$$

Therefore:
$$\vec{\tilde{\eta}}(x)^\top \vec{\tilde{\ell}}(f(x)) = \vec{\eta}(x)^\top T T^{-1} \vec{\ell}(f(x))$$

Since $T T^{-1} = I_K$ (the identity matrix):
$$\vec{\tilde{\eta}}(x)^\top \vec{\tilde{\ell}}(f(x)) = \vec{\eta}(x)^\top I_K \vec{\ell}(f(x)) = \vec{\eta}(x)^\top \vec{\ell}(f(x))$$

**Step 4: Expand the clean conditional expectation.**

$$\vec{\eta}(x)^\top \vec{\ell}(f(x)) = \sum_{i=1}^K P(Y=i \mid X=x) \ell(f(x), i) = \mathbb{E}_{Y \mid X=x} [\ell(f(x), Y)]$$

**Step 5: Integrate over the marginal distribution of $X$.**

By the Law of Iterated Expectations:
$$\mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\tilde{\ell}(f(X), \tilde{Y})] = \mathbb{E}_X [\mathbb{E}_{\tilde{Y} \mid X} [\tilde{\ell}(f(X), \tilde{Y})]] = \mathbb{E}_X [\mathbb{E}_{Y \mid X} [\ell(f(X), Y)]] = \mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)]$$

This completes the proof. $\blacksquare$
