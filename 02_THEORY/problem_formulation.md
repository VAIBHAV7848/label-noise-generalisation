# Mathematical Problem Formulation

## 1. Clean Distribution & Risk Minimization

Let $\mathcal{X} \subseteq \mathbb{R}^d$ denote the input feature space and $\mathcal{Y} = \{1, 2, \dots, K\}$ denote the label space for $K$-class classification. 

Let $\mathcal{D}$ be an unknown true joint distribution over $\mathcal{X} \times \mathcal{Y}$. A classifier is represented by a hypothesis function $f: \mathcal{X} \to \Delta^{K-1}$, where $\Delta^{K-1} = \{ \mathbf{p} \in \mathbb{R}^K : p_k \ge 0, \sum_{k=1}^K p_k = 1 \}$ is the probability simplex. The predicted label is $\hat{y}(x) = \arg\max_{k \in \mathcal{Y}} f_k(x)$.

For a given loss function $\ell: \Delta^{K-1} \times \mathcal{Y} \to \mathbb{R}_+$, the **Clean Expected Risk** is defined as:
$$R(f) = \mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)] = \mathbb{E}_X \left[ \sum_{k=1}^K \eta_k(X) \ell(f(X), k) \right]$$
where $\eta_k(x) = P(Y=k \mid X=x)$ is the true class posterior probability.

The **Bayes Optimal Classifier** $f^*$ minimizes the clean risk under 0-1 loss $\ell_{0-1}(f(x), y) = \mathbb{I}(\arg\max_k f_k(x) \ne y)$:
$$f^*(x) = \arg\max_{k \in \mathcal{Y}} \eta_k(x)$$

---

## 2. Noisy Distribution & Transition Matrix Model

In the presence of label noise, we do not observe samples from $\mathcal{D}$. Instead, we observe samples $(X, \tilde{Y})$ drawn from a corrupted distribution $\tilde{\mathcal{D}}$, where $\tilde{Y} \in \mathcal{Y}$ is the corrupted noisy label.

### 2.1 Class-Conditional Noise Transition Matrix
Under the Class-Conditional Noise (CCN) model, the corruption probability depends exclusively on the true latent class $Y$:
$$P(\tilde{Y}=j \mid X=x, Y=i) = P(\tilde{Y}=j \mid Y=i) = T_{ij}$$
where $T \in [0, 1]^{K \times K}$ is the **Noise Transition Matrix** satisfying:
$$\sum_{j=1}^K T_{ij} = 1, \quad \forall i \in \{1, \dots, K\}$$

### 2.2 Corrupted Posterior Probability
The corrupted posterior distribution $\tilde{\eta}_j(x) = P(\tilde{Y}=j \mid X=x)$ relates to the clean posterior $\eta_i(x) = P(Y=i \mid X=x)$ by the Law of Total Probability:
$$\tilde{\eta}_j(x) = \sum_{i=1}^K P(\tilde{Y}=j \mid Y=i, X=x) P(Y=i \mid X=x) = \sum_{i=1}^K T_{ij} \eta_i(x)$$
In compact vector notation:
$$\vec{\tilde{\eta}}(x) = T^\top \vec{\eta}(x)$$
where $\vec{\eta}(x) = [\eta_1(x), \dots, \eta_K(x)]^\top \in \Delta^{K-1}$.

---

## 3. Noisy Expected Risk & Empirical Risk Minimization

The **Noisy Expected Risk** under corrupted distribution $\tilde{\mathcal{D}}$ is:
$$R_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\ell(f(X), \tilde{Y})] = \mathbb{E}_X \left[ \sum_{j=1}^K \tilde{\eta}_j(X) \ell(f(X), j) \right]$$

Given an observed noisy training dataset $\tilde{S} = \{(x_1, \tilde{y}_1), \dots, (x_n, \tilde{y}_n)\} \stackrel{i.i.d.}{\sim} \tilde{\mathcal{D}}$, standard Empirical Risk Minimization (ERM) minimizes:
$$\hat{R}_{\tilde{S}}(f) = \frac{1}{n} \sum_{i=1}^n \ell(f(x_i), \tilde{y}_i)$$

### The Core Theoretical Discrepancy
Because $\vec{\tilde{\eta}}(x) = T^\top \vec{\eta}(x) \ne \vec{\eta}(x)$ when $T \ne I$, the minimizer of noisy risk $\tilde{f}^* = \arg\min_f R_{\tilde{\mathcal{D}}}(f)$ generally does not coincide with the clean Bayes optimal classifier $f^* = \arg\min_f R_{\mathcal{D}}(f)$. The resulting sub-optimality is quantified by the **Excess Risk**:
$$\mathcal{E}(\hat{f}) = R(\hat{f}) - \min_{f \in \mathcal{F}} R(f)$$
