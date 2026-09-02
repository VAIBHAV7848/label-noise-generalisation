# Algorithm Specification: Transition Matrix Estimation & Loss Correction

## Algorithm 1: Anchor-Point Noise Transition Matrix Estimation

```
Input:
  - Noisy Training Dataset \tilde{S} = {(x_n, \tilde{y}_n)}_{n=1}^N
  - Base neural network f_theta trained on \tilde{S} using standard cross-entropy
  - Number of classes K
  - Confidence percentile threshold alpha (default 97th percentile)

Output:
  - Estimated Transition Matrix \hat{T} in [0, 1]^(K x K)

Procedure:
  1. Initialize \hat{T} as a zeros matrix in R^(K x K)
  2. Compute predicted posterior probabilities for all training instances:
       P_nj = [f_theta(x_n)]_j  for n in {1, ..., N}, j in {1, ..., K}
  3. For each class i in {1, ..., K}:
       a. Let S_i = { x_n in \tilde{S} : \tilde{y}_n = i }
       b. Identify top-alpha percentile anchor points x in S_i maximizing P_ni
       c. Average the predicted posterior vectors over identified anchor set:
            \hat{T}_{ij} = (1 / |Anchors_i|) * sum_{x_n in Anchors_i} P_nj
  4. Normalize rows to ensure valid transition probabilities:
       \hat{T}_{ij} <- \hat{T}_{ij} / sum_{k=1}^K \hat{T}_{ik}  for all i, j
  5. Return \hat{T}
```

---

## Algorithm 2: Forward & Backward Loss-Corrected Training

```
Input:
  - Noisy Training Dataset \tilde{S}
  - Estimated Transition Matrix \hat{T}
  - Model architecture f_theta
  - Optimizer (e.g. SGD with momentum, lr, weight decay)
  - Correction Mode in {FORWARD, BACKWARD}

Output:
  - Trained robust model parameters theta*

Procedure:
  1. If Correction Mode == BACKWARD:
       Compute matrix inverse \hat{T}^(-1)
  2. For epoch = 1 to MaxEpochs:
       For each mini-batch {(x_b, \tilde{y}_b)}_{b=1}^B in \tilde{S}:
         a. Compute model softmax output: p_b = f_theta(x_b) in Delta^(K-1)
         b. If Correction Mode == FORWARD:
              p_corrected_b = \hat{T}^T * p_b
              loss = -(1/B) * sum_{b=1}^B log([p_corrected_b]_{\tilde{y}_b})
         c. Else if Correction Mode == BACKWARD:
              ell_vec_b = -log(p_b)  # Base cross entropy loss vector
              loss_b = [\hat{T}^(-1) * ell_vec_b]_{\tilde{y}_b}
              loss = (1/B) * sum_{b=1}^B loss_b
         d. Compute gradients: grad = nabla_theta(loss)
         e. Update parameters: theta <- OptimizerStep(theta, grad)
  3. Return theta*
```
