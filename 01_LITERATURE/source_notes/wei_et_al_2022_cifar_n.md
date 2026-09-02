# Source Note: Wei et al. (ICLR 2022)

- **Title**: Learning with Noisy Labels Revisited: A Study on ImageNet and CIFAR-10N
- **Authors**: Jiaheng Wei, Zhaowei Zhu, Hao Cheng, Tongliang Liu, Gang Niu, Yang Liu
- **Venue**: International Conference on Learning Representations (ICLR 2022)
- **DOI / URL**: https://openreview.net/forum?id=TBWA6PLJZQm

---

## 1. Problem Studied
The critical divergence between synthetic label noise (e.g. uniform or class-conditional matrices) and real-world human annotator noise. Demonstrates that prior algorithms tuned exclusively on synthetic noise frequently collapse or perform sub-optimally when evaluated on real-world crowdsourced error.

---

## 2. Key Contributions

### 2.1 CIFAR-10N and CIFAR-100N Benchmarks
The authors collected real human annotations for all 50,000 training images in CIFAR-10 from Amazon Mechanical Turk:
- **Clean Label**: True ground truth.
- **Aggregate Label**: Majority vote across 3 human annotators (Noise rate: $\approx 9.03\%$).
- **Random Label 1/2/3**: Single annotator noisy labels (Noise rate: $\approx 17-18\%$).
- **Worst Label**: Least accurate annotator's label (Noise rate: $\approx 40.21\%$).

### 2.2 Instance-Dependent Nature of Real Noise
Real human error is shown to be strongly instance-dependent ($P(\tilde{Y} \mid X, Y)$), concentrated on ambiguous, low-resolution, or boundary instances rather than uniformly distributed across class categories.

---

## 3. Relevance to Our Project
CIFAR-10N serves as our definitive real-world noisy benchmark to evaluate whether theoretical findings and algorithms developed under synthetic class-conditional matrices hold on human-generated corruptions.
