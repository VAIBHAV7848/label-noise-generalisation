# Seed Protocol & Determinism Specification

## 1. Random Seed Set
All experiments are strictly repeated across the following 5 seeds:

```python
EXPERIMENT_SEEDS = [42, 1337, 2024, 7, 999]
```

---

## 2. Determinism Enforcement Policy
In all subsequent implementation code (Phase 1+), determinism must be enforced via the standardized seeding routine:

```python
import os
import random
import numpy as np
import torch

def set_deterministic_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
