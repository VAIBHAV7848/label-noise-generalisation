# The Effect of Label Noise on Generalisation: A Theoretical and Empirical Analysis for Classifiers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Phase](https://img.shields.io/badge/Research_Phase-Phase_0:_Foundation_&_Architecture-blue.svg)](#)
[![Traceability](https://img.shields.io/badge/Traceability-Full_Evidence_Audit-success.svg)](#)

## 1. Project Overview

This repository houses the formal research investigation: **"The Effect of Label Noise on Generalisation: A Theoretical and Empirical Analysis for Classifiers"**. 

The primary objective is to investigate the exact mechanisms by which label noise alters empirical risk minimization, shifts optimal Bayes decision boundaries, induces statistical bias, degrades prediction calibration, and drives capacity-dependent memorisation across distinct classifier families.

The repository serves as the **Single Source of Truth (SSOT)** for all mathematical derivations, empirical protocols, adversarial novelty audits, literature matrices, decision logs, and experimental records.

---

## 2. Research Principles & Methodology

1. **Evidence over Assumptions**: No claims of novelty or performance are accepted without mathematical proof or statistically rigorous empirical validation ($p$-values, confidence intervals, multi-seed variance).
2. **Strict Academic Traceability**: Every design choice, mathematical assumption, dataset selection, and baseline inclusion is cross-referenced with foundational literature and recorded in `10_DECISIONS/decision_log.md`.
3. **No Hallucinated or Saturated Claims**: Saturated domains (e.g., simplistic 2-layer MLPs on synthetic MNIST symmetric noise) are explicitly flagged, audited, and contrasted with realistic noise distributions and modern benchmarks.
4. **Reproducibility by Design**: Complete determinism across seeds, hardware environments, dependency lockfiles, and configuration files.

---

## 3. Repository Architecture

```
label-noise-generalisation/
│
├── README.md                           # Master repository documentation & index
│
├── 00_PROJECT/                         # Project charter & high-level specification
│   ├── research_question.md            # Primary & secondary research questions
│   ├── objectives.md                   # Concrete theoretical and empirical goals
│   ├── scope.md                        # Scope boundaries, inclusions & exclusions
│   ├── hypotheses.md                   # Formally testable scientific hypotheses
│   ├── research_decisions.md           # Foundational structural decisions
│   └── roadmap.md                      # Phased execution timeline
│
├── 01_LITERATURE/                      # Exhaustive literature review & taxonomies
│   ├── literature_matrix.csv           # Structured matrix of 25+ primary papers
│   ├── related_work.md                 # Thematic synthesis of established literature
│   ├── research_gap.md                 # Genuine unresolved gaps vs. saturated topics
│   └── source_notes/                   # Deep technical notes for pivotal papers
│
├── 02_THEORY/                          # Mathematical foundations & proofs
│   ├── problem_formulation.md          # Clean vs. noisy risk, Bayes optimality
│   ├── assumptions.md                  # Noise transition models, identifiability
│   ├── propositions.md                 # Formal theoretical propositions
│   ├── derivations/                    # Step-by-step risk & loss derivations
│   ├── proofs/                         # Complete proofs for unbiased estimators
│   └── theory_summary.md               # Unified theoretical synthesis
│
├── 03_METHOD/                          # Methodological specifications
│   ├── proposed_method.md              # Systematic framework for evaluation
│   ├── transition_matrix.md            # Transition matrix $T$ estimation & bounds
│   ├── noise_models.md                 # Symmetric, class-conditional & instance-dependent
│   ├── algorithms/                     # Algorithmic pseudocode and workflows
│   └── design_decisions.md             # Methodological rationale
│
├── 04_EXPERIMENTS/                     # Empirical experimental design
│   ├── datasets.md                     # Synthetic & real-world noisy benchmark specs
│   ├── experimental_protocol.md        # Seed protocols, cross-validation & hardware specs
│   ├── baselines.md                    # Standard CE, robust losses, loss correction, sample selection
│   └── ablations.md                    # Planned ablation dimensions & sensitivity tests
│
├── 05_RESULTS/                         # Experimental results (populated post-Phase 0)
│   ├── raw/                            # Untouched raw log files & runs
│   ├── processed/                      # Aggregated metric JSONs/CSVs
│   ├── tables/                         # Formatted LaTeX/Markdown tables
│   ├── figures/                        # High-resolution vector plots
│   └── statistical_analysis/           # Significance tests & bootstrap CIs
│
├── 06_ANALYSIS/                        # Analytical findings & discussion
│   ├── observations.md                 # Empirical observations
│   ├── interpretation.md               # Scientific interpretation & mechanism analysis
│   ├── limitations.md                  # Theoretical & experimental limitations
│   └── claim_evidence_map.md           # Bidirectional link between claims and proof
│
├── 07_REVIEW/                          # Adversarial peer review & quality gates
│   ├── novelty_audit.md                # Adversarial novelty check against prior art
│   ├── claim_verification.md           # Verification checklist for all manuscript claims
│   ├── weaknesses.md                   # Known weaknesses & reviewer attack surfaces
│   └── reviewer_simulation/            # Simulated Tier-1 conference review reports
│
├── 08_PAPER/                           # Publication manuscript drafts & camera-ready
│   ├── outline.md                      # Target conference paper outline (NeurIPS/ICML format)
│   ├── manuscript/                     # LaTeX sources
│   └── references/                     # BibTeX reference files
│
├── 09_REPRODUCIBILITY/                 # Reproducibility artifacts
│   ├── environment.yml                 # Conda environment definition
│   ├── requirements.txt                # Pinned pip requirements
│   ├── seeds.md                        # Deterministic seed assignments
│   ├── reproduction.md                 # Exact reproduction instructions
│   └── config/                         # YAML run configurations
│
├── 10_DECISIONS/                       # Research governance & historical record
│   ├── decision_log.md                 # Chronological log of accepted research decisions
│   └── rejected_ideas.md               # Catalog of rejected avenues with literature proof
│
└── src/                                # Source code scaffold (Phase 1+ execution)
```

---

## 4. Current Phase Status: Phase 0 (Research Foundation)

- **Phase 0 Objective**: Establish whether a strong, defensible, non-saturated research question exists, formalize the mathematics, document the literature matrix, audit novelty adversarially, and define the complete experimental blueprint.
- **Phase 0 Rule**: Zero code implementation, zero model training, zero fabricated results.
- **Next Milestone**: External research review & approval of the Phase 0 Research Blueprint prior to launching Phase 1 (Core Implementation).
