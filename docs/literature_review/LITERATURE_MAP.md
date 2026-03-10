# MI-EEG Literature Map (2024-2025): Positioning LLM-EEG v2

> **Scope**: 9 attached papers + 6 newly identified papers + LLM-EEG v2  
> **Date**: 2026-02-07

---

## 1. Landscape Overview

```
                            MI-EEG CLASSIFICATION LANDSCAPE (2024-2025)
                            ============================================

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                 │
    │   CLASSIFICATION ACCURACY (BCI IV-2a, 4-class)                                  │
    │                                                                                 │
    │   90%+ ─────────────────────────────────────────────────── (ceiling)             │
    │                                                                                 │
    │   86% ── Transformer Model (86.46% subj-dep)                                    │
    │   85% ── CIACNet (85.15%), Multi-day (85.32% 2C)                                │
    │   84% ── SATrans-Net (84.72%)                                                   │
    │   83% ── EEG-DCNet (83.31%), CLTNet (83.02%),                                   │
    │          MSCFormer (82.95%), Transformer-GCN (82.9%), DB-BISAN (~83%)            │
    │   81% ── AMEEGNet (81.17%)                                                      │
    │   80% ── BrainGridNet (80.26%)                                                  │
    │                                                                                 │
    │   ···    (accuracy gap)                                                         │
    │                                                                                 │
    │   67% ── LLM-EEG v2 (66.67% — synthetic/limited validation)                    │
    │                                                                                 │
    │   UNIQUE INNOVATION ────────────────────────────────────────────────             │
    │   RL Preprocessing (APA) ── LLM-EEG v2 (ONLY)                                  │
    │   Decision Validation (DVA) ── LLM-EEG v2 (ONLY)                               │
    │   LLM Explainability ── LLM-EEG v2 (Phi-3), NeuroLM (Core decoder)             │
    │   SHAP+PLV Interpret. ── GAT-GRU-Transformer                                   │
    │   Grad-CAM ── SATrans-Net                                                       │
    │   Cross-Subject ── ADFR (76.48%), Transformer Model (74.48%)                    │
    │                                                                                 │
    └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Method Classification by Architecture Type

### 2.1 CNN-Dominant Methods
| Method | Year | Key CNN Innovation | Acc (IV-2a) |
|--------|------|--------------------|:-----------:|
| AMEEGNet | 2025 | Multi-scale EEGNet + ECA | 81.17% |
| EEG-DCNet | 2024 | Dilated (atrous) convolutions + SE | 83.31% |
| BrainGridNet | 2025 | Two-branch depthwise + grid encoding | 80.26% |
| Feature Reweighting | 2025 | Channel reweighting mechanism | N/R |

### 2.2 Transformer-Dominant / Hybrid Methods
| Method | Year | Key Transformer Innovation | Acc (IV-2a) |
|--------|------|-----------------------------|:-----------:|
| Transformer Model | 2025 | TCN + Transformer (DSTS) | 86.46% |
| MSCFormer | 2025 | Multi-scale Conv + Transformer | 82.95% |
| SATrans-Net | 2025 | Top-K Sparse Attention Transformer | 84.72% |
| CLTNet | 2025 | CNN + LSTM + Transformer | 83.02% |
| DB-BISAN | 2025 | Blocked-Integration Self-Attention | ~83% |

### 2.3 Graph Neural Network Methods
| Method | Year | Key GNN Innovation | Acc |
|--------|------|--------------------|:-----------:|
| Transformer-GCN | 2025 | GCN + PCC/COH/PLV adjacency | 82.9% (IV-2a) |
| GAT-GRU-Transformer | 2025 | Hierarchical GAT + GRU + Transformer | 55.76% (5-finger) |

### 2.4 Domain Adaptation / Transfer Learning
| Method | Year | Key DA Innovation | Cross-Subj Acc |
|--------|------|-------------------|:--------------:|
| ADFR | 2024 | MMD + IDFL + Entropy Minimization | 76.48% |
| Transformer Model | 2025 | Subject-independent evaluation | 74.48% |

### 2.5 Pipeline / Agent / LLM-Based
| Method | Year | Key Innovation | Unique Features |
|--------|------|----------------|:---------------:|
| **LLM-EEG v2** | 2025 | RL+DVA+LLM composable pipeline | APA, DVA, Phi-3 |
| NeuroLM | 2025 | LLM as EEG decoder (foundation model) | Multi-task LLM |

### 2.6 Dataset Contributions
| Method | Year | Contribution |
|--------|------|-------------|
| Multi-day EEG Study | 2025 | New multi-day, high-quality MI-EEG dataset |

---

## 3. Feature Presence Matrix (Visual)

```
Feature                    | LLM-EEG | AMEEGNet | DCNet | FeatRew | CIACNet | CLTNet | MultiDay | TransfModel | MSCFormer | BrainGrid | SATrans | T-GCN | DB-BISAN | ADFR | GAT-GRU-T | NeuroLM
========================== | ======= | ======== | ===== | ======= | ======= | ====== | ======== | =========== | ========= | ========= | ======= | ===== | ======== | ==== | ========= | =======
RL Adaptive Preproc (APA)  |    X    |          |       |         |         |        |          |             |           |           |         |       |          |      |           |
Decision Validation (DVA)  |    X    |          |       |         |         |        |          |             |           |           |         |       |          |      |           |
LLM Explainability         |    X    |          |       |         |         |        |          |             |           |           |         |       |          |      |           |    X*
Signal Quality Metrics     |    X    |          |       |         |         |        |          |             |           |           |         |       |          |      |           |
Composable Interfaces      |    X    |          |       |         |         |        |          |             |           |           |         |       |          |      |           |
Multi-Classifier Support   |    X    |          |       |         |         |        |          |             |           |           |         |       |          |      |           |
Attention Mechanisms        |         |    X     |   X   |    X    |    X    |   X    |          |      X      |     X     |           |    X    |   X   |    X     |      |     X     |    X
Multi-Scale Features        |         |    X     |   X   |         |         |        |          |             |     X     |     X     |         |       |    X     |      |     X     |
Cross-Subject Eval          |         |          |       |         |         |        |    ~     |      X      |           |           |         |   X   |          |   X  |           |    X
Grad-CAM / SHAP / PLV      |         |          |       |         |         |        |          |             |           |           |    X    |       |          |      |     X     |
End-to-End Deep Learning    |         |    X     |   X   |    X    |    X    |   X    |    X     |      X      |     X     |     X     |    X    |   X   |    X     |   X  |     X     |    X

X = present; X* = LLM as core decoder (different from explainability)
```

---

## 4. Research Gap Analysis

### 4.1 Gaps that LLM-EEG v2 Fills
| Gap | Description | How LLM-EEG v2 Addresses |
|-----|-------------|--------------------------|
| **No RL in BCI preprocessing** | All papers use fixed preprocessing | APA learns optimal preprocessing profiles via Q-learning |
| **No prediction reliability filter** | All papers report all predictions | DVA filters unreliable predictions (Accept/Reject/Review) |
| **No natural-language BCI explainability** | SHAP/Grad-CAM are visual; no NLP explanations | Phi-3 LLM generates textual explanations |
| **No composable BCI software architecture** | All papers are monolithic models | Interface-driven design (IDataLoader, IPreprocessor, etc.) |

### 4.2 Gaps in LLM-EEG v2 (Opportunities)
| Gap | Description | Papers That Address It |
|-----|-------------|----------------------|
| **SOTA classification accuracy** | 66.67% vs 85%+ | CIACNet, Transformer Model, SATrans-Net |
| **Cross-subject generalization** | Not evaluated | ADFR, Transformer Model |
| **Multi-dataset validation** | Only BCI IV-2a | AMEEGNet, SATrans-Net (3 datasets) |
| **End-to-end deep learning** | Uses CSP+traditional ML primarily | All DL papers |
| **Multi-scale feature extraction** | Single CSP scale | AMEEGNet, MSCFormer, BrainGridNet |
| **Graph-based spatial modeling** | CSP only | Transformer-GCN, GAT-GRU-T |

---

## 5. Recommended Integration Paths

Based on the literature analysis, the following integration paths would strengthen LLM-EEG v2:

1. **Replace CSP with multi-scale CNN features** (inspired by AMEEGNet, MSCFormer) to improve accuracy
2. **Add domain adaptation module** (inspired by ADFR) for cross-subject generalization
3. **Integrate sparse attention** (inspired by SATrans-Net) for efficient temporal modeling
4. **Add Grad-CAM/SHAP alongside LLM** (inspired by SATrans-Net, GAT-GRU-T) for multi-modal explainability
5. **Evaluate on BCI IV-2b and HGD** to match the multi-dataset validation standard

---

## 6. Citation Summary

| # | Method | Year | Source | DOI/URL |
|---|--------|------|--------|---------|
| 1 | LLM-EEG v2 | 2025 | This work | https://github.com/erlika/llm-eeg |
| 2 | AMEEGNet | 2025 | Attached | [Attached PDF] |
| 3 | EEG-DCNet | 2024 | Attached | https://arxiv.org/abs/2411.17705 |
| 4 | Feature Reweighting | 2025 | Attached | [Attached XML] |
| 5 | CIACNet | 2025 | Attached | https://pmc.ncbi.nlm.nih.gov/articles/PMC11841462/ |
| 6 | CLTNet | 2025 | Attached | [Attached XML] |
| 7 | Multi-day EEG | 2025 | Attached | [Attached XML] |
| 8 | Transformer Model | 2025 | Attached | [Attached XML] |
| 9 | MSCFormer | 2025 | Attached | [Attached XML] |
| 10 | BrainGridNet | 2025 | Attached | [Attached XML] |
| 11 | SATrans-Net | 2025 | Web Research | https://www.nature.com/articles/s41598-025-30806-8 |
| 12 | Transformer-GCN | 2025 | Web Research | https://www.sciencedirect.com/science/article/abs/pii/S1568494624014601 |
| 13 | DB-BISAN | 2025 | Web Research | https://www.mdpi.com/2078-2489/16/7/582 |
| 14 | ADFR | 2024 | Web Research | https://link.springer.com/article/10.1186/s12859-024-06024-w |
| 15 | GAT-GRU-Transformer | 2025 | Web Research | https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2025.1599960/full |
| 16 | NeuroLM | 2025 | Web Research | https://openreview.net/forum?id=Io9yFt7XH7 |

**Dataset References**:
- BCI Competition IV-2a: https://www.bbci.de/competition/iv/#download
- High Gamma Dataset: https://braindecode.org/stable/generated/braindecode.datasets.HGD.html

---

*End of Literature Map*
