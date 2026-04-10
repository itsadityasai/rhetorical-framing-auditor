# Rhetorical Framing Auditor

[![ACL 2024](https://img.shields.io/badge/Paper-ACL_2024-blue.svg)](docs/acl_latex3.tex)

This repository contains the complete experimental pipeline and analysis code for the paper: **"Fact Omission vs. Rhetorical Framing: Disentangling the Predictors of Media Bias."**

We investigate whether media bias is better predicted by *fact omission* (selection bias) or by the rhetorical positioning of shared facts (*framing bias*) using Rhetorical Structure Theory (RST). Our key finding is that fact omission accounts for approximately 72% of the predictive power for media bias, while RST structural framing accounts for only 28%. Left and right sources share remarkably few facts (average 3.95 per triplet).

---

## 📁 Repository Structure

The codebase has been meticulously reorganized to perfectly map to the experiments and methodology detailed in our paper. 

```text
rhetorical-framing-auditor/
├── pipeline/                     # 1. DATA & PIPELINE (Appendices A-C)
│   ├── split_triplets.py         # Constructs the left/center/right triplets, ensures non-leakage (80/10/10 splits)
│   ├── parse_rst.py              # Generates RST constituency trees via isanlp_rst_v3
│   ├── build_facts.py            # Helper module for SBERT embeddings & cross-encoder extraction
│   ├── build_clusters.py         # Helper module for agglomerative clustering
│   ├── run_fact_clustering.py    # Main script: Clusters EDUs across triplets, prunes with cross-encoder
│   ├── build_dfi.py              # Legacy helper for bipartite extraction
│   ├── build_dfi_from_splits.py  # Main script: Extracts bipartite Coverage & Structural (DFI) features
│   └── modules/                  # Internal project utility classes (caching, clustering utils)
│
├── experiments/                  # 2. EXPERIMENTS (Appendices D-E)
│   ├── 01_full_classification/   # → Exp 1 (Full Dataset), Exp 2 (DFI Alts), Exp 3 (Cluster Order), Exp 7 (Models)
│   │   └── train_dfi_alternatives.py # Trains RF, SVM, LR, MLP models on bipartite features
│   ├── 02_pure_3way_analysis/    # → Exp 4 (RST-only Features), Exp 6 (Pure 3-Way Cluster Analysis)
│   │   └── train_rst_only.py     # Isolates 3-way clusters to eliminate omission signals entirely
│   ├── 03_explainability_demo/   # → Exp 8 (Explainable Prediction Demo)
│   │   └── explain_predictions.py    # Uses Saabas treeinterpreter to map predictions back to exact EDUs
│   ├── bert_baseline.py          # Legacy/Baseline: Semantic baseline modeling
│   ├── run_svm.py                # Legacy/Baseline: Early SVM-only experiments
│   └── train_svm_from_dfi_splits.py  # Legacy/Baseline: Padded DFI vector training
│
├── presentation/                 # 3. PRESENTATION ARTIFACTS
│   ├── slides.tex                # Comprehensive Beamer presentation slides for the paper
│   ├── generate_slide_diagrams.py# Python script generating the pie/bar charts used in the slides
│   └── diagrams/                 # Generated PNG visual assets
│
├── docs/                         # 4. DOCUMENTATION
│   ├── acl_latex3.tex            # Finalized ACL-style paper (The central document)
│   └── custom.bib                # Bibliography references
│
├── data/                         # 5. GENERATED ARTIFACTS
│   ├── valid_facts_results_recluster_gpu.json # Final cached clusters & EDU lookups
│   └── valid_dfi_splits_recluster_gpu/        # Train/val/test feature matrices
│
├── archive/                      # 6. ARCHIVED EXPERIMENTS
│   └── ...                       # Older exploratory methodologies and abandoned pathways
│
├── params.yaml                   # Global project configuration parameters
└── GPU_FRESH_CLUSTERING_TRAINING_INSTRUCTIONS.txt # Instructions for running the clusterer from scratch
```

---

## 🧪 Mapping to the Paper

To reproduce specific sections of the paper, navigate to the corresponding directory:

### Methodology (Sections 3 & 4)
- **Dataset Construction & RST Parsing (App. A)**: `pipeline/split_triplets.py` and `pipeline/parse_rst.py`
- **Fact Clustering (App. B)**: `pipeline/run_fact_clustering.py`
- **Bipartite Feature Extraction (App. C)**: `pipeline/build_dfi_from_splits.py`

### Experiments (Section 5)
- **Experiments 1, 2, 3, & 7 (App. D)**: Go to `experiments/01_full_classification/train_dfi_alternatives.py`. This script dynamically trains the Random Forest classifier that achieved our state-of-the-art 89.77% accuracy using the bipartite coverage features.
- **Experiments 4 & 6 (App. D)**: Go to `experiments/02_pure_3way_analysis/train_rst_only.py`. This evaluates the baseline model on only facts covered by all three political orientations (where accuracy drops significantly to ~61%).
- **Experiment 8 (App. E)**: Go to `experiments/03_explainability_demo/explain_predictions.py`. This script outputs human-readable reports mapping Random Forest feature activations (using `treeinterpreter`) to explicit textual EDUs (e.g., specific omitted facts).

---

## 🚀 Quick Start

Ensure you have your environment configured (see `params.yaml` for paths and thresholds).

1. **Re-run Feature Extraction**:
   ```bash
   python pipeline/build_dfi_from_splits.py
   ```
2. **Train the Primary Model (Exp 1)**:
   ```bash
   cd experiments/01_full_classification
   python train_dfi_alternatives.py
   ```
3. **Generate Explainability Reports (Exp 8)**:
   ```bash
   cd experiments/03_explainability_demo
   python explain_predictions.py
   ```

*Note: For complete details on the theoretical framing, findings, and methodology, please compile and read `docs/acl_latex3.tex`.*
