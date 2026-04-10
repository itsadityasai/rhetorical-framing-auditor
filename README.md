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
│   ├── split_triplets.py         # Constructs the left/center/right triplets & splits into 80/10/10
│   ├── parse_rst.py              # Generates RST constituency trees via isanlp_rst_v3
│   ├── build_facts.py            # Helper module for SBERT embeddings & cross-encoder extraction
│   ├── build_clusters.py         # Helper module for agglomerative clustering
│   ├── run_fact_clustering.py    # Main script: Clusters EDUs across triplets, prunes with cross-encoder
│   ├── build_dfi.py              # Legacy helper for bipartite extraction
│   ├── build_dfi_from_splits.py  # Main script: Extracts bipartite Coverage & Structural (DFI) features
│   └── modules/                  # Internal project utility classes (caching, clustering utils)
│
├── experiments/                  # 2. EXPERIMENTS (Appendices D-E)
│   ├── 01_full_classification/   # → Exp 1, 2, 3, 7: Train RF, SVM, LR, MLP on bipartite features
│   │   └── train_dfi_alternatives.py 
│   ├── 02_pure_3way_analysis/    # → Exp 4, 6: Isolates 3-way clusters to eliminate omission signals
│   │   └── train_rst_only.py     
│   ├── 03_explainability_demo/   # → Exp 8: Saabas treeinterpreter mapping predictions to explicit EDUs
│   │   └── explain_predictions.py    
│   ├── bert_baseline.py          # Baseline: Semantic baseline modeling
│   ├── run_svm.py                # Baseline: Early SVM-only experiments
│   └── train_svm_from_dfi_splits.py  # Baseline: Padded DFI vector training
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
│   └── ...                       # Exploratory methodologies and abandoned pathways
│
└── params.yaml                   # Global project configuration parameters
```

---

## 🚀 Step-by-Step Execution Guide

To reproduce the findings from the paper, follow these steps in chronological order. All scripts should be run from the root directory of the project. Make sure your environment is configured and dependencies are installed.

### Phase 1: Data Processing & Pipeline

**Step 1: Triplet Construction & Dataset Splitting (Appendix A)**  
This script pairs articles into (Left, Center, Right) triplets covering the exact same event using cosine similarity, and splits them securely into 80/10/10 train/validation/test sets without data leakage.
```bash
python pipeline/split_triplets.py
```

**Step 2: Discourse Parsing (Appendix A)**  
Generates RST constituency trees over Elementary Discourse Units (EDUs) for all articles in the valid triplets using `isanlp_rst_v3`.
```bash
python pipeline/parse_rst.py
```

**Step 3: Fact Clustering (Appendix B)**  
Embeds the EDUs using SBERT, clusters them to identify shared facts across the triplet, and rigidly prunes hallucinated matches using a cross-encoder threshold.
```bash
python pipeline/run_fact_clustering.py
```

**Step 4: Bipartite Feature Extraction (Appendix C)**  
Extracts the explicit Coverage ($\delta_{cov}$) and Structural ($\delta_{str}$) features via our 1-to-1-to-1 matching heuristic, saving the zero-padded DFI vectors to the `data/` directory.
```bash
python pipeline/build_dfi_from_splits.py
```

---

### Phase 2: Running the Experiments

Once the pipeline artifacts are generated, you can run the isolated experiments detailed in **Section 5** of the paper.

**Experiments 1, 2, 3, & 7: Full Dataset Classification (Appendix D)**  
Trains the main Random Forest model (alongside SVM, LR, and MLP baselines) on the full bipartite vectors. This reproduces our primary finding of **89.77% accuracy**.
```bash
python experiments/01_full_classification/train_dfi_alternatives.py
```

**Experiments 4 & 6: Pure 3-Way Cluster Analysis (Appendix D)**  
Isolates ONLY the facts that were reported by all three political orientations (eliminating the selection/omission bias signal). This proves that structural framing alone is a weak predictor, dropping accuracy to **~61.46%**.
```bash
python experiments/02_pure_3way_analysis/train_rst_only.py
```

**Experiment 8: Explainable Prediction Demo (Appendix E)**  
Leverages the `treeinterpreter` library (Saabas method) on our trained Random Forest model. It outputs concrete, human-readable reports mapping prediction logits directly to explicit EDUs (e.g., "Center mentioned X, but Right omitted it").
```bash
python experiments/03_explainability_demo/explain_predictions.py
```

---

## 📖 Citation

If you use this codebase or methodology in your research, please refer to the primary manuscript located at `docs/acl_latex3.tex`.
