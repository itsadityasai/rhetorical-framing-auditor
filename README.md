# Rhetorical Framing Auditor

This repository contains the data pipeline and experiments for analyzing media bias through:

- fact omission (coverage asymmetry), and
- rhetorical framing (RST structural positioning).

## What this repository contains

### Core directories

- [pipeline](pipeline): dataset construction and feature generation pipeline.
	- [pipeline/split_triplets.py](pipeline/split_triplets.py): builds left/center/right triplets and splits.
	- [pipeline/parse_rst.py](pipeline/parse_rst.py): parses documents into RST outputs.
	- [pipeline/run_fact_clustering.py](pipeline/run_fact_clustering.py): clusters semantically aligned EDUs.
	- [pipeline/build_dfi_from_splits.py](pipeline/build_dfi_from_splits.py): builds DFI rows using predefined split files.
	- [pipeline/modules](pipeline/modules): canonical implementations (`FactCluster`, `DFIGenerator`, run logger).

- [experiments](experiments): active experimental scripts.
	- [experiments/01_full_classification](experiments/01_full_classification): full classification experiments.
	- [experiments/02_pure_3way_analysis](experiments/02_pure_3way_analysis): omission-controlled structural analysis.
	- [experiments/03_explainability_demo](experiments/03_explainability_demo): explainability artifacts.
	- Additional tracks now under `experiments/`: `omission-based`, `strengthen-str`, `aggregate-vector`, `aggregate-rf`, `hybrid-approach`, `experimental-design`, `alternative-models`, `ordering-str`, `universal-str`, and `improved-clustering`.

- [data](data): generated artifacts and cached intermediate files.
	- `valid_*` files/dirs are active reclustered artifacts used by current workflows.
	- [data/ablation](data/ablation): ablation results and model checkpoints.

- [docs](docs): manuscript and paper assets.
- [presentation](presentation): slides and generated figures.

### Root-level files

- [params.yaml](params.yaml): central configuration.
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md): operational project context and artifact lineage.
- [GPU_FRESH_CLUSTERING_TRAINING_INSTRUCTIONS.txt](GPU_FRESH_CLUSTERING_TRAINING_INSTRUCTIONS.txt): runbook for fresh clustering/training.

## Environment and execution

Run all commands from repository root.

Example environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If a requirements file is not present, install dependencies used by the scripts you run (e.g., `numpy`, `scikit-learn`, `pyyaml`, `sentence-transformers`, `transformers`).

## Recommended run order

### 1) Build/refresh pipeline artifacts

```bash
python pipeline/split_triplets.py
python pipeline/parse_rst.py
python pipeline/run_fact_clustering.py
python data/build_facts_from_clusters.py \
	--clusters data/valid_cluster_results_recluster_gpu.json \
	--out data/valid_facts_results_recluster_gpu.json \
	--meta data/valid_facts_results_recluster_gpu_meta.json
python pipeline/build_dfi_from_splits.py \
	--facts data/valid_facts_results_recluster_gpu.json \
	--split-dir data/valid_triplet_splits \
	--out-dir data/valid_dfi_splits_recluster_gpu
```

### 2) Run active experiments

```bash
python experiments/01_full_classification/train_dfi_alternatives.py
python experiments/02_pure_3way_analysis/train_rst_only.py
python experiments/03_explainability_demo/explain_predictions.py
python experiments/run_structural_ablation.py
python experiments/run_structural_ablation_size3.py
```

## Notes on path conventions

- Active scripts write outputs under `experiments/.../results` or `data/...`.
- Formerly archived scripts now live under `experiments/...` and write outputs under `experiments/.../results`.
- Compatibility imports via `modules.*` are provided by wrappers in [modules](modules) that re-export implementations from [pipeline/modules](pipeline/modules).

## Citation

For manuscript details, see [docs/acl_latex3.tex](docs/acl_latex3.tex).
