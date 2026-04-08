# Project Context for Next AI Agent

Last updated: 2026-04-06
Repository: rhetorical-framing-auditor

## 1) What this project is about

Rhetorical Framing Auditor (RFA) studies framing bias across left/center/right news by comparing how aligned facts are positioned in discourse structure (RST).

Core idea:
- Build left-center-right triplets discussing similar events.
- Parse articles into EDUs + RST relations.
- Cluster semantically aligned EDUs across the three articles.
- Build DFI (Document Framing Index) vectors using prominence scoring:
  - $W(d,s)=\alpha^{d+1}\gamma^s$
  - where $d$ is depth and $s$ is satellite-edge count to root.
- Train binary classifier on left-vs-center and right-vs-center DFI examples.

## 2) Current project status snapshot

Current active pipeline artifacts are the fresh GPU reclustered valid-subset artifacts:
- `data/valid_triplets.json`
- `data/valid_triplet_splits/`
- `data/valid_cluster_results_recluster_gpu.json`
- `data/valid_cluster_results_recluster_gpu_meta.json`
- `data/valid_facts_results_recluster_gpu.json`
- `data/valid_facts_results_recluster_gpu_meta.json`
- `data/valid_dfi_splits_recluster_gpu/`
- `data/valid_svm_metrics_recluster_gpu.json`
- `data/ablation/structural_ablation_recluster_gpu.json`
- `data/ablation/structural_ablation_size3_recluster_gpu.json`

Most older artifacts were moved (not deleted) into `data/optional/`.

## 3) File and folder map (what everything is for)

### 3.1 Repository root

- `README.md`: old project overview. Partly outdated (mentions `main.py`, while current active workflow uses `run_fact_clustering.py` and split-based DFI scripts).
- `GPU_FRESH_CLUSTERING_TRAINING_INSTRUCTIONS.txt`: operational runbook for fresh reclustered pipeline.
- `params.yaml`: central config, but contains legacy paths for many scripts; some newer scripts use explicit fresh defaults in code.

Core scripts:
- `run_fact_clustering.py`: active clustering entrypoint for valid triplets; writes cluster artifact + meta.
- `build_dfi_from_splits.py`: builds DFI rows from facts and assigns rows to predefined leakage-safe split files.
- `train_svm_from_dfi_splits.py`: trains/evaluates SVM on DFI split files.
- `ablation.py`: unrestricted structural ablation (baseline, without_s, without_d, without_both) with model persistence.
- `ablation_size3_clusters.py`: coverage-controlled ablation using only tri-side clusters (default exact size-3 all-bias clusters).
- `split_triplets.py`: leakage-safe triplet splitting via component constraints and MILP.
- `build_clusters.py`: older cluster builder using params paths.
- `build_facts.py`: older fact builder using params paths.
- `build_dfi.py`: older end-to-end DFI build + split builder using params paths.
- `run_svm.py`: older SVM runner with optional grid search (legacy full-data workflow).
- `parse_rst.py`: RST parser pipeline for remaining raw docs.
- `bert_baseline.py`: 3-class article-level baseline script (left/center/right) using Hugging Face trainer.

### 3.2 `modules/`

- `modules/FactCluster.py`:
  - EDU filtering rules.
  - SBERT embedding + agglomerative clustering.
  - Cross-encoder pair validation.
  - Cluster retention constraints (center-required and min-bias constraints).
  - Fact enrichment from RST metadata (`depth`, `role`, `satellite_edges_to_root`).
- `modules/DFIGenerator.py`:
  - Prominence scoring $W(d,s)$.
  - Omission handling (`missing side -> score 0`).
  - DFI vector construction (`dfi_left`, `dfi_right`).
- `modules/run_logger.py`:
  - Standard run logging to `logs/<script_subdir>/<timestamp>.log` with tee to stdout/stderr.

### 3.3 `data/` (active)

- `data/raw/`: source article corpus.
- `data/rst_output/`: parsed RST JSONs per article.
- `data/valid_triplets.json`: filtered valid triplet set used by active workflow.
- `data/valid_triplets_meta.json`: metadata for valid triplet filtering.
- `data/valid_triplet_splits/`: leakage-safe split files (`train.json`, `val.json`, `test.json`, `meta.json`).
- `data/valid_cluster_results_recluster_gpu.json`: active fresh cluster results.
- `data/valid_cluster_results_recluster_gpu_meta.json`: clustering stats and EDU filter stats.
- `data/valid_facts_results_recluster_gpu.json`: active fact rows from fresh cluster results.
- `data/valid_facts_results_recluster_gpu_meta.json`: fact-build metadata.
- `data/valid_dfi_splits_recluster_gpu/`: active DFI splits and metadata.
- `data/valid_svm_metrics_recluster_gpu.json`: active SVM metrics.
- `data/ablation/`: ablation outputs + saved model artifacts.
- `data/file-description.txt`: human inventory of active data folder.

### 3.4 `data/optional/` (archived/legacy)

This folder stores older or comparison artifacts moved from the root `data/` (kept, not deleted).

Important contents:
- `data/optional/extra/`: historical valid-subset artifacts (all3-only and centerpair-posthoc variants), helper scripts.
- `data/optional/ablation/nuclearity_ablation.json`: legacy nuclearity ablation results.
- `data/optional/svm_sweep_results.json`: legacy SVM hyperparameter sweep results.
- `data/optional/valid_dfi_splits/`, `data/optional/valid_dfi_splits_centerpair_posthoc/`: historical DFI split dirs.
- `data/optional/triplet_splits/`, `data/optional/dfi_splits/`: older split artifacts.
- `data/optional/file-description.txt`: detailed archive manifest.

### 3.5 `docs/`

- `docs/dataset_section.tex`: primary methodology/results section under active editing.

### 3.6 `logs/`

- Run logs grouped by script category.
- Relevant ablation logs currently include:
  - `logs/ablation/20260406_013008.log` (size3 coverage-controlled ablation run)
  - `logs/ablation/20260406_120452.log` (unrestricted structural ablation run)

## 4) Canonical active workflow (recommended sequence)

1. Fresh clustering on valid triplets:
   - `python run_fact_clustering.py --triplets data/valid_triplets.json --out data/valid_cluster_results_recluster_gpu.json --meta data/valid_cluster_results_recluster_gpu_meta.json`
2. Build facts:
   - `python data/build_facts_from_clusters.py --clusters data/valid_cluster_results_recluster_gpu.json --out data/valid_facts_results_recluster_gpu.json --meta data/valid_facts_results_recluster_gpu_meta.json`
3. Build DFI splits from predefined leakage-safe triplet splits:
   - `python build_dfi_from_splits.py --facts data/valid_facts_results_recluster_gpu.json --split-dir data/valid_triplet_splits --out-dir data/valid_dfi_splits_recluster_gpu`
4. Train/evaluate SVM:
   - `python train_svm_from_dfi_splits.py --split-dir data/valid_dfi_splits_recluster_gpu --out data/valid_svm_metrics_recluster_gpu.json`
5. Run unrestricted structural ablation:
   - `python ablation.py --facts data/valid_facts_results_recluster_gpu.json --split-dir data/valid_dfi_splits_recluster_gpu --out data/ablation/structural_ablation_recluster_gpu.json --model-dir data/ablation/models_recluster_gpu`
6. Run coverage-controlled size3 ablation:
   - `python ablation_size3_clusters.py --facts data/valid_facts_results_recluster_gpu.json --split-dir data/valid_dfi_splits_recluster_gpu --out data/ablation/structural_ablation_size3_recluster_gpu.json --model-dir data/ablation/models_size3_recluster_gpu`

## 5) DFI and label construction details (critical)

Given a triplet with $K$ retained clusters:
- For each cluster, side scores are computed for left/center/right.
- Missing side in a cluster is set to 0 (omission case).
- Left and right deltas are built against center:
  - $\Delta_{left}=W_{left}-W_{center}$
  - $\Delta_{right}=W_{right}-W_{center}$
- This yields:
  - `dfi_left` length $K$
  - `dfi_right` length $K$
- Each triplet always yields two training examples:
  - left-vs-center label 0
  - right-vs-center label 1
- Therefore class balance is exactly 50/50 in each split.

Padding/truncation:
- SVM uses fixed input width equal to max train vector length.
- Shorter vectors are right-padded with zeros.
- Longer eval vectors are truncated.

## 6) Experiments run so far and where results are stored

### 6.1 Active fresh recluster baseline SVM

Artifact:
- `data/valid_svm_metrics_recluster_gpu.json`

Setup:
- Train/val/test DFI rows: 1383 / 171 / 176
- Binary examples: 2766 / 342 / 352
- SVM: RBF, C=10, gamma=0.1, degree=3
- Feature mode: raw padded DFI, input_dim=69

Results:
- Validation: accuracy 0.6988, macro-F1 0.6987
- Test: accuracy 0.6705, macro-F1 0.6702

### 6.2 Active unrestricted structural ablation (all retained clusters)

Artifact:
- `data/ablation/structural_ablation_recluster_gpu.json`
- Models: `data/ablation/models_recluster_gpu/*.pkl`

Test results:
- baseline: acc 0.6705, macro-F1 0.6702
- without_s: acc 0.7017, macro-F1 0.7008
- without_d: acc 0.6591, macro-F1 0.6591
- without_both: acc 0.7528, macro-F1 0.7504

Interpretation:
- In unrestricted clusters, removing both structure terms gave the highest score.
- This indicates omission/coverage asymmetry is a very strong signal in this setting.

### 6.3 Active coverage-controlled size3 ablation (exact size-3 all-bias clusters)

Artifact:
- `data/ablation/structural_ablation_size3_recluster_gpu.json`
- Models: `data/ablation/models_size3_recluster_gpu/*.pkl`

Filtered data size:
- Triplets after filter: train 489, val 84, test 72
- Binary samples: train 978, val 168, test 144
- Input dim: 13

Test results:
- baseline: acc 0.5347, macro-F1 0.5309
- without_s: acc 0.5625, macro-F1 0.5615
- without_d: acc 0.5139, macro-F1 0.4852
- without_both: acc 0.5000, macro-F1 0.3333

Interpretation:
- When omission/coverage cue is constrained away, `without_both` collapses to chance.
- Depth appears useful (removing depth hurts).
- Satellite contribution seems weaker/noisier than depth in this strict setting.
- Lower absolute scores are expected due to large data reduction from strict filter.

### 6.4 Legacy historical experiments (archived in `data/optional/`)

All3-only valid-subset pipeline:
- Metrics file: `data/optional/extra/valid_svm_metrics.json`
- Test: accuracy 0.5217, macro-F1 0.5052

Center-pair-or-all3 posthoc pipeline:
- Metrics file: `data/optional/extra/valid_svm_metrics_centerpair_posthoc.json`
- Test: accuracy 0.6370, macro-F1 0.6349

Legacy nuclearity ablation on older full-data split:
- File: `data/optional/ablation/nuclearity_ablation.json`
- Baseline test macro-F1: 0.6466
- Depth-only test macro-F1: 0.6800
- Delta test macro-F1: +0.0334

Legacy SVM grid search on older full-data split:
- File: `data/optional/svm_sweep_results.json`
- Best overall val accuracy: 0.6569
- Best params: RBF, C=100, gamma=0.01

## 7) End-to-end data lineage and counts (active path)

Valid triplet filtering (`data/valid_triplets_meta.json`):
- Input capped triplets: 2631
- Excluded due to missing-RST article list: 883
- Valid triplets: 1748
- Unique articles in valid triplets: 2617

Leakage-safe split (`data/valid_triplet_splits/meta.json`):
- Exact split achieved: 1398 / 174 / 176
- No doc overlap across splits.

Fresh clustering (`data/valid_cluster_results_recluster_gpu_meta.json`):
- Input triplets: 1748
- Successful with clusters: 1730
- Empty/dropped: 18
- Errors: 0
- Clusters: 16170
- Clustered EDUs: 47668
- EDU filter drop ratio: 0.1293

Facts build (`data/valid_facts_results_recluster_gpu_meta.json`):
- Input rows: 1730
- Output rows: 1730
- Errors: 0

DFI split build (`data/valid_dfi_splits_recluster_gpu/meta.json`):
- Facts rows: 1730
- Matched: 1730
- Unmatched: 0
- DFI rows: 1383 / 171 / 176

## 8) Important caveats and gotchas for the next agent

1. `params.yaml` still contains many legacy default paths (`bias_triplets`, `cluster_results`, `facts_results`, `data/dfi_splits/*`).
   - Do not assume params paths always point to active fresh artifacts.
   - Newer scripts (`ablation.py`, `ablation_size3_clusters.py`) have fresh defaults hardcoded.

2. `run_fact_clustering.py` default output names are timestamped `recluster_centerpair_*`.
   - Always pass explicit `--out` and `--meta` when reproducing the active fresh run naming.

3. `parse_rst.py` uses `paths.dirs.remaining_raw_jsons` from `params.yaml`.
   - In current workspace organization, many legacy files moved into `data/optional/`.
   - Verify input dirs before running parse jobs.

4. `README.md` is partially stale relative to current active scripts and artifact names.
   - Prefer `GPU_FRESH_CLUSTERING_TRAINING_INSTRUCTIONS.txt` and this context file.

5. `data/optional/` is intentionally a non-destructive archive.
   - Do not delete unless explicitly asked.

6. `bert_baseline.py` currently reads `article["text"]` from article JSON; raw files usually use `content`.
   - This likely needs validation/fix before use.

## 9) Where documentation was updated in this session

- `docs/dataset_section.tex` now includes:
  - DFI construction explanation.
  - Explicit 50/50 class-balance table.
  - Coverage-controlled size3 ablation methodology + result table + interpretation.

## 10) Suggested immediate resume checklist for another agent

1. Read this file first.
2. Read `docs/dataset_section.tex` to sync method/results narrative with code.
3. Confirm active artifact presence under `data/` and `data/ablation/`.
4. If re-running experiments, use explicit CLI paths, not only `params.yaml` defaults.
5. If extending experiments, keep all new outputs under `data/ablation/` with clear naming + log files.

## 11) Quick references (active outputs)

- Baseline SVM metrics: `data/valid_svm_metrics_recluster_gpu.json`
- Unrestricted ablation report: `data/ablation/structural_ablation_recluster_gpu.json`
- Size3 coverage-controlled ablation report: `data/ablation/structural_ablation_size3_recluster_gpu.json`
- Unrestricted models: `data/ablation/models_recluster_gpu/`
- Size3 models: `data/ablation/models_size3_recluster_gpu/`
- Ablation logs: `logs/ablation/`

