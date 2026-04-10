import json
import os
import random
import time
from collections import defaultdict, deque
import yaml

from modules.DFIGenerator import DFIGenerator
from modules.run_logger import init_run_logging, log_run_results, close_run_logging

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

FACTS_PATH = params["paths"]["files"]["facts"]
OUT_DIR = params["paths"]["dirs"]["dfi_splits"]

TRAIN_RATIO = params["dfi"]["splits"]["train_ratio"]
VAL_RATIO = params["dfi"]["splits"]["val_ratio"]
TEST_RATIO = params["dfi"]["splits"]["test_ratio"]
RANDOM_SEED = params["dfi"]["splits"]["random_seed"]
RATIO_TOLERANCE = params["dfi"]["splits"]["ratio_tolerance"]

DFI_ALPHA = params["dfi"]["alpha"]
DFI_GAMMA = params["dfi"]["gamma"]

RUN_LOG = init_run_logging(
	script_subdir="build_dfi",
	hyperparams={
		"facts_path": FACTS_PATH,
		"out_dir": OUT_DIR,
		"dfi": {"alpha": DFI_ALPHA, "gamma": DFI_GAMMA},
		"splits": {
			"train_ratio": TRAIN_RATIO,
			"val_ratio": VAL_RATIO,
			"test_ratio": TEST_RATIO,
			"random_seed": RANDOM_SEED,
			"ratio_tolerance": RATIO_TOLERANCE,
		},
	},
)



def load_facts(path):
	with open(path, "r") as f:
		return json.load(f)


def save_json(path, payload):
	with open(path, "w") as f:
		json.dump(payload, f, indent=2)


def print_progress(current, total, elapsed, errors):
	if current > 0:
		eta_seconds = (elapsed / current) * (total - current)
		eta_min, eta_sec = divmod(int(eta_seconds), 60)
		eta_str = f"ETA: {eta_min}m {eta_sec}s"
	else:
		eta_str = "ETA: --"

	print(f"\r{current}/{total} - {eta_str} - {errors} errors   ", end="", flush=True)


def build_triplet_dfi(fact_result):
	clusters = fact_result.get("clusters", {})
	edu_lookup = fact_result.get("edu_lookup", {})

	dfi = DFIGenerator(alpha=DFI_ALPHA, gamma=DFI_GAMMA, clusters=clusters, edu_lookup=edu_lookup)
	cluster_ps = dfi.get_ps()
	dfi_left, dfi_right = dfi.get_DFIs(cluster_ps)

	return {
		"triplet_idx": fact_result.get("triplet_idx"),
		"triplet": fact_result.get("triplet"),
		"dfi_left": dfi_left,
		"dfi_right": dfi_right,
		"num_clusters": len(cluster_ps),
	}


def extract_triplet_doc_ids(row):
	triplet = row.get("triplet", {})
	doc_ids = set()
	for key in ["left", "center", "right"]:
		path = triplet.get(key)
		if path:
			doc_ids.add(os.path.basename(path).replace(".json", ""))
	return doc_ids


def build_connected_components(rows):
	# Two triplets are connected if they share any document ID.
	triplet_docs = [extract_triplet_doc_ids(r) for r in rows]
	doc_to_triplets = defaultdict(list)

	for i, docs in enumerate(triplet_docs):
		for doc_id in docs:
			doc_to_triplets[doc_id].append(i)

	unseen = set(range(len(rows)))
	components = []

	while unseen:
		start = unseen.pop()
		queue = deque([start])
		component = [start]

		while queue:
			cur = queue.popleft()
			for doc_id in triplet_docs[cur]:
				for nbr in doc_to_triplets[doc_id]:
					if nbr in unseen:
						unseen.remove(nbr)
						queue.append(nbr)
						component.append(nbr)

		components.append(component)

	return components, triplet_docs


def split_triplets(rows, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=RANDOM_SEED):
	if abs((train_ratio + val_ratio + test_ratio) - 1.0) > RATIO_TOLERANCE:
		raise ValueError("Split ratios must sum to 1.0")

	n = len(rows)
	targets = {
		"train": int(n * train_ratio),
		"val": int(n * val_ratio),
	}
	targets["test"] = n - targets["train"] - targets["val"]

	components, triplet_docs = build_connected_components(rows)
	rng = random.Random(seed)
	# Shuffle first for tie-breaking, then assign larger connected components first.
	rng.shuffle(components)
	components.sort(key=len, reverse=True)
	component_sizes = sorted((len(c) for c in components), reverse=True)

	assigned = {
		"train": [],
		"val": [],
		"test": [],
	}
	counts = {"train": 0, "val": 0, "test": 0}
	priority = ["train", "val", "test"]

	for comp in components:
		comp_size = len(comp)

		# First try to fill the split with the largest remaining deficit (train-priority tie-break).
		deficit_splits = [
			s for s in priority
			if targets[s] - counts[s] > 0
		]

		if deficit_splits:
			best_split = max(
				deficit_splits,
				key=lambda s: (targets[s] - counts[s], -priority.index(s))
			)
		else:
			# If all targets are already met/over, place component where overshoot is smallest.
			best_split = min(
				priority,
				key=lambda s: (
					(counts[s] + comp_size - targets[s]) ** 2,
					priority.index(s),
				)
			)

		assigned[best_split].extend(comp)
		counts[best_split] += comp_size

	train_rows = [rows[i] for i in assigned["train"]]
	val_rows = [rows[i] for i in assigned["val"]]
	test_rows = [rows[i] for i in assigned["test"]]

	# Safety check: no document overlap across splits.
	def docs_for(indices):
		docs = set()
		for i in indices:
			docs.update(triplet_docs[i])
		return docs

	train_docs = docs_for(assigned["train"])
	val_docs = docs_for(assigned["val"])
	test_docs = docs_for(assigned["test"])

	if (train_docs & val_docs) or (train_docs & test_docs) or (val_docs & test_docs):
		raise RuntimeError("Split leakage detected: some document IDs appear in multiple sets")

	split_info = {
		"targets": targets,
		"achieved": counts,
		"components": {
			"count": len(components),
			"largest": component_sizes[0] if component_sizes else 0,
			"smallest": component_sizes[-1] if component_sizes else 0,
			"singletons": sum(1 for s in component_sizes if s == 1),
		},
	}

	return train_rows, val_rows, test_rows, split_info


if __name__ == "__main__":
	os.makedirs(OUT_DIR, exist_ok=True)

	facts = load_facts(FACTS_PATH)
	total = len(facts)
	print(f"Building DFI rows for {total} triplets\n")

	dfi_rows = []
	errors = 0
	start_time = time.time()

	for i, fact_result in enumerate(facts):
		elapsed = time.time() - start_time
		print_progress(i + 1, total, elapsed, errors)

		try:
			dfi_rows.append(build_triplet_dfi(fact_result))
		except Exception as e:
			errors += 1
			print(f"\n[ERROR] Triplet {fact_result.get('triplet_idx', i)}: {e}")

	print()

	train_rows, val_rows, test_rows, split_info = split_triplets(dfi_rows)

	def collect_docs(split_rows):
		docs = set()
		for row in split_rows:
			docs.update(extract_triplet_doc_ids(row))
		return docs

	train_docs = collect_docs(train_rows)
	val_docs = collect_docs(val_rows)
	test_docs = collect_docs(test_rows)

	print("Leakage-safe component stats:")
	print(
		f"components={split_info['components']['count']} | "
		f"largest={split_info['components']['largest']} | "
		f"singletons={split_info['components']['singletons']}"
	)
	print(
		"Target vs achieved triplets: "
		f"train {split_info['targets']['train']}->{split_info['achieved']['train']}, "
		f"val {split_info['targets']['val']}->{split_info['achieved']['val']}, "
		f"test {split_info['targets']['test']}->{split_info['achieved']['test']}"
	)

	save_json(params["paths"]["files"]["dfi_train"], train_rows)
	save_json(params["paths"]["files"]["dfi_val"], val_rows)
	save_json(params["paths"]["files"]["dfi_test"], test_rows)

	save_json(
		params["paths"]["files"]["dfi_meta"],
		{
			"total_input_triplets": total,
			"total_built_rows": len(dfi_rows),
			"errors": errors,
			"seed": RANDOM_SEED,
			"ratios": {
				"train": TRAIN_RATIO,
				"val": VAL_RATIO,
				"test": TEST_RATIO,
			},
			"split_sizes": {
				"train": len(train_rows),
				"val": len(val_rows),
				"test": len(test_rows),
			},
			"split_info": split_info,
			"doc_sizes": {
				"train": len(train_docs),
				"val": len(val_docs),
				"test": len(test_docs),
			},
			"doc_overlap": {
				"train_val": len(train_docs & val_docs),
				"train_test": len(train_docs & test_docs),
				"val_test": len(val_docs & test_docs),
			},
		},
	)

	total_time = time.time() - start_time
	print(f"\nCompleted in {total_time:.1f}s")
	print(f"Built rows: {len(dfi_rows)} (errors: {errors})")
	print(f"Train/Val/Test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
	print(f"Saved split files to {OUT_DIR}")

	log_run_results(
		RUN_LOG,
		{
			"total_input_triplets": total,
			"total_built_rows": len(dfi_rows),
			"errors": errors,
			"split_sizes": {
				"train": len(train_rows),
				"val": len(val_rows),
				"test": len(test_rows),
			},
			"doc_sizes": {
				"train": len(train_docs),
				"val": len(val_docs),
				"test": len(test_docs),
			},
			"doc_overlap": {
				"train_val": len(train_docs & val_docs),
				"train_test": len(train_docs & test_docs),
				"val_test": len(val_docs & test_docs),
			},
			"elapsed_seconds": round(total_time, 3),
		},
	)
	close_run_logging(RUN_LOG, status="success")
