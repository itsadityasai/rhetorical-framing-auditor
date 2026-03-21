import json
import os
import random
import time

from modules.DFIGenerator import DFIGenerator


DATA_DIR = "data"
FACTS_PATH = os.path.join(DATA_DIR, "facts_results.json")
OUT_DIR = os.path.join(DATA_DIR, "dfi_splits")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 782


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

	dfi = DFIGenerator(clusters=clusters, edu_lookup=edu_lookup)
	cluster_ps = dfi.get_ps()
	dfi_left, dfi_right = dfi.get_DFIs(cluster_ps)

	return {
		"triplet_idx": fact_result.get("triplet_idx"),
		"triplet": fact_result.get("triplet"),
		"dfi_left": dfi_left,
		"dfi_right": dfi_right,
		"num_clusters": len(cluster_ps),
	}


def split_triplets(rows, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=RANDOM_SEED):
	if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
		raise ValueError("Split ratios must sum to 1.0")

	n = len(rows)
	idx = list(range(n))
	rng = random.Random(seed)
	rng.shuffle(idx)

	n_train = int(n * train_ratio)
	n_val = int(n * val_ratio)
	n_test = n - n_train - n_val

	train_idx = set(idx[:n_train])
	val_idx = set(idx[n_train:n_train + n_val])
	test_idx = set(idx[n_train + n_val:n_train + n_val + n_test])

	train_rows = [rows[i] for i in range(n) if i in train_idx]
	val_rows = [rows[i] for i in range(n) if i in val_idx]
	test_rows = [rows[i] for i in range(n) if i in test_idx]

	return train_rows, val_rows, test_rows


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

	train_rows, val_rows, test_rows = split_triplets(dfi_rows)

	save_json(os.path.join(OUT_DIR, "train.json"), train_rows)
	save_json(os.path.join(OUT_DIR, "val.json"), val_rows)
	save_json(os.path.join(OUT_DIR, "test.json"), test_rows)

	save_json(
		os.path.join(OUT_DIR, "meta.json"),
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
		},
	)

	total_time = time.time() - start_time
	print(f"\nCompleted in {total_time:.1f}s")
	print(f"Built rows: {len(dfi_rows)} (errors: {errors})")
	print(f"Train/Val/Test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
	print(f"Saved split files to {OUT_DIR}")
