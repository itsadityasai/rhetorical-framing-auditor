import argparse
import json
import os
import time

from modules.DFIGenerator import DFIGenerator


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def triplet_key(triplet):
    return (triplet.get("left"), triplet.get("center"), triplet.get("right"))


def build_triplet_dfi(fact_row):
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    dfi = DFIGenerator(clusters=clusters, edu_lookup=edu_lookup)
    cluster_ps = dfi.get_ps()
    dfi_left, dfi_right = dfi.get_DFIs(cluster_ps)

    return {
        "triplet_idx": fact_row.get("triplet_idx"),
        "triplet": fact_row.get("triplet"),
        "dfi_left": dfi_left,
        "dfi_right": dfi_right,
        "num_clusters": len(cluster_ps),
    }


def main():
    parser = argparse.ArgumentParser(description="Build DFI rows from facts using predefined triplet splits")
    parser.add_argument("--facts", default="data/valid_facts_results.json")
    parser.add_argument("--split-dir", default="data/valid_triplet_splits")
    parser.add_argument("--out-dir", default="data/valid_dfi_splits")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    facts = load_json(args.facts)

    train_split = load_json(os.path.join(args.split_dir, "train.json"))
    val_split = load_json(os.path.join(args.split_dir, "val.json"))
    test_split = load_json(os.path.join(args.split_dir, "test.json"))

    train_keys = {triplet_key(r) for r in train_split}
    val_keys = {triplet_key(r) for r in val_split}
    test_keys = {triplet_key(r) for r in test_split}

    train_rows = []
    val_rows = []
    test_rows = []

    matched = 0
    unmatched = 0
    errors = 0

    start = time.time()

    for row in facts:
        try:
            key = triplet_key(row.get("triplet", {}))
            dfi_row = build_triplet_dfi(row)

            if key in train_keys:
                train_rows.append(dfi_row)
                matched += 1
            elif key in val_keys:
                val_rows.append(dfi_row)
                matched += 1
            elif key in test_keys:
                test_rows.append(dfi_row)
                matched += 1
            else:
                unmatched += 1
        except Exception as e:
            errors += 1
            print(f"[ERROR] DFI build failure for triplet_idx={row.get('triplet_idx')}: {e}")

    save_json(os.path.join(args.out_dir, "train.json"), train_rows)
    save_json(os.path.join(args.out_dir, "val.json"), val_rows)
    save_json(os.path.join(args.out_dir, "test.json"), test_rows)

    meta = {
        "input_facts": args.facts,
        "split_dir": args.split_dir,
        "out_dir": args.out_dir,
        "facts_rows": len(facts),
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "errors": errors,
        "dfi_rows": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "elapsed_seconds": round(time.time() - start, 3),
    }
    save_json(os.path.join(args.out_dir, "meta.json"), meta)

    print("DFI BUILD SUMMARY")
    print("=" * 64)
    print(f"Facts rows: {len(facts)}")
    print(f"Matched to split rows: {matched}")
    print(f"Unmatched: {unmatched}")
    print(f"Errors: {errors}")
    print(f"DFI rows train/val/test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"Saved: {os.path.join(args.out_dir, 'meta.json')}")


if __name__ == "__main__":
    main()
