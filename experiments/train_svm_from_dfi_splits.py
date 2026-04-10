import argparse
import json

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_xy_raw(rows):
    x = []
    y = []
    for row in rows:
        x.append(list(row["dfi_left"]))
        y.append(0)
        x.append(list(row["dfi_right"]))
        y.append(1)
    return x, np.array(y)


def pad_or_truncate(raw_x, target_len):
    arr = np.zeros((len(raw_x), target_len), dtype=float)
    for i, vec in enumerate(raw_x):
        lim = min(len(vec), target_len)
        if lim > 0:
            arr[i, :lim] = np.array(vec[:lim], dtype=float)
    return arr


def evaluate(model, rows, max_len):
    x_raw, y = build_xy_raw(rows)
    x = pad_or_truncate(x_raw, max_len)
    pred = model.predict(x)
    return {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train SVM on DFI split files")
    parser.add_argument("--split-dir", default="data/valid_dfi_splits")
    parser.add_argument("--out", default="data/valid_svm_metrics.json")
    args = parser.parse_args()

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    svm_params = params["svm"]["model"]

    train_rows = load_json(f"{args.split_dir}/train.json")
    val_rows = load_json(f"{args.split_dir}/val.json")
    test_rows = load_json(f"{args.split_dir}/test.json")

    x_train_raw, y_train = build_xy_raw(train_rows)
    max_len = max((len(v) for v in x_train_raw), default=0)
    x_train = pad_or_truncate(x_train_raw, max_len)

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_params["kernel"],
            C=svm_params["C"],
            gamma=svm_params["gamma"],
            degree=svm_params["degree"],
        ),
    )
    model.fit(x_train, y_train)

    train_metrics = evaluate(model, train_rows, max_len)
    val_metrics = evaluate(model, val_rows, max_len)
    test_metrics = evaluate(model, test_rows, max_len)

    out = {
        "split_dir": args.split_dir,
        "input_triplets": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "input_examples_binary_left_right": {
            "train": int(train_metrics["samples"]),
            "val": int(val_metrics["samples"]),
            "test": int(test_metrics["samples"]),
        },
        "svm_params": svm_params,
        "feature_mode": "raw_padded_dfi",
        "input_dim": int(max_len),
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "testing_protocol_note": "Each triplet contributes two binary examples: left-vs-center (label 0) and right-vs-center (label 1).",
    }
    save_json(args.out, out)

    print("SVM TRAINING SUMMARY")
    print("=" * 64)
    print(f"Triplets train/val/test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"Examples train/val/test: {train_metrics['samples']}/{val_metrics['samples']}/{test_metrics['samples']}")
    print(f"Input dim: {max_len}")
    print(f"Val   acc={val_metrics['accuracy']:.4f}, macro_f1={val_metrics['macro_f1']:.4f}")
    print(f"Test  acc={test_metrics['accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
