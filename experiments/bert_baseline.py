import json
import os
import yaml

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from modules.run_logger import init_run_logging, log_run_results, close_run_logging

# =========================
# 1. CONFIG
# =========================

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

bert_params = params["bert_baseline"]
trainer_params = bert_params["training_args"]
path_params = params["paths"]["files"]
path_dirs = params["paths"]["dirs"]

TRAIN_SPLIT = path_params["dfi_train"]
VAL_SPLIT = path_params["dfi_val"]
TEST_SPLIT = path_params["dfi_test"]

MODEL_NAME = bert_params["model_name"]
MAX_LENGTH = bert_params["max_length"]
SEED = bert_params["seed"]

BATCH_SIZE = bert_params["batch_size"]
LR = bert_params["learning_rate"]
EPOCHS = bert_params["epochs"]

OUT_DIR = path_dirs["bert_baseline"]
METRICS_PATH = path_params["bert_metrics"]

LABEL_MAP = bert_params["label_map"]
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

RUN_LOG = init_run_logging(
	script_subdir="bert",
	hyperparams={
		"model_name": MODEL_NAME,
		"max_length": MAX_LENGTH,
		"seed": SEED,
		"batch_size": BATCH_SIZE,
		"learning_rate": LR,
		"epochs": EPOCHS,
		"trainer_args": trainer_params,
		"train_split": TRAIN_SPLIT,
		"val_split": VAL_SPLIT,
		"test_split": TEST_SPLIT,
		"out_dir": OUT_DIR,
	},
)



# =========================
# 2. HELPERS
# =========================

def load_json(path):
	with open(path, "r") as f:
		return json.load(f)


def get_doc_label_map(split_rows):
	"""
	Build unique doc -> label map from split triplets.
	Label comes from slot (left/center/right), not from any random split.
	"""
	doc_to_label = {}
	for row in split_rows:
		triplet = row.get("triplet", {})
		for slot in ["left", "center", "right"]:
			rel_path = triplet.get(slot)
			if not rel_path:
				continue
			label = LABEL_MAP[slot]
			if rel_path in doc_to_label and doc_to_label[rel_path] != label:
				raise ValueError(f"Inconsistent label for {rel_path}")
			doc_to_label[rel_path] = label
	return doc_to_label


def load_text_from_rel_path(rel_path):
    article = load_json(rel_path)
    return article["text"]


def build_texts_labels(split_rows):
	doc_to_label = get_doc_label_map(split_rows)
	texts, labels = [], []
	for rel_path, label in doc_to_label.items():
		texts.append(load_text_from_rel_path(rel_path))
		labels.append(label)
	return texts, labels, set(doc_to_label.keys())


def tokenize(tokenizer, texts):
	return tokenizer(
		texts,
		truncation=True,
		padding=True,
		max_length=MAX_LENGTH,
	)


class NewsDataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):
		item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
		item["labels"] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)


def compute_metrics(eval_pred):
	logits, labels = eval_pred
	preds = np.argmax(logits, axis=1)
	return {
		"accuracy": float(accuracy_score(labels, preds)),
		"macro_f1": float(f1_score(labels, preds, average="macro")),
	}


# =========================
# 3. LOAD LEAKAGE-SAFE SPLITS
# =========================

set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

train_rows = load_json(TRAIN_SPLIT)
val_rows = load_json(VAL_SPLIT)
test_rows = load_json(TEST_SPLIT)

X_train, y_train, train_docs = build_texts_labels(train_rows)
X_val, y_val, val_docs = build_texts_labels(val_rows)
X_test, y_test, test_docs = build_texts_labels(test_rows)

print(f"Unique docs -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
print(
	"Doc overlap check -> "
	f"train/val={len(train_docs & val_docs)}, "
	f"train/test={len(train_docs & test_docs)}, "
	f"val/test={len(val_docs & test_docs)}"
)


# =========================
# 4. TOKENIZE + DATASETS
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_enc = tokenize(tokenizer, X_train)
val_enc = tokenize(tokenizer, X_val)
test_enc = tokenize(tokenizer, X_test)

train_dataset = NewsDataset(train_enc, y_train)
val_dataset = NewsDataset(val_enc, y_val)
test_dataset = NewsDataset(test_enc, y_test)


# =========================
# 5. MODEL + TRAINER
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
	MODEL_NAME,
	num_labels=len(LABEL_MAP),
	id2label=ID2LABEL,
	label2id=LABEL_MAP,
)

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,

	eval_strategy=trainer_params["eval_strategy"],
	save_strategy=trainer_params["save_strategy"],

	logging_steps=trainer_params["logging_steps"],
	load_best_model_at_end=trainer_params["load_best_model_at_end"],
	metric_for_best_model=trainer_params["metric_for_best_model"],

	report_to=trainer_params["report_to"],
	fp16=trainer_params["fp16"],
)

trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=val_dataset,
	compute_metrics=compute_metrics,
)


# =========================
# 6. TRAIN + EVAL
# =========================

print("Training BERT baseline...")
trainer.train()

val_preds = trainer.predict(val_dataset)
test_preds = trainer.predict(test_dataset)

y_val_pred = np.argmax(val_preds.predictions, axis=1)
y_test_pred = np.argmax(test_preds.predictions, axis=1)

val_acc = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average="macro")
test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average="macro")

print("\n=== VAL RESULTS ===")
print(f"Accuracy: {val_acc:.4f}")
print(f"Macro-F1: {val_f1:.4f}")
print(classification_report(y_val, y_val_pred, target_names=["left", "center", "right"], digits=4, zero_division=0))

print("\n=== TEST RESULTS ===")
print(f"Accuracy: {test_acc:.4f}")
print(f"Macro-F1: {test_f1:.4f}")
print(classification_report(y_test, y_test_pred, target_names=["left", "center", "right"], digits=4, zero_division=0))


# =========================
# 7. SAVE METRICS
# =========================

out = {
	"model": MODEL_NAME,
	"seed": SEED,
	"max_length": MAX_LENGTH,
	"epochs": EPOCHS,
	"learning_rate": LR,
	"batch_size": BATCH_SIZE,
	"doc_counts": {
		"train": len(X_train),
		"val": len(X_val),
		"test": len(X_test),
	},
	"val": {
		"accuracy": float(val_acc),
		"macro_f1": float(val_f1),
	},
	"test": {
		"accuracy": float(test_acc),
		"macro_f1": float(test_f1),
	},
}

with open(METRICS_PATH, "w") as f:
	json.dump(out, f, indent=2)

print(f"Saved metrics to {METRICS_PATH}")

log_run_results(
	RUN_LOG,
	{
		"metrics_path": METRICS_PATH,
		"val": out["val"],
		"test": out["test"],
		"doc_counts": out["doc_counts"],
	},
)
close_run_logging(RUN_LOG, status="success")