import os
import signal
import time
import torch
import orjson
import yaml
from isanlp_rst.parser import Parser
from modules.run_logger import init_run_logging, log_run_results, close_run_logging

import gc
gc.disable()  # faster loops

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

parse_params = params["parse_rst"]
parser_params = parse_params["parser"]
paths = params["paths"]

RST_TIMEOUT_SECONDS = parse_params["timeout_seconds"]
PROGRESS_EVERY_TRIPLETS = parse_params["progress_every_triplets"]
RAW_JSON_DIR = paths["dirs"]["raw_jsons"]
RST_OUTPUT_DIR = paths["dirs"]["rst_output"]
TRIPLETS_PATH = paths["files"]["triplets"]

RUN_LOG = init_run_logging(
    script_subdir="parse_rst",
    hyperparams={
        "triplets_path": TRIPLETS_PATH,
        "raw_json_dir": RAW_JSON_DIR,
        "rst_output_dir": RST_OUTPUT_DIR,
        "timeout_seconds": RST_TIMEOUT_SECONDS,
        "progress_every_triplets": PROGRESS_EVERY_TRIPLETS,
        "parser": parser_params,
    },
)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError()


import unicodedata
import re

# REPORT: explain text normalization choices and their importance for RST parsing
def normalize_text(text):

    text = unicodedata.normalize("NFKC", text)

    replacements = {
        '\u2588': ' ',   
        '\u00A0': ' ',   
        '\u200B': '',    
        '\u200C': '',    
        '\u200D': '',    
        '\uFEFF': '',    
        '\u00AD': '',    

        '\u201c': '"',
        '\u201d': '"',
        '\u201e': '"',
        '\u201f': '"',

        '\u2019': "'",
        '\u2018': "'",
        '\u2032': "'",

        '\u2014': '-',
        '\u2013': '-',
        '\u2212': '-',   

        '\u2026': '...',

        '\u2022': '-',
        '\u25CF': '-',

        '\n': ' ',
        '\r': ' ',
        '\t': ' '
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


parser = None


def get_rst(text, article_id):
    # REPORT: explain timeout mechanism and why we need it
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(RST_TIMEOUT_SECONDS)
    try:
        with torch.inference_mode():
            res = parser(text)
        signal.alarm(0)
    except TimeoutError:
        return None

    root = res["rst"][0]

    edus = []
    relations = []

    def traverse(node, depth=0):
        if node.left is None and node.right is None:
            edus.append({
                "id": node.id,
                "text": node.text.strip(),
                "depth": depth
            })
            return

        left_child = node.left
        right_child = node.right

        relations.append({
            "parent": node.id,
            "left": left_child.id if left_child else None,
            "right": right_child.id if right_child else None,
            "relation": node.relation,
            "nuclearity": node.nuclearity
        })

        if left_child:
            traverse(left_child, depth + 1)
        if right_child:
            traverse(right_child, depth + 1)

    traverse(root)

    return {
        "article_id": article_id,
        "edus": edus,
        "relations": relations
    }


os.makedirs(RST_OUTPUT_DIR, exist_ok=True)


def process_doc(article_id, done, failed):
    
    if article_id in done or article_id in failed:
        return article_id not in failed
    
    try:
        file_path = os.path.join(RAW_JSON_DIR, f"{article_id}.json")
        with open(file_path, "rb") as f:
            data = orjson.loads(f.read())

        text = normalize_text(data["content"])
        result = get_rst(text, article_id)

        if result is None:
            print(f"[TIMEOUT] {article_id}")
            failed.add(article_id)
            return False

        output_path = os.path.join(RST_OUTPUT_DIR, f"{article_id}.json")
        with open(output_path, "wb") as f:
            f.write(orjson.dumps(result))
        
        done.add(article_id)
        return True

    except Exception as e:
        print(f"[ERROR] {article_id}: {e}")
        failed.add(article_id)
        return False


if __name__ == "__main__":
    print("Starting RST parsing...")

    print("Initializing parser (loading model)...")
    parser = Parser(
        hf_model_name=parser_params["hf_model_name"],
        hf_model_version=parser_params["hf_model_version"],
        cuda_device=parser_params["cuda_device"],
        relinventory=parser_params["relinventory"],
    )
    # REPORT: explain parser params
    print("Parser ready!")

    # Load triplets
    with open(TRIPLETS_PATH, "rb") as f:
        triplets = orjson.loads(f.read())
    print(f"Loaded {len(triplets)} triplets")

    # Already processed files
    done = set(os.path.basename(f).replace(".json", "") for f in os.listdir(RST_OUTPUT_DIR) if f.endswith(".json"))
    print(f"Already processed: {len(done)} documents")

    # Track failed documents (timeout/error) - skip all triplets with these
    failed = set()

    processed = 0
    skipped_triplets = 0
    total_triplets = len(triplets)
    start_time = time.time()

    for i, triplet in enumerate(triplets):
        # Extract article IDs from paths
        left_id = os.path.basename(triplet["left"]).replace(".json", "")
        center_id = os.path.basename(triplet["center"]).replace(".json", "")
        right_id = os.path.basename(triplet["right"]).replace(".json", "")

        # Skip triplet if any doc has already failed
        if left_id in failed or center_id in failed or right_id in failed:
            skipped_triplets += 1
            continue

        # Process each doc in the triplet (skips if already done)
        for article_id in [left_id, center_id, right_id]:
            if article_id not in done and article_id not in failed:
                success = process_doc(article_id, done, failed)
                if success:
                    processed += 1

        # Print progress every 10 triplets
        if (i + 1) % PROGRESS_EVERY_TRIPLETS == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total_triplets - i - 1) / rate
            eta_mins = int(remaining // 60)
            eta_secs = int(remaining % 60)
            print(f"{i + 1}/{total_triplets} triplets done | ETA: {eta_mins}m {eta_secs}s")

    print(f"\nDone! Newly processed: {processed}, Failed docs: {len(failed)}, Skipped triplets: {skipped_triplets}")

    log_run_results(
        RUN_LOG,
        {
            "total_triplets": total_triplets,
            "newly_processed_docs": processed,
            "failed_docs": len(failed),
            "skipped_triplets": skipped_triplets,
            "already_processed_docs": len(done),
            "elapsed_seconds": round(time.time() - start_time, 3),
        },
    )
    close_run_logging(RUN_LOG, status="success")
