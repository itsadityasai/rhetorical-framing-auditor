from sentence_transformers import SentenceTransformer
import json
import glob
import os
import sys
from sklearn.neighbors import NearestNeighbors
import yaml

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from modules.run_logger import init_run_logging, log_run_results, close_run_logging

with open("../params.yaml", "r") as f:
    params = yaml.safe_load(f)

sort_params = params["sort_data"]
sbert_params = params["models"]["sbert"]

model = SentenceTransformer(sbert_params["model_name"])

texts = []
files = glob.glob(sort_params["input_glob"])
biases = []


for file in files:
    with open(file) as f:
        data = json.load(f)

    texts.append(data["content"][: sort_params["text_char_limit"]])
    # REPORT: explain why :1000
    biases.append(data["bias_text"])


embeddings = model.encode(
    texts,
    batch_size=sort_params["encode_batch_size"],
    show_progress_bar=True
)


nn_params = sort_params["nearest_neighbors"]
k = nn_params["k"]

nn = NearestNeighbors(
    n_neighbors=k,
    metric=nn_params["metric"],
    algorithm=nn_params["algorithm"]
)

# REPORT: explain k value and nearest neighbors params

nn.fit(embeddings)

distances, indices = nn.kneighbors(embeddings)

threshold = sort_params["similarity_threshold"] # REPORT: mention chosen threshold
triples = []

RUN_LOG = init_run_logging(
    script_subdir="sort_data",
    hyperparams={
        "input_glob": sort_params["input_glob"],
        "text_char_limit": sort_params["text_char_limit"],
        "encode_batch_size": sort_params["encode_batch_size"],
        "nearest_neighbors": nn_params,
        "similarity_threshold": threshold,
        "output_path": sort_params["output_path"],
        "sbert_model_name": sbert_params["model_name"],
    },
)


for i in range(len(texts)):

    neighbors = []

    for j, d in zip(indices[i], distances[i]):

        if i == j:
            continue

        similarity = 1 - d

        if similarity > threshold:
            neighbors.append(j)

    
    for a in range(len(neighbors)):
        for b in range(a + 1, len(neighbors)):

            j = neighbors[a]
            k2 = neighbors[b]

            group_bias = {
                biases[i],
                biases[j],
                biases[k2]
            }

            if group_bias == {"left", "center", "right"}:
                triples.append({
                    "left": None,
                    "center": None,
                    "right": None
                })

                for idx in [i, j, k2]:
                    triples[-1][biases[idx]] = files[idx]

print("Found", len(triples), "triples")

with open(sort_params["output_path"], "w") as f:
    json.dump(triples, f, indent=2)

log_run_results(
    RUN_LOG,
    {
        "num_documents": len(texts),
        "num_triples": len(triples),
        "output_path": sort_params["output_path"],
    },
)
close_run_logging(RUN_LOG, status="success")
    
    
# REPORT: while most of the ones we checked are accurate there may be a very small percentage (likely ~1%) that are not actually true triplets.

