from sentence_transformers import SentenceTransformer
import json
import glob
from sklearn.neighbors import NearestNeighbors

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = []
files = glob.glob('raw/jsons/*.json')
biases = []


for file in files:
    with open(file) as f:
        data = json.load(f)

    texts.append(data["content"][:1000])
    biases.append(data["bias_text"])


embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True
)


k = 10  

nn = NearestNeighbors(
    n_neighbors=k,
    metric="cosine",
    algorithm="brute"
)

nn.fit(embeddings)

distances, indices = nn.kneighbors(embeddings)

threshold = 0.8
triples = []

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

with open("bias_triplets.json", "w") as f:
    json.dump(triples, f, indent=2)