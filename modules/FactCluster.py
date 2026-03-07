from sentence_transformers import SentenceTransformer
SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

from sentence_transformers import CrossEncoder
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

def idify_edus(article): # article: {article_id, bias:left, center, right, edus: [{id, text}]}
    article_id = article["article_id"]
    edus = article["edus"]
    for edu in edus:
        edu["id"] = f"{article_id}_{edu['id']}"
        edu["bias"] = article["bias"]
    return edus

def encode_edus(edus): # each edu is a {id, text} object

    ids = [edu["id"] for edu in edus]
    embeddings = sbert_model.encode([edu["text"] for edu in edus], batch_size=64, convert_to_numpy=True, normalize_embeddings=True) 
    # REPORT: explain encode() params
    return dict(zip(ids, embeddings))


def cluster(encoded_edus):
    from sklearn.cluster import AgglomerativeClustering
    from collections import defaultdict
    import numpy as np

    clustering = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=0.35,
        n_clusters=None
    )
    # REPORT: explain clustering params
    # TODO: experiment with clustering params
    
    ids = list(encoded_edus.keys())
    embeddings = np.array(list(encoded_edus.values()))

    labels = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)

    for edu_id, label in zip(ids, labels):
        clusters[int(label)].append(edu_id)

    return clusters

def validate_cluster(cluster_ids, edu_lookup, threshold=0.7):
    # TODO: try different thresholds
    # REPORT: explain threshold choice and its impact on precision/recall of fact clustering

    if len(cluster_ids) < 2:
        return []

    pairs = []
    pair_map = []

    for i in range(len(cluster_ids)):
        for j in range(i+1, len(cluster_ids)):
            id1 = cluster_ids[i]
            id2 = cluster_ids[j]

            text1 = edu_lookup[id1]["text"]
            text2 = edu_lookup[id2]["text"]

            pairs.append((text1, text2))
            pair_map.append((id1, id2))

    scores = cross_encoder.predict(pairs, batch_size=64)

    valid_pairs = []
    for (id1, id2), score in zip(pair_map, scores):
        if score > threshold:
            valid_pairs.append((id1, id2))

    return valid_pairs


class FactCluster:
    def __init__(self, articles):
        self.articles = articles
        self.edu_lookup = {}
        self.encoded_edus = {}
        self.clusters = {}

        for article in articles:
            edus = idify_edus(article)
            self.edu_lookup.update({edu["id"]: edu for edu in edus})
            self.encoded_edus.update(encode_edus(edus))
        self.clusters = cluster(self.encoded_edus)
        
    def refine_clusters(self, refine_threshold=0.7):
        refined = {}

        for cluster_id, edu_ids in self.clusters.items():

            valid_pairs = validate_cluster(
                edu_ids,
                self.edu_lookup,
                threshold=refine_threshold
            )
            
            valid_edus = set()
            for id1, id2 in valid_pairs:
                valid_edus.add(id1)
                valid_edus.add(id2)
            
            if len(valid_edus) < 2:
                continue
            if all(self.edu_lookup[eid]["bias"] != "center" for eid in valid_edus):
                continue # dont include cluster if there is no center edu as we won't be able to calculate dfi later
            


            
            refined[cluster_id] = list(valid_edus)

        self.clusters = refined
        # must be called separately
        
    # TODO: build_facts() => include nuclearity etc
    
    def get_clusters(self):
        return self.clusters
   