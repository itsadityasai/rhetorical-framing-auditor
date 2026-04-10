import json
import os
import re
from sentence_transformers import SentenceTransformer
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SBERT_MODEL_NAME = params["models"]["sbert"]["model_name"]
SBERT_ENCODE_BATCH_SIZE = params["models"]["sbert"]["encode_batch_size"]
SBERT_NORMALIZE = params["models"]["sbert"]["normalize_embeddings"]
sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

from sentence_transformers import CrossEncoder
CROSS_ENCODER_NAME = params["models"]["cross_encoder"]["model_name"]
CROSS_ENCODER_BATCH_SIZE = params["models"]["cross_encoder"]["predict_batch_size"]
cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

AGGLOMERATIVE_PARAMS = params["fact_clustering"]["agglomerative"]
PAIR_VALIDATION_THRESHOLD = params["fact_clustering"]["pair_validation"]["threshold"]
DEFAULT_REFINE_THRESHOLD = params["fact_clustering"]["refine_threshold"]
SINGLETON_KEEP_BIAS = params["fact_clustering"]["singleton_keep_bias"]
MIN_BIASES_PER_CLUSTER = params["fact_clustering"].get("min_biases_per_cluster", 2)

EDU_FILTER_PARAMS = params["fact_clustering"].get("edu_filter", {})
EDU_FILTER_ENABLED = EDU_FILTER_PARAMS.get("enabled", True)
EDU_FILTER_MIN_TOKENS = EDU_FILTER_PARAMS.get("min_tokens", 3)
EDU_FILTER_DROP_PUNCT_ONLY = EDU_FILTER_PARAMS.get("drop_punct_only", True)
EDU_FILTER_DROP_URL_LIKE = EDU_FILTER_PARAMS.get("drop_url_like", True)
EDU_FILTER_DROP_SOCIAL_META = EDU_FILTER_PARAMS.get("drop_social_meta", True)
EDU_FILTER_DROP_SHORT_ATTR = EDU_FILTER_PARAMS.get("drop_short_attribution", True)
EDU_FILTER_SHORT_ATTR_MAX_TOKENS = EDU_FILTER_PARAMS.get("short_attribution_max_tokens", 8)
EDU_FILTER_BOILERPLATE_PHRASES = [
    p.lower() for p in EDU_FILTER_PARAMS.get(
        "boilerplate_phrases",
        [
            "story highlights",
            "just watched",
            "must watch",
            "read more",
            "add interest",
            "more :",
            "follow us",
            "newsletter",
        ],
    )
]
DATA_DIR = params["paths"]["dirs"]["data"]

URL_OR_SOCIAL_PATTERN = re.compile(r"https?://|www\.|\.com\b|\.org\b|pic\.twitter|twitter\.com|<\s*a\s+href", re.IGNORECASE)
PUNCT_ONLY_PATTERN = re.compile(r"[^A-Za-z0-9]+")
SHORT_ATTR_PATTERN = re.compile(r"\b(said|says|told|according to|asked|wrote|added)\b", re.IGNORECASE)


def _token_count(text):
    return len(text.split())


def _is_edu_fact_candidate(text):
    if text is None:
        return False

    text = text.strip()
    if not text:
        return False

    if not EDU_FILTER_ENABLED:
        return True

    token_count = _token_count(text)

    if token_count < EDU_FILTER_MIN_TOKENS:
        return False

    if EDU_FILTER_DROP_PUNCT_ONLY and PUNCT_ONLY_PATTERN.fullmatch(text):
        return False

    lowered = text.lower()

    if EDU_FILTER_DROP_URL_LIKE and URL_OR_SOCIAL_PATTERN.search(lowered):
        return False

    if EDU_FILTER_DROP_SOCIAL_META and any(phrase in lowered for phrase in EDU_FILTER_BOILERPLATE_PHRASES):
        return False

    if EDU_FILTER_DROP_SHORT_ATTR and token_count <= EDU_FILTER_SHORT_ATTR_MAX_TOKENS and SHORT_ATTR_PATTERN.search(lowered):
        return False

    return True


def idify_edus(article): # article: {article_id, bias:left, center, right, edus: [{id, text}]}
    article_id = article["article_id"]
    edus = article["edus"]

    filtered = []
    for edu in edus:
        text = (edu.get("text") or "").strip()
        if not _is_edu_fact_candidate(text):
            continue

        kept = dict(edu)
        kept["id"] = f"{article_id}_{edu['id']}"
        kept["bias"] = article["bias"]
        kept["text"] = text
        filtered.append(kept)

    return filtered

def encode_edus(edus): # each edu is a {id, text} object

    ids = [edu["id"] for edu in edus]
    embeddings = sbert_model.encode(
        [edu["text"] for edu in edus],
        batch_size=SBERT_ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=SBERT_NORMALIZE,
    )
    # REPORT: explain encode() params
    return dict(zip(ids, embeddings))


def cluster(encoded_edus):
    from sklearn.cluster import AgglomerativeClustering
    from collections import defaultdict
    import numpy as np

    if len(encoded_edus) == 0:
        return {}

    if len(encoded_edus) == 1:
        only_id = next(iter(encoded_edus.keys()))
        return {0: [only_id]}

    clustering = AgglomerativeClustering(
        metric=AGGLOMERATIVE_PARAMS["metric"],
        linkage=AGGLOMERATIVE_PARAMS["linkage"],
        distance_threshold=AGGLOMERATIVE_PARAMS["distance_threshold"],
        n_clusters=AGGLOMERATIVE_PARAMS["n_clusters"],
    )
    # REPORT: explain clustering params
    
    ids = list(encoded_edus.keys())
    embeddings = np.array(list(encoded_edus.values()))

    labels = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)

    for edu_id, label in zip(ids, labels):
        clusters[int(label)].append(edu_id)

    return clusters

def validate_cluster(cluster_ids, edu_lookup, threshold=PAIR_VALIDATION_THRESHOLD):
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

    scores = cross_encoder.predict(pairs, batch_size=CROSS_ENCODER_BATCH_SIZE)

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
        self.filter_stats = {
            "input_edus": 0,
            "kept_edus": 0,
            "dropped_edus": 0,
        }

        for article in articles:
            raw_edus = article.get("edus", [])
            self.filter_stats["input_edus"] += len(raw_edus)

            edus = idify_edus(article)
            self.filter_stats["kept_edus"] += len(edus)

            self.edu_lookup.update({edu["id"]: edu for edu in edus})
            if edus:
                self.encoded_edus.update(encode_edus(edus))

        self.filter_stats["dropped_edus"] = self.filter_stats["input_edus"] - self.filter_stats["kept_edus"]
        self.clusters = cluster(self.encoded_edus)
        
    def refine_clusters(self, refine_threshold=DEFAULT_REFINE_THRESHOLD):
        refined = {}

        for cluster_id, edu_ids in self.clusters.items():

            # Keep singleton clusters if they are center EDUs.
            if len(edu_ids) == 1:
                if MIN_BIASES_PER_CLUSTER <= 1:
                    only_id = edu_ids[0]
                    if self.edu_lookup[only_id]["bias"] == SINGLETON_KEEP_BIAS:
                        refined[cluster_id] = [only_id]
                continue # skip validate_cluster

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
            biases_in_cluster = {self.edu_lookup[eid]["bias"] for eid in valid_edus}
            if len(biases_in_cluster) < MIN_BIASES_PER_CLUSTER:
                continue
            


            
            refined[cluster_id] = list(valid_edus)

        self.clusters = refined
        # must be called separately
        
    @staticmethod
    def _roles(nuclearity):
        if nuclearity == "NS":
            return "N", "S"
        if nuclearity == "SN":
            return "S", "N"
        if nuclearity == "NN":
            return "N", "N"
        return None, None

    @staticmethod
    def _satellite_counts(edus, relations):
        parent_of = {}
        role_of = {}

        for rel in relations:
            parent = rel.get("parent")
            left = rel.get("left")
            right = rel.get("right")
            left_role, right_role = FactCluster._roles(rel.get("nuclearity"))

            if left is not None:
                parent_of[left] = parent
                role_of[left] = left_role
            if right is not None:
                parent_of[right] = parent
                role_of[right] = right_role

        sat_edges = {}
        local_role = {}

        for edu in edus:
            edu_id = edu["id"]
            local_role[edu_id] = role_of.get(edu_id)

            count = 0
            cur = edu_id
            seen = set()
            while cur in parent_of and cur not in seen:
                seen.add(cur)
                if role_of.get(cur) == "S":
                    count += 1
                cur = parent_of[cur]

            sat_edges[edu_id] = count

        return sat_edges, local_role

    @staticmethod
    def _article_id(path):
        return os.path.basename(path).replace(".json", "")

    @staticmethod
    def _lookup_for_triplet(triplet, data_dir="data"):
        rst_output_dir = os.path.join(data_dir, "rst_output")
        lookup = {}

        for bias_key in ["left", "center", "right"]:
            triplet_path = triplet.get(bias_key)
            if not triplet_path:
                continue

            article_id = FactCluster._article_id(triplet_path)
            rst_path = os.path.join(rst_output_dir, f"{article_id}.json")

            if not os.path.exists(rst_path):
                continue

            with open(rst_path, "r") as f:
                rst = json.load(f)

            edus = rst.get("edus", [])
            relations = rst.get("relations", [])
            sat_edges, local_role = FactCluster._satellite_counts(edus, relations)

            for edu in edus:
                full_edu_id = f"{article_id}_{edu['id']}"
                lookup[full_edu_id] = {
                    "text": edu["text"],
                    "bias": bias_key,
                    "depth": edu.get("depth"),
                    "role": local_role.get(edu["id"]),
                    "satellite_edges_to_root": sat_edges.get(edu["id"], 0),
                }

        return lookup

    @staticmethod
    def build_facts(cluster_result):
        triplet = cluster_result["triplet"]
        clusters = cluster_result["clusters"]
        edu_lookup = cluster_result["edu_lookup"]
        rst_lookup = FactCluster._lookup_for_triplet(triplet, data_dir=DATA_DIR)
        
        enriched_lookup = {}
        for edu_id, edu_info in edu_lookup.items():
            rst_info = rst_lookup.get(edu_id, {})
            enriched_lookup[edu_id] = {
                "text": rst_info.get("text", edu_info.get("text")),
                "bias": rst_info.get("bias", edu_info.get("bias")),
                "depth": rst_info.get("depth", 0),
                "role": rst_info.get("role"),
                "satellite_edges_to_root": rst_info.get("satellite_edges_to_root", 0),
            }
        
        facts = []
        for cluster_id, edu_ids in clusters.items():
            fact_edus = [enriched_lookup[edu_id] for edu_id in edu_ids if edu_id in enriched_lookup]
            facts.append({
                "cluster_id": cluster_id,
                "edus": fact_edus
            })

        return {
            "triplet_idx": cluster_result.get("triplet_idx"),
            "triplet": triplet,
            "clusters": clusters,
            "edu_lookup": {edu_id: enriched_lookup[edu_id] for edu_ids in clusters.values() for edu_id in edu_ids if edu_id in enriched_lookup},
            "facts": facts,
        }

    
    def get_clusters(self):
        return self.clusters
