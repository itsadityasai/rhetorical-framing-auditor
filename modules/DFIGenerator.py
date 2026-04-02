class DFIGenerator:
    def __init__(self, alpha=0.8, gamma=0.5, clusters=None, edu_lookup=None):
        self.alpha = alpha
        self.gamma = gamma
        self.clusters = clusters
        self.edu_lookup = edu_lookup

    def W(self, depth, sat_count):  # TODO: tune these later
        return (self.alpha ** (depth + 1)) * (self.gamma ** sat_count)
        # REPORT: explain simplified formula

    def get_ps(self, clusters=None, edu_lookup=None):
        """
        Returns:
            dict: {
                cluster_id: {
                    "left": W,
                    "center": W,
                    "right": W
                }
            }
        """
        if clusters is None:
            clusters = self.clusters
        if edu_lookup is None:
            edu_lookup = self.edu_lookup
        
        if self.clusters is None or self.edu_lookup is None:
            raise ValueError("clusters and edu_lookup must be provided either as arguments or during initialization")

      

        cluster_ps = {}

        for cluster_id, edus in clusters.items():

            # group edus in cluster by doc
            doc_edu_map = {"left": [], "center": [], "right": []}

            for edu_id in edus:
                meta = edu_lookup.get(edu_id)
                if meta is None:
                    continue
                doc = meta["bias"]
                doc_edu_map[doc].append(edu_id)

            if len(doc_edu_map["center"]) == 0:
                continue  # already handled, just for good practice

            # compute W for each doc, then take max across EDUs in doc as doc-level W
            W_doc = {}

            for doc in ["left", "center", "right"]:
                edu_ids = doc_edu_map[doc]

                if len(edu_ids) == 0:
                    W_doc[doc] = 0  # omission case
                    continue

                scores = []
                for eid in edu_ids:
                    meta = edu_lookup[eid]

                    depth = meta.get("depth", 0)
                    sat_count = meta.get("satellite_edges_to_root", 0)

                    score = self.W(depth, sat_count)
                    scores.append(score)

                # collapse multiple EDUs → max
                W_doc[doc] = max(scores)

            cluster_ps[cluster_id] = W_doc

        self.cluster_ps = cluster_ps
        return cluster_ps

    @staticmethod
    def build_features(deltas):
        return list(deltas)

    def get_DFIs(self, cluster_ps=None):
        if cluster_ps is None:
            cluster_ps = self.cluster_ps

        deltas_left = []
        deltas_right = []

        for scores in cluster_ps.values():
            W_left = scores["left"]
            W_center = scores["center"]
            W_right = scores["right"]

            deltas_left.append(W_left - W_center)
            deltas_right.append(W_right - W_center)

        dfi_left = self.build_features(deltas_left)
        dfi_right = self.build_features(deltas_right)

        return dfi_left, dfi_right