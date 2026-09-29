"""Module 5 - Label integrity via kNN agreement, local entropy and cluster purity (no training)."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

DETECTOR = "label_integrity_knn_v1"


def detect_label_anomalies(subset, E, excl, class_names, cfg, store, flags, seed=0):
    """subset: labelled, readable samples aligned with rows of E."""
    c = cfg["label_integrity"]
    m = len(subset)
    if m < c["min_neighbours"] + 2:
        return {"status": "UNAVAILABLE", "reason": "too few labelled samples"}
    ids = [s.sample_id for s in subset]
    lab = np.array([s.primary_label for s in subset])
    max_excl = max((sum(1 for sid in ids if g in excl.get(sid, ())) for g in
                    {g for v in excl.values() for g in v}), default=0)
    kq = min(m, c["k"] + 1 + max_excl)
    dist, ind = NearestNeighbors(n_neighbors=kq, metric="cosine", algorithm="brute").fit(E).kneighbors(E)
    ncls = max(len(set(lab.tolist())), 2)
    kk = int(np.clip(m // c["kmeans_cluster_size"], 2, 200))
    km = KMeans(n_clusters=min(kk, m), n_init=3, random_state=seed).fit(E).labels_
    cl_major, cl_purity = {}, {}
    for cid in set(km.tolist()):
        cnt = Counter(lab[km == cid].tolist())
        top, tc = cnt.most_common(1)[0]
        cl_major[cid], cl_purity[cid] = top, tc / sum(cnt.values())
    n_flag = 0
    for r in range(m):
        mine = excl.get(ids[r], set())
        cand = [(j, d) for j, d in zip(ind[r], dist[r])
                if j != r and not (mine and mine & excl.get(ids[j], set()))][: c["k"]]
        if len(cand) < c["min_neighbours"]:
            continue
        w = np.clip(1 - np.array([d for _, d in cand]), 0, None) + 1e-6
        votes = Counter()
        for (j, _), wi in zip(cand, w):
            votes[int(lab[j])] += wi
        tot = sum(votes.values())
        share = {k: v / tot for k, v in votes.items()}
        dom, dom_share = max(share.items(), key=lambda kv: kv[1])
        assigned = int(lab[r])
        a_share = share.get(assigned, 0.0)
        ent = -sum(p * np.log(p) for p in share.values() if p > 0) / np.log(ncls)
        if dom != assigned and dom_share >= c["min_dominant_share"] and a_share <= c["max_assigned_share"]:
            cid = km[r]
            agree_cluster = cl_major[cid] == dom
            sev = "high" if (dom_share >= 0.9 and agree_cluster and cl_purity[cid] >= 0.8) else "medium"
            conf = dom_share * (1.0 if agree_cluster else 0.85)
            ev = {"assigned_label": class_names.get(assigned, str(assigned)),
                  "dominant_neighbour_label": class_names.get(dom, str(dom)),
                  "neighbour_label_agreement": round(a_share, 3),
                  "dominant_neighbour_share": round(dom_share, 3),
                  "local_label_entropy": round(float(ent), 3), "k": len(cand),
                  "cluster_majority_label": class_names.get(cl_major[cid], str(cl_major[cid])),
                  "cluster_purity": round(float(cl_purity[cid]), 3),
                  "nearest_neighbours": [subset[j].sample_id for j, _ in cand[:5]]}
            store.add("sample", ids[r], "label_anomaly", sev, conf, DETECTOR, ev, "REVIEW",
                      ["Similarity-based evidence only; visually ambiguous classes can trigger this."])
            flags.setdefault(ids[r], set()).add("label_anomaly")
            n_flag += 1
    return {"status": "OK", "flagged": n_flag, "assessed": m}
