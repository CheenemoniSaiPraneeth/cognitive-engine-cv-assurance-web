"""Module 4 - Duplicate / flooding detection (exact, near, semantic) + conflicting-label check."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors

DETECTOR = "duplicate_flooding_v1"
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def groups(self):
        g = {}
        for i in range(len(self.p)):
            g.setdefault(self.find(i), []).append(i)
        return [m for m in g.values() if len(m) > 1]


def _hamming_pairs(h1: np.ndarray, h2: np.ndarray, max1: int, max2: int, chunk=512):
    """Yield index pairs (i<j) with hamming(h1)<=max1 AND hamming(h2)<=max2."""
    n = len(h1)
    for s in range(0, n, chunk):
        a1, a2 = h1[s:s + chunk, None], h2[s:s + chunk, None]
        d1 = _POP[(a1 ^ h1[None, :]).view(np.uint8)].reshape(a1.shape[0], n, 8).sum(-1)
        d2 = _POP[(a2 ^ h2[None, :]).view(np.uint8)].reshape(a2.shape[0], n, 8).sum(-1)
        ii, jj = np.where((d1 <= max1) & (d2 <= max2))
        for i, j in zip(ii, jj):
            if s + i < j:
                yield s + i, j


def find_groups(samples, E, cfg):
    """samples: readable SampleRecords aligned with rows of E (normalised embeddings)."""
    n = len(samples)
    d = cfg["duplicates"]
    groups = []
    # exact
    by_sha = {}
    for i, s in enumerate(samples):
        by_sha.setdefault(s.image_sha256, []).append(i)
    exact = [m for m in by_sha.values() if len(m) > 1]
    groups += [("exact", m) for m in exact]
    # near (perceptual hash); union-find over pairs
    ph = np.array([int(s.perceptual_hash, 16) for s in samples], dtype=np.uint64)
    dh = np.array([int(s.difference_hash, 16) for s in samples], dtype=np.uint64)
    uf = UnionFind(n)
    for i, j in _hamming_pairs(ph, dh, d["phash_max_dist"], d["dhash_max_dist"]):
        uf.union(i, j)
    groups += [("near", m) for m in uf.groups()]
    # semantic (embedding cosine)
    nn = NearestNeighbors(radius=1 - d["semantic_min_cosine"], metric="cosine", algorithm="brute").fit(E)
    nbrs = nn.radius_neighbors(E, return_distance=False)
    uf2 = UnionFind(n)
    for i, js in enumerate(nbrs):
        for j in js:
            if j != i:
                uf2.union(i, int(j))
    groups += [("semantic", m) for m in uf2.groups()]
    return groups


def detect_duplicates(samples, E, cfg, store, flags, class_names):
    """Returns (groups_out, excl_map). excl_map[sample_id] -> set(group ids) for exact/near groups."""
    d = cfg["duplicates"]
    n = len(samples)
    raw = find_groups(samples, E, cfg)
    seen_sets = []
    out, excl = [], {}
    ordered = sorted(raw, key=lambda t: ({"exact": 0, "near": 1, "semantic": 2}[t[0]], -len(t[1])))
    gid = 0
    for level, members in ordered:
        ms = frozenset(members)
        if any(ms == s for s in seen_sets):
            continue  # already reported at a more specific level
        seen_sets.append(ms)
        if level == "semantic" and any(ms <= s for s in seen_sets[:-1]):
            continue
        gid += 1
        gname = f"G-{gid:03d}"
        ids = [samples[i].sample_id for i in members]
        cls = Counter(samples[i].primary_label for i in members if samples[i].primary_label is not None)
        srcs = Counter((samples[i].source_id or samples[i].contributor_id) for i in members
                       if (samples[i].source_id or samples[i].contributor_id))
        size, frac = len(members), len(members) / n
        top_cls = cls.most_common(1)[0] if cls else (None, 0)
        top_src = srcs.most_common(1)[0] if srcs else (None, 0)
        flooding = size >= d["flood_min_group_size"] or frac >= d["flood_min_fraction"]
        review = size >= d["review_min_group_size"]
        if level == "semantic" and not flooding:
            review = False  # semantic similarity alone is weak evidence for small groups
        rec = {"group_id": gname, "level": level, "size": size, "fraction": round(frac, 4),
               "flooding": bool(flooding and review), "members": ids}
        out.append(rec)
        if level in ("exact", "near"):
            for sid in ids:
                excl.setdefault(sid, set()).add(gname)
        ev = {"level": level, "group_size": size, "fraction_of_dataset": round(frac, 4),
              "top_class": class_names.get(top_cls[0], str(top_cls[0])) if top_cls[0] is not None else None,
              "top_class_fraction": round(top_cls[1] / max(sum(cls.values()), 1), 3) if cls else None,
              "top_source": top_src[0], "top_source_fraction": round(top_src[1] / max(sum(srcs.values()), 1), 3) if srcs else None,
              "member_count": size, "member_sample_ids": ids[:200],
              "member_ids_truncated": size > 200}
        if flooding and review:
            sev = "high" if (frac >= 0.05 or size >= 25) else "medium"
            cat = "duplicate_flooding"
            disp = "REVIEW"
            conf = min(0.99, 0.6 + 0.4 * min(1.0, size / 30))
        elif review:
            sev, cat, disp, conf = "low", "duplicate_group", "REVIEW", 0.6
        else:
            sev, cat, disp, conf = "low", "duplicate_pair_informational", "ACCEPT", 0.5
        store.add("group", gname, cat, sev, conf, DETECTOR, ev, disp,
                  ["Duplicates can be legitimate (burst capture, augmentation); policy is domain-specific."])
        if disp == "REVIEW":
            for sid in ids:
                flags.setdefault(sid, set()).add("duplicate_group")
        # conflicting labels inside exact/near groups
        if level in ("exact", "near"):
            labs = {samples[i].primary_label for i in members if samples[i].primary_label is not None}
            if len(labs) > 1:
                store.add("group", gname, "conflicting_duplicate_labels", "medium", 0.9, DETECTOR,
                          {"labels": sorted(class_names.get(l, str(l)) for l in labs), "member_sample_ids": ids[:200]},
                          "REVIEW", ["Identical/near-identical images carry different labels."])
                for sid in ids:
                    flags.setdefault(sid, set()).add("conflicting_duplicate_labels")
    return out, excl
