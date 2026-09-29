"""Module 3 - white-box parameter analysis (reference diff + reference-free dormant-unit check)."""
from __future__ import annotations

import numpy as np

from .adapters import Unavailable


def tensor_stats(a):
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "mean": float(a.mean()), "std": float(a.std()), "l2": float(np.linalg.norm(a)),
            "abs_max": float(np.abs(a).max()), "sparsity": float((a == 0).mean())}


def diff_tensors(a, b, cfg):
    """Compare received tensor a with trusted b (same shape)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = a - b
    nb = np.linalg.norm(b) + 1e-12
    rel = float(np.linalg.norm(d) / nb)
    out = {"rel_l2": rel, "max_abs_diff": float(np.abs(d).max()), "changed_fraction": float((np.abs(d) > 0).mean())}
    localized = False
    if a.ndim == 2:  # test both axes; a localized edit (e.g. one repurposed neuron) concentrates on few units of one axis
        k = cfg["localized_max_units"]
        for axname, ax in (("columns", 0), ("rows", 1)):
            unit_e = (d ** 2).sum(axis=ax)
            n_units, tot = len(unit_e), unit_e.sum() + 1e-30
            order = np.argsort(-unit_e)
            share = float(unit_e[order[:k]].sum() / tot); changed = int((unit_e > 0).sum())
            if n_units >= 8 and share >= cfg["localized_unit_share"] and changed <= 2 * k:
                localized = True
                out.update({"localized_axis": axname, "top_units": [int(i) for i in order[:k]],
                            "top_units_energy_share": share, "units_changed": changed, "units_total": n_units})
                break
    if rel <= cfg["identical_tol"]: cls = "identical"
    elif rel < cfg["noise_rel_l2"]: cls = "numerical_noise"
    elif localized:
        cls = "localized_change"
    elif rel < cfg["small_rel_l2"]: cls = "small_diffuse_change"
    else: cls = "large_change"
    out["classification"] = cls
    return out


def robust_z(x):
    x = np.asarray(x, np.float64)
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826 + 1e-12
    return (x - med) / mad


def analyze_parameters(adapter, ref_adapter, ref_images, cfg, store, model_id):
    """Returns (result_dict, status). Adds findings to store."""
    cfgp = cfg["params"]
    res = {"status": "OK", "layer_inventory": None, "tensor_stats": {}, "reference_diff": None, "dormant_units": None}
    try:
        res["layer_inventory"] = adapter.layers()
        params = adapter.parameters()
    except Unavailable as e:
        return {"status": "UNAVAILABLE", "reason": str(e)}, "UNAVAILABLE"
    res["tensor_stats"] = {k: tensor_stats(v) for k, v in params.items()}
    res["parameter_count"] = int(sum(v.size for v in params.values()))

    # -- reference diff
    if ref_adapter is not None:
        try:
            rp = ref_adapter.parameters()
            missing, extra = sorted(set(rp) - set(params)), sorted(set(params) - set(rp))
            shape_mm = {k: {"received": list(params[k].shape), "trusted": list(rp[k].shape)}
                        for k in set(rp) & set(params) if params[k].shape != rp[k].shape}
            diffs = {k: diff_tensors(params[k], rp[k], cfgp) for k in set(rp) & set(params) if k not in shape_mm}
            res["reference_diff"] = {"missing_tensors": missing, "extra_tensors": extra, "shape_mismatch": shape_mm, "tensors": diffs}
            if missing or extra or shape_mm:
                store.add("model", model_id, "architecture_mismatch", "high", 0.95, "param_structure_v1",
                          {"missing": missing, "extra": extra, "shape_mismatch": shape_mm}, "REVIEW",
                          ["Different architecture indicates a different model; it does not by itself indicate malice."])
            for k, d in diffs.items():
                c = d["classification"]
                if c in ("identical", "numerical_noise"): continue
                sev = {"small_diffuse_change": "low", "localized_change": "high", "large_change": "high"}[c]
                disp = "REVIEW"
                evd = {"tensor": k, **{kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in d.items()}}
                cat = "parameter_localized_change" if c == "localized_change" else "parameter_difference"
                store.add("model", model_id, cat, sev, 0.9, "param_diff_v1", evd, disp,
                          ["Parameter differences are evidence of model difference, not automatically of compromise."],
                          reference_assets=["trusted_reference_model"])
        except Unavailable as e:
            res["reference_diff"] = {"status": "UNAVAILABLE", "reason": str(e)}

    # -- reference-free: dormant high-influence units
    try:
        if not ref_images:
            raise Unavailable("no reference images for activation statistics")
        acts = adapter.activations(ref_images)
        infl = adapter.unit_influence()
        dorm = {}
        for name, A in acts.items():
            active = (A > 1e-6).mean(axis=0)
            z = robust_z(infl[name])
            flagged = [int(i) for i in np.where((active <= cfgp["dormant_active_frac"]) & (z >= cfgp["influence_z"]))[0]]
            dorm[name] = {"units": int(A.shape[1]), "dead_or_dormant_units": int((active <= cfgp["dormant_active_frac"]).sum()),
                          "flagged_units": flagged,
                          "flagged_detail": [{"unit": i, "active_fraction": round(float(active[i]), 4),
                                              "influence_robust_z": round(float(z[i]), 1)} for i in flagged]}
            if flagged:
                store.add("model", model_id, "dormant_high_influence_unit", "high", 0.75, "activation_stats_v1",
                          {"layer": name, **dorm[name]}, "REVIEW",
                          ["Dormant-on-clean units with unusually large downstream influence are a known backdoor "
                           "signature, but also occur in benign networks."])
        res["dormant_units"] = dorm
    except Unavailable as e:
        res["dormant_units"] = {"status": "UNAVAILABLE", "reason": str(e)}
    return res, "OK"
