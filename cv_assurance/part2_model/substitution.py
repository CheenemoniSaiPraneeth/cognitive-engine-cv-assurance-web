"""Module 7 - substitution analysis: combine identity, architecture, parameter, behaviour, trigger evidence."""


def substitution_analysis(fp, param_res, beh_cmp, store, model_id, cfg, has_ref):
    ev = {"identity_status": fp["identity_status"]}
    if fp["identity_status"] == "NO_EXPECTED_DIGEST" and not has_ref:
        store.add("model", model_id, "identity_unverifiable", "low", 0.9, "substitution_v1", {"reason": "no trusted digest or reference artifact supplied"},
                  "UNAVAILABLE", ["Identity comparison requires a trusted digest."])
        return {"verdict": "UNVERIFIABLE_NO_TRUSTED_REFERENCE", **ev}
    if fp["identity_status"] == "MATCH":
        return {"verdict": "IDENTICAL_ARTIFACT", **ev}
    # hash mismatch from here
    rd = (param_res or {}).get("reference_diff") if isinstance(param_res, dict) else None
    arch_mm = bool(rd and (rd.get("missing_tensors") or rd.get("extra_tensors") or rd.get("shape_mismatch")))
    classes = [d["classification"] for d in (rd or {}).get("tensors", {}).values()] if rd and "tensors" in rd else None
    agree = min((v["agreement"] for v in beh_cmp.values()), default=None) if beh_cmp else None
    normal_agree = beh_cmp["normal:identity"]["agreement"] if beh_cmp else None
    ev.update({"architecture_mismatch": arch_mm, "parameter_classes": sorted(set(classes)) if classes else None,
               "min_variant_agreement": agree, "normal_agreement": normal_agree})
    trig = any(f.category == "trigger_candidate_response" for f in store.findings)
    ev["trigger_response_found"] = trig
    if trig:
        verdict, sev, disp = "MODIFIED_ARTIFACT_WITH_TRIGGER_RESPONSE", "high", "REVIEW"
    elif arch_mm or (normal_agree is not None and normal_agree < 0.85):
        verdict, sev, disp = "DIFFERENT_MODEL_SUSPECTED", "high", "REVIEW"
    elif classes is not None and set(classes) <= {"identical", "numerical_noise"} and (agree is None or agree >= 0.99):
        verdict, sev, disp = "SAME_WEIGHTS_DIFFERENT_BYTES", "low", "REVIEW"   # e.g. re-serialised / metadata change
    elif classes is not None and set(classes) <= {"identical", "numerical_noise", "small_diffuse_change"} and agree is not None and agree >= 0.95:
        verdict, sev, disp = "MINOR_DIFFUSE_WEIGHT_CHANGE_BEHAVIOUR_PRESERVED", "low", "REVIEW"   # benign fine-tune-like update
    elif classes is None and agree is not None and agree >= 0.95:
        verdict, sev, disp = "BEHAVIOURALLY_EQUIVALENT_INTERNALS_UNVERIFIED", "low", "REVIEW"
    else:
        verdict, sev, disp = "MODIFIED_ARTIFACT", "medium", "REVIEW"
    store.add("model", model_id, "model_substitution_assessment", sev, 0.85, "substitution_v1", {"verdict": verdict, **ev}, disp,
              ["Hash mismatch proves the artifact differs, not that it was maliciously modified. Check the approved-version registry."])
    return {"verdict": verdict, **ev}
