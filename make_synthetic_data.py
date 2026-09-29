#!/usr/bin/env python3
"""Create a ready-to-run synthetic package for the full Parts 1-5 pipeline (no training of the assessed data; the two fixture models
are built by closed-form ridge fit, only to have artifacts to assess).

  python make_synthetic_data.py --out data_demo [--n-clean 300] [--format coco|yolo]

data_demo/
  dataset/              COCO (or YOLO) training set: clean + duplicate flood + label flips + OOD + trigger patch + corrupt files (source contributor_x)
                        + ground_truth.json (attack manifest - evaluation only, never given to detectors)
  models/               trusted_reference.npz  (clean)   |   trojan_checker.npz (backdoored: checkerboard bottom-right -> 'circle')
  streams/              stable_stream/ (4 normal batches)  |  shifted_stream/ (normal, normal, camera change, illumination change) + metadata.json each
  RUN.txt               exact commands
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cv_assurance.benchmark.part2_models import build_benchmark
from cv_assurance.benchmark.part4_bench import make_batch
from cv_assurance.benchmark.synthetic import generate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data_demo"); ap.add_argument("--n-clean", type=int, default=300)
    ap.add_argument("--format", default="coco", choices=["coco", "yolo"]); a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    generate(out / "dataset", a.n_clean, a.format)
    tmp = out / "_bench"; build_benchmark(tmp)
    (out / "models").mkdir(exist_ok=True)
    for n in ("trusted_reference", "trojan_checker"):
        (out / "models" / f"{n}.npz").write_bytes((tmp / "models" / f"{n}.npz").read_bytes())
    import shutil; shutil.rmtree(tmp)
    for name, kinds in (("stable_stream", ["none"] * 4), ("shifted_stream", ["none", "none", "camera", "illumination"])):
        d = out / "streams" / name; meta = {}
        for b, k in enumerate(kinds):
            meta.update(make_batch(d, f"b{b}", 40, k, 500 + b + len(name), source="cam_B" if k == "camera" else "cam_A", ts0=b))
        (d / "metadata.json").write_text(json.dumps(meta, indent=1))
    fmt = a.format
    (out / "RUN.txt").write_text(f"""# full Parts 1-5, TRUSTED model served (expect: data findings + shift REVIEW, no model/inference failures)
python -m cv_assurance part5 integrated --dataset {out}/dataset --model {out}/models/trusted_reference.npz --trusted-model {out}/models/trusted_reference.npz \\
  --stream stable_stream={out}/streams/stable_stream --stream shifted_stream={out}/streams/shifted_stream --format {fmt} --out {out}/run_trusted

# full Parts 1-5, TROJAN model served (expect converging QUARANTINE cases across Parts 1-4)
python -m cv_assurance part5 integrated --dataset {out}/dataset --model {out}/models/trojan_checker.npz --trusted-model {out}/models/trusted_reference.npz \\
  --stream stable_stream={out}/streams/stable_stream --stream shifted_stream={out}/streams/shifted_stream --format {fmt} --out {out}/run_trojan

# open {out}/run_trojan/part5/report_part5.html
""")
    print(f"synthetic package -> {out}\nsee {out}/RUN.txt")


if __name__ == "__main__":
    main()
