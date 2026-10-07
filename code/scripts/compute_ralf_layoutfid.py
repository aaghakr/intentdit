"""LayoutFID with the FIDNetV3 layout encoders released with RALF (Horita et al., CVPR 2024).

RALF trained one FIDNetV3 encoder per dataset (PKU: ``pku10``, CGL: ``cgl``) on the
same train split we use, and computes FID between encoder features of generated and
ground-truth test layouts, each truncated to 10 elements. This script follows that
protocol for any prediction file in the shared format.

Before scoring, it checks the adapter: ground-truth test features computed from our
annotations are compared with RALF's cached ground-truth features
(``cache/eval_gt_features/<name>_FIDNetV3_features.pth``).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import sqrtm

ROOT = Path(__file__).resolve().parents[2]
MAX_ELEMENTS = 10
# Our class ids (1 text, 2 logo, 3 underlay, 4 embellishment) -> RALF ids (sorted names).
OURS_TO_RALF = {
    "pku": {1: 1, 2: 0, 3: 2},  # ['logo', 'text', 'underlay']
    "cgl": {1: 2, 2: 1, 3: 3, 4: 0},  # ['embellishment', 'logo', 'text', 'underlay']
}
RALF_NAME = {"pku": "pku10", "cgl": "cgl"}
NUM_CLASSES = {"pku": 3, "cgl": 4}
DUMMY = (0, [0.5], [0.5], [0.05], [0.05])  # RALF's placeholder for empty layouts


def load_encoder(ralf_root: Path, dataset: str, device: torch.device):
    sys.path.insert(0, str(ralf_root))
    from image2layout.train.fid.model import FIDNetV3

    model = FIDNetV3(num_label=NUM_CLASSES[dataset], max_bbox=MAX_ELEMENTS)
    state = torch.load(ralf_root / f"cache/PRECOMPUTED_WEIGHT_DIR/fidnet/{RALF_NAME[dataset]}/model_best.pth.tar",
                       map_location="cpu")
    model.load_state_dict(state["state_dict"])
    return model.eval().to(device)


def features(model, layouts: list[dict], device: torch.device, batch: int = 512) -> np.ndarray:
    out = []
    for start in range(0, len(layouts), batch):
        chunk = layouts[start : start + batch]
        size = len(chunk)
        tensors = {k: torch.zeros(size, MAX_ELEMENTS) for k in ("center_x", "center_y", "width", "height")}
        label = torch.zeros(size, MAX_ELEMENTS, dtype=torch.long)
        mask = torch.zeros(size, MAX_ELEMENTS, dtype=torch.bool)
        for i, lay in enumerate(chunk):
            n = min(len(lay["label"]), MAX_ELEMENTS)
            if n == 0:
                lay = {"label": [DUMMY[0]], "center_x": DUMMY[1], "center_y": DUMMY[2], "width": DUMMY[3], "height": DUMMY[4]}
                n = 1
            label[i, :n] = torch.tensor(lay["label"][:n])
            for k in tensors:
                tensors[k][i, :n] = torch.tensor(lay[k][:n], dtype=torch.float32)
            mask[i, :n] = True
        inputs = {**{k: v.to(device) for k, v in tensors.items()}, "label": label.to(device), "mask": mask.to(device)}
        with torch.no_grad():
            out.append(model.extract_features(inputs).cpu().numpy())
    return np.concatenate(out, axis=0)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff @ diff + np.trace(cov_a + cov_b - 2 * covmean))


def stem(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0]


def ground_truth(dataset: str) -> dict[str, dict]:
    frame = pd.read_csv(ROOT / f"data/dataset/{dataset}/split/csv/test.csv")
    mapping = OURS_TO_RALF[dataset]
    layouts: dict[str, dict] = {}
    for _, row in frame.iterrows():
        x1, y1, x2, y2 = json.loads(str(row.box_elem).replace("(", "[").replace(")", "]"))
        x1, x2 = sorted((x1 / 513.0, x2 / 513.0))
        y1, y2 = sorted((y1 / 750.0, y2 / 750.0))
        x1, x2, y1, y2 = (min(max(v, 0.0), 1.0) for v in (x1, x2, y1, y2))
        lay = layouts.setdefault(row.poster_path, {"label": [], "center_x": [], "center_y": [], "width": [], "height": []})
        lay["label"].append(mapping[int(row.cls_elem)])
        lay["center_x"].append((x1 + x2) / 2); lay["center_y"].append((y1 + y2) / 2)
        lay["width"].append(x2 - x1); lay["height"].append(y2 - y1)
    return layouts


def from_prediction_file(path: Path, dataset: str) -> dict[str, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mapping = OURS_TO_RALF[dataset]
    layouts = {}
    for name, rows in zip(payload["img_names"], payload["test_output"]):
        lay = {"label": [], "center_x": [], "center_y": [], "width": [], "height": []}
        for row in rows.tolist():
            cls = int(round(row[0]))
            if cls <= 0 or cls not in mapping:
                continue
            cx, cy, w, h = row[1:]
            x1, x2 = max(cx - w / 2, 0.0), min(cx + w / 2, 1.0)
            y1, y2 = max(cy - h / 2, 0.0), min(cy + h / 2, 1.0)
            if (x2 - x1) * (y2 - y1) < 1e-3:  # degenerate boxes are filtered before evaluation
                continue
            lay["label"].append(mapping[cls])
            lay["center_x"].append((x1 + x2) / 2); lay["center_y"].append((y1 + y2) / 2)
            lay["width"].append(x2 - x1); lay["height"].append(y2 - y1)
        layouts[os.path.basename(str(name))] = lay
    return layouts


def from_ralf_pickle(path: Path, dataset: str, id_to_name: dict[str, str]) -> dict[str, dict]:
    results = pickle.load(open(path, "rb"))["results"]
    return {id_to_name[r["id"]]: {k: list(r[k]) for k in ("label", "center_x", "center_y", "width", "height")} for r in results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--ralf-root", default="/home/viplab/Aagha/Archive/layout/RALF")
    parser.add_argument("--prediction", nargs=2, action="append", default=[], metavar=("LABEL", "PATH"),
                        help="Shared-format prediction file; repeat for each method/seed.")
    parser.add_argument("--ralf-pickle", nargs=2, action="append", default=[], metavar=("LABEL", "PATH"))
    parser.add_argument("--reference", choices=("ralf_cached", "ours"), default="ralf_cached",
                        help="Ground-truth features: RALF's cached test features (official) or recomputed from our annotations.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ralf_root = Path(args.ralf_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_encoder(ralf_root, args.dataset, device)

    # Image order of RALF's test split, and id -> our file name.
    order = torch.load(ralf_root / f"cache/dataset/{args.dataset}_DATAID_TO_IDX.pt", map_location="cpu")["test"]
    ids = [k for k, _ in sorted(order.items(), key=lambda kv: kv[1])]
    if args.dataset == "cgl":
        csv = pd.read_csv(ROOT / "data/dataset/cgl/split/csv/test.csv")
        stem_to_name = {stem(f): p for f, p in zip(csv.file_name, csv.poster_path)}
        id_to_name = {i: stem_to_name[i] for i in ids}
    else:
        id_to_name = {i: f"{i}.png" for i in ids}
    names = [id_to_name[i] for i in ids]

    gt = ground_truth(args.dataset)
    gt_feats = features(model, [gt[n] for n in names], device)
    cached = torch.load(ralf_root / f"cache/eval_gt_features/{RALF_NAME[args.dataset]}_FIDNetV3_features.pth",
                        map_location="cpu")["layout"]["test"].numpy()
    check = {
        "max_abs_diff_vs_ralf_cached_gt": float(np.abs(gt_feats - cached).max()),
        "mean_cosine_vs_ralf_cached_gt": float(np.mean(np.sum(gt_feats * cached, 1)
                                                       / (np.linalg.norm(gt_feats, axis=1) * np.linalg.norm(cached, axis=1)))),
        "fid_ours_gt_vs_ralf_cached_gt": frechet(gt_feats, cached),
    }
    print("adapter check:", check)

    reference = cached if args.reference == "ralf_cached" else gt_feats
    results = {"dataset": args.dataset, "n_images": len(names), "reference": args.reference,
               "adapter_check": check, "fid": {}, "truncated_fraction": {}}
    sources = [(label, from_prediction_file(Path(p), args.dataset)) for label, p in args.prediction]
    sources += [(label, from_ralf_pickle(Path(p), args.dataset, id_to_name)) for label, p in args.ralf_pickle]
    for label, layouts in sources:
        missing = [n for n in names if n not in layouts]
        if missing:
            raise ValueError(f"{label}: {len(missing)} test images missing")
        fake = [layouts[n] for n in names]
        results["fid"][label] = frechet(features(model, fake, device), reference)
        results["truncated_fraction"][label] = float(np.mean([len(l["label"]) > MAX_ELEMENTS for l in fake]))
        print(f"{label:45s} LayoutFID={results['fid'][label]:.3f}  truncated={results['truncated_fraction'][label]:.3f}")
    Path(args.output).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
