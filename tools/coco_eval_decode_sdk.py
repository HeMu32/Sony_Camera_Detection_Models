"""Decode SDK pilot outputs and calibrate Sony pose channels on disjoint images."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from coco_eval_infer import ANNOTATIONS, PINNED, RUN, append_predictions, digest, read_json
from coco_eval_simulator import MODELS


ANIMAL_HEAT_MAX = {
    "animal-r5": .00425675744 * (255 - 8),
    "animal-1m2": .00450269459 * (255 - 15),
}


def transform(points, geom, grid=True):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if grid:
        points = points * 8
    return (points - geom["offset"]) / geom["scale_xy"]


def slot18_boxes(heat, geometry, peaks):
    points = [(x, y, s) for layer in heat for x, y, s in peaks(layer, .02)]
    if not points:
        return []
    xy = np.array([p[:2] for p in points])
    lo, hi = xy.min(0), xy.max(0)
    height = max(hi[1] - lo[1], 1. / 8)
    lo -= [.06 * height, .12 * height]
    hi += [.10 * height, .05 * height]
    xy = np.clip(transform([lo, hi], geometry), 0, geometry["source_size"])
    if not (xy[1] > xy[0]).all():
        return []
    confidence = max(p[2] for p in points) / ANIMAL_HEAT_MAX["animal-1m2"]
    return [{"label": "animal", "score": float(np.clip(confidence, 0., 1.)),
             "bbox": xy.ravel().tolist()}]


def fit_mapping(calibration):
    """Assign global peaks to names; require hold-one-out and local margin support."""
    distances = np.asarray(calibration, dtype=float)
    count = np.isfinite(distances[:, :, 0]).sum(0)

    def assign(trials):
        present = np.isfinite(trials)
        cost = np.divide(
            np.where(present, np.minimum(trials, .5), 0.).sum(0), present.sum(0),
            out=np.full(trials.shape[1:], .5), where=present.sum(0) > 0,
        )
        rows, channels = linear_sum_assignment(cost)
        return dict(zip(rows, channels)), cost

    assignment, cost = assign(distances)
    loo = [assign(np.delete(distances, i, axis=0))[0] for i in range(len(distances))]
    detail, usable = [], []
    for coco, channel in assignment.items():
        selected = distances[:, coco, channel]
        observed = selected[np.isfinite(selected)]
        correct = float(np.count_nonzero(observed <= .15) / len(observed)) if len(observed) else 0.
        repeat = float(sum(a[coco] == channel for a in loo) / len(loo))
        next_cost = np.min(np.delete(cost[coco], channel))
        margin = float(next_cost - cost[coco, channel])
        passes = bool(count[coco] >= 5 and correct >= .625 and repeat >= .75 and margin >= .015)
        detail.append({
            "coco_index": int(coco), "channel": int(channel), "annotated_visible": int(count[coco]),
            "fraction_peak_within_0.15_sqrt_bbox_area": correct, "leave_one_image_out_agreement": repeat,
            "cost_gap_vs_nearest_alternative": margin, "accepted": passes,
        })
        if passes:
            usable.append([int(coco), int(channel)])
    return usable, detail


def calibration_distances(coco, calibration, heat_lookup, geom_lookup):
    by_person = defaultdict(list)
    for ann in coco["annotations"]:
        if not ann["iscrowd"] and ann["num_keypoints"] and min(ann["bbox"][2:]) > 0:
            by_person[ann["image_id"]].append(ann)
    distances = []
    for iid in calibration:
        (ann,) = by_person[iid]
        heat = heat_lookup(iid)
        coco_points = np.asarray(ann["keypoints"], dtype=float).reshape(17, 3)
        flat = heat[:17].reshape(17, -1)
        yi, xi = np.unravel_index(flat.argmax(1), heat.shape[-2:])
        xy = transform(np.column_stack((xi, yi)) + .5, geom_lookup(iid))
        normalized = np.linalg.norm(coco_points[:, None, :2] - xy[None], axis=2)
        normalized /= np.sqrt(ann["bbox"][2] * ann["bbox"][3])
        normalized[coco_points[:, 2] != 2] = np.nan
        distances.append(normalized)
    return distances


def decode(root, stage, calibration_only=False, models=MODELS):
    sys.path.insert(0, str(root / PINNED / "tools"))
    from inspect_nbg_spatial_outputs import outputs
    from render_animal_pose_boxes import components, to_boxes
    from render_pose_boxes import peaks, pose_boxes

    plan = read_json(stage / "job.json")
    subsets = plan["subsets"]
    keys = {(kind, iid): key for kind, iid, key in plan["jobs"]}
    source = stage / "out"
    times = defaultdict(list)
    for kind, iid, key in plan["jobs"]:
        prefix = f"{kind}-{iid:012d}"
        path = source / f"{prefix}.done.json"
        is_calibration = kind == "human" and iid in subsets["calibration"]
        if not is_calibration and (calibration_only or kind not in models):
            continue
        done = read_json(path)
        if done["image_id"] != iid or done["model"] != kind:
            raise ValueError(f"SDK completion mismatch: {prefix}")
        times[kind].append(done["seconds"])

    def tensors(kind, iid, indices):
        prefix = source / f"{kind}-{iid:012d}"
        return outputs(prefix, prefix.with_suffix(".log"), indices=indices)

    def geom(kind, iid):
        return plan["geometry"][keys[kind, iid]]

    coco = read_json(root / "data/annotations" / ANNOTATIONS[1])
    distances = calibration_distances(coco, subsets["calibration"],
                                      lambda iid: tensors("human", iid, {1})[1][0],
                                      lambda iid: geom("human", iid))
    mapping, detail = fit_mapping(distances)
    names = next(c["keypoints"] for c in coco["categories"] if c["name"] == "person")
    for row in detail:
        row["name"] = names[row["coco_index"]]
    (stage / "pose-mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (stage / "pose-calibration.json").write_text(json.dumps({
        "scope": "tentative empirical alignment; no firmware names; calibration not scored",
        "calibration_image_ids": subsets["calibration"], "channels": detail,
        "median_sdk_wall_seconds": {k: float(np.median(v)) for k, v in times.items()},
    }, indent=2) + "\n")
    print(f"Empirically accepted keypoint channels: {mapping}", flush=True)
    if calibration_only:
        return

    for task in ("pose", "animal"):
        (stage / f"{task}-ids.json").write_text(json.dumps(subsets[task], indent=2) + "\n")

    def decode_job(kind, iid):
        geometry = geom(kind, iid)
        size = np.array(geometry["source_size"], dtype=float)
        if kind == "human":
            arrays = tensors(kind, iid, {1})
            boxes = pose_boxes(arrays[1][0], .25)
            label = "person"
        elif kind == "animal-r5":
            vector, heatmap = (a[0] for a in tensors(kind, iid, {0, 1}))
            boxes = to_boxes(components(vector, heatmap))
            label = "animal"
        else:
            return slot18_boxes(tensors(kind, iid, {2})[2][0], geometry, peaks)
        results = []
        for box in boxes:
            xy = np.clip(transform(box["xyxy_grid"], geometry), 0, size)
            if not (xy[1] > xy[0]).all():
                continue
            if kind == "human":
                confidence = box["center_pair_score"]
            else:
                confidence = max(p[3] for p in box["joints_grid"])
                confidence = np.clip(confidence / ANIMAL_HEAT_MAX[kind], 0., 1.)
            row = {"label": label, "score": float(confidence), "bbox": xy.ravel().tolist()}
            if kind == "human":
                keypoints = [None] * 17
                for channel, x, y, value in box["joints_grid"]:
                    xy = transform([x, y], geometry)[0]
                    if (xy >= 0).all() and (xy < size).all():
                        keypoints[channel] = [*xy.tolist(), value]
                row["keypoints"] = keypoints
            results.append(row)
        return results

    for kind in models:
        spec = MODELS[kind]
        images = subsets["pose" if kind == "human" else "animal"]
        manifest = {"backend": "x86-sdk-original-nbg", "model": kind, "graph_sha256": spec[4],
                    "job_sha256": digest(stage / "job.json"), "image_ids": images}
        if kind in ANIMAL_HEAT_MAX:
            manifest["animal_score"] = {"method": "peak/max_representable_heat",
                                        "max_representable_heat": ANIMAL_HEAT_MAX[kind]}
        append_predictions(stage / f"sony-{kind}.jsonl", manifest,
                           lambda done: ((iid, decode_job(kind, iid)) for iid in images if iid not in done))
    print("Decoded all SDK outputs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--stage", choices=("sdk-pilot", "sdk-full"), default="sdk-pilot")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    args = parser.parse_args()
    decode(args.run, args.run / args.stage, args.calibration_only, args.models)


if __name__ == "__main__":
    main()
