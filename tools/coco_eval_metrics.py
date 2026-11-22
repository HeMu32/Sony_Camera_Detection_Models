"""Score collected predictions using Ultralytics matching and AP conventions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
import ultralytics
from scipy.optimize import linear_sum_assignment
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import OKS_SIGMA, ap_per_class, box_iou, kpt_iou, smooth

from coco_eval_infer import ANNOTATIONS, RUN, digest, load_predictions, read_json


PROBE_CONF = .85


class Matcher:
    iouv = torch.linspace(.5, .95, 10)
    match_predictions = BaseValidator.match_predictions


def xyxy(boxes, *, coco=False):
    arr = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if coco:
        arr[:, 2:] += arr[:, :2]
    return arr


def pose_array(rows, channels, *, gt, is_yolo):
    values = np.full((len(rows), len(channels), 3), np.inf, dtype=np.float32)
    for i, row in enumerate(rows):
        points = row["keypoints"]
        if gt:
            points = np.asarray(points, dtype=np.float32).reshape(17, 3)
        for j, (coco, sony) in enumerate(channels):
            pt = points[coco if gt or is_yolo else sony]
            if pt is not None:
                values[i, j] = pt
    return torch.from_numpy(values)


def targets_for(coco, groups, pose):
    id_to_group = {c["id"]: name for c in coco["categories"] for name, classes in groups.items()
                   if c["name"] in classes}
    targets = defaultdict(list)
    for row in coco["annotations"]:
        label = id_to_group.get(row["category_id"])
        if label is None or row.get("iscrowd", 0) or (pose and not row["num_keypoints"]):
            continue
        if min(row["bbox"][2:]) <= 0:
            continue
        targets[row["image_id"]].append(dict(row, label=label))
    return targets


def prediction_rows(paths, groups, image_ids):
    combined = {iid: [] for iid in image_ids}
    seen_groups = set()
    reverse = {cls: group for group, names in groups.items() for cls in (group, *names)}
    for path in paths:
        coverage = set()
        source_groups = set()
        for row in load_predictions(path):
            iid = row["image_id"]
            if iid not in combined:
                continue
            coverage.add(iid)
            for prediction in row["pred"]:
                label = reverse.get(prediction["label"])
                if label is not None:
                    if not 0. <= prediction["score"] <= 1.:
                        raise ValueError(f"confidence outside [0, 1] in {path} for image {iid}")
                    source_groups.add(label)
                    combined[iid].append(dict(prediction, label=label))
        if coverage != set(image_ids):
            raise ValueError(f"{path} lacks {len(image_ids)-len(coverage)} scored images")
        if source_groups & seen_groups:
            raise ValueError(f"multiple prediction streams cover the same class: {path}")
        seen_groups |= source_groups
    return combined


class KeypointErrors:
    def __init__(self, length):
        self.errors = [[] for _ in range(length)]
        self.norm_errors = [[] for _ in range(length)]
        self.annotated = np.zeros(length, dtype=np.int64)
        self.matches = 0
        self.people = 0

    def add(self, gt, pred, box_iou_scores, scale):
        self.annotated += (gt[:, :, 2] != 0).sum(0).numpy()
        self.people += len(gt)
        if not box_iou_scores.numel():
            return
        similarity = box_iou_scores.numpy()
        pairs = linear_sum_assignment(np.where(similarity >= .5, similarity, 0.), maximize=True)
        for g, p in zip(*pairs):
            if similarity[g, p] < .5:
                continue
            self.matches += 1
            for k in np.flatnonzero(gt[g, :, 2].numpy()):
                pt = pred[p, k, :2]
                if torch.isfinite(pt).all():
                    distance = torch.linalg.vector_norm(pt - gt[g, k, :2]).item()
                    self.errors[k].append(distance)
                    self.norm_errors[k].append(distance / float(scale[g]))

    def summary(self, names):
        result = {}
        for name, count, err, norm in zip(names, self.annotated, self.errors, self.norm_errors):
            num = int(count)
            result[name] = {
                "annotated": num, "paired_with_prediction": len(err),
                "mean_px_matched": float(np.mean(err)) if err else None,
                "median_px_matched": float(np.median(err)) if err else None,
                "mean_error_per_sqrt_box_area_matched": float(np.mean(norm)) if norm else None,
                "PCK_0.10_all_gt": float(np.count_nonzero(np.asarray(norm) <= .10)/num) if num else None,
            }
        errors = [v for points in self.errors for v in points]
        norm = [v for points in self.norm_errors for v in points]
        count = int(self.annotated.sum())
        pooled = {
            "annotated": count, "paired_with_prediction": len(errors),
            "mean_px_matched": float(np.mean(errors)) if errors else None,
            "median_px_matched": float(np.median(errors)) if errors else None,
            "mean_error_per_sqrt_box_area_matched": float(np.mean(norm)) if norm else None,
            "PCK_0.10_all_gt": float(np.count_nonzero(np.asarray(norm) <= .10) / count) if count else None,
        }
        return {"bbox_iou_min": .5, "matched_people": self.matches, "eligible_people": self.people,
                "pooled": pooled, "per_keypoint": result}


def score(coco, preds, groups, channels=None, is_yolo=False):
    pose = channels is not None
    classes = list(groups)
    ids = {c: i for i, c in enumerate(classes)}
    targets = targets_for(coco, groups, pose)
    tp, tp_fixed, conf, pred_class, true_class = [], [], [], [], []
    matcher = Matcher()
    errors = KeypointErrors(len(channels)) if pose else None
    pose_error_inputs = []
    n_images = defaultdict(int)
    for iid, rows in preds.items():
        gt = targets[iid]
        gc = torch.tensor([ids[a["label"]] for a in gt], dtype=torch.int64)
        pc = torch.tensor([ids[a["label"]] for a in rows], dtype=torch.int64)
        gboxes, pboxes = xyxy([a["bbox"] for a in gt], coco=True), xyxy([a["bbox"] for a in rows])
        sim = box_iou(gboxes, pboxes)
        if pose:
            gpts, ppts = (pose_array(g, channels, gt=is_gt, is_yolo=is_yolo)
                          for g, is_gt in ((gt, True), (rows, False)))
            area = (gboxes[:, 2:] - gboxes[:, :2]).prod(1)
            scales = area.sqrt()
            errors.add(gpts, ppts, sim, scales)
            pose_error_inputs.append((gpts, ppts, sim, scales, [a["score"] for a in rows]))
            sim = kpt_iou(gpts, ppts, .53 * area, OKS_SIGMA[[i for i, _ in channels]].tolist())
        tp.append(matcher.match_predictions(pc, gc, sim).numpy())
        if not pose:
            keep = torch.tensor([a["score"] >= PROBE_CONF for a in rows], dtype=torch.bool)
            tp_fixed.append(matcher.match_predictions(pc[keep], gc, sim[:, keep]).numpy()[:, 0])
        conf.extend(a["score"] for a in rows)
        pred_class.extend(pc.tolist())
        true_class.extend(gc.tolist())
        for label in set(a["label"] for a in gt):
            n_images[label] += 1
    pc, gc = np.asarray(pred_class, dtype=int), np.asarray(true_class, dtype=int)
    confidence = np.asarray(conf)
    tp, fp, p, r, f1, ap, eligible, _, _, f1_curve, conf_grid, _ = ap_per_class(
        np.concatenate(tp, axis=0), confidence, pc, gc,
    )
    best_conf = float(conf_grid[smooth(f1_curve.mean(0), .1).argmax()]) if eligible.size else None
    values = {}
    for i, label in enumerate(classes):
        row = {"gt_images": n_images[label], "gt_instances": int((gc == i).sum()),
               "predictions": int((pc == i).sum()), "precision": None, "recall": None, "f1": None,
               "ap50": None, "ap75": None, "ap50_95": None, "tp_at_max_f1": None, "fp_at_max_f1": None}
        found = np.flatnonzero(eligible == i)
        if len(found):
            j = found[0]
            row.update(precision=float(p[j]), recall=float(r[j]), f1=float(f1[j]),
                       ap50=float(ap[j, 0]), ap75=float(ap[j, 5]), ap50_95=float(ap[j].mean()),
                       tp_at_max_f1=int(tp[j]), fp_at_max_f1=int(fp[j]))
        if not pose:
            class_at_fixed = pc[confidence >= PROBE_CONF] == i
            num_pred = int(class_at_fixed.sum())
            num_tp = int(np.concatenate(tp_fixed)[class_at_fixed].sum())
            row["at_conf_0.85_iou_0.50"] = {
                "tp": num_tp, "fp": num_pred - num_tp,
                "precision": num_tp / num_pred if num_pred else None,
                "recall": num_tp / row["gt_instances"] if row["gt_instances"] else None,
            }
        values[label] = row
    result = {"metric": "OKS" if pose else "box IoU", "image_count": len(preds), "classes": values,
              "confidence_at_smoothed_max_mean_f1": best_conf}
    if pose:
        names = next(c["keypoints"] for c in coco["categories"] if c["name"] == "person")
        selected = [names[i] for i, _ in channels]
        result["keypoint_errors"] = errors.summary(selected)
        if best_conf is not None:
            operating_errors = KeypointErrors(len(channels))
            for gpts, ppts, sim, scales, scores in pose_error_inputs:
                keep = torch.tensor(scores, dtype=torch.float32) >= best_conf
                operating_errors.add(gpts, ppts[keep], sim[:, keep], scales)
            result["keypoint_errors_at_smoothed_max_mean_f1"] = operating_errors.summary(selected)
    return result


def evaluate(root, task, paths, out, class_names, subset=None, mapping=None):
    torch.set_num_threads(2)
    pose = task == "pose"
    ann = root / "data/annotations" / ANNOTATIONS[pose]
    coco = read_json(ann)
    ids = sorted(subset if subset is not None else [im["id"] for im in coco["images"]])
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= {i["id"] for i in coco["images"]}:
        raise ValueError("invalid scored image set")
    groups = {name: [name] if name != "animal" else ["cat", "dog"] for name in class_names}
    if pose and list(groups) != ["person"]:
        raise ValueError("pose scoring requires person category")
    if any(name not in {c["name"] for c in coco["categories"]}
           for members in groups.values() for name in members):
        raise ValueError("class outside COCO annotations")
    members = [name for names in groups.values() for name in names]
    if len(members) != len(set(members)):
        raise ValueError("category groups must not overlap")
    pred = prediction_rows(paths, groups, ids)
    manifests = [read_json(p.with_suffix(".manifest.json")) for p in paths]
    backends = {m["backend"] for m in manifests}
    modes = {m.get("mode") for m in manifests}
    is_yolo = backends == {"ultralytics-predict"}
    multi_sony = (not pose and all(m["backend"].startswith("sony-") for m in manifests)
                  and modes in ({"simulator"}, {"simple"}))
    if len(modes) != 1 or (len(backends) != 1 and not multi_sony):
        raise ValueError("prediction streams use different inference backends")
    channels = list(enumerate(range(17))) if pose and mapping is None else mapping
    if pose and not is_yolo and not mapping:
        raise ValueError("cannot score Sony poses without an independently established mapping")
    result = score(coco, pred, groups, channels, is_yolo)
    result.update({
        "ultralytics": ultralytics.__version__, "annotations": ann.name, "annotation_sha256": digest(ann),
        "inputs": {p.name: digest(p) for p in paths}, "image_ids": ids, "category_mapping": groups,
        "pose_channels": channels,
    })
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{out.name}: {len(ids)} images {json.dumps(result['classes'])}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("detect", "pose"))
    p.add_argument("output", type=Path)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--classes", nargs="+")
    p.add_argument("--subset", type=Path, help="JSON list of image IDs")
    p.add_argument("--mapping", type=Path, help="JSON list of validated [COCO index, Sony channel]")
    args = p.parse_args()
    evaluate(args.run, args.task, args.inputs, args.output,
             args.classes or (["person"] if args.task == "pose" else ["bird", "airplane"]),
             read_json(args.subset) if args.subset else None, read_json(args.mapping) if args.mapping else None)


if __name__ == "__main__":
    main()
