"""Measure the localization ceiling of saved COCO candidate streams."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(positives, candidates, confidence, threshold):
    counts = dict(images=len(candidates), positive_images=len(positives),
                  gt_instances=sum(map(len, positives.values())),
                  positive_with_candidate=0, positive_top1_hit=0,
                  positive_oracle_hit=0, oracle_matched_gt=0, negative_with_candidate=0)
    for image_id, rows in candidates.items():
        boxes = sorted((row for row in rows if row["score"] >= confidence),
                       key=lambda row: row["score"], reverse=True)
        if not boxes:
            continue
        if image_id not in positives:
            counts["negative_with_candidate"] += 1
            continue
        counts["positive_with_candidate"] += 1
        gt = np.asarray(positives[image_id], dtype=float)
        gt[:, 2:] += gt[:, :2]
        pred = np.asarray([row["bbox"] for row in boxes])
        wh = (np.minimum(gt[:, None, 2:], pred[None, :, 2:]) -
              np.maximum(gt[:, None, :2], pred[None, :, :2])).clip(0)
        intersection = wh.prod(axis=2)
        union = ((gt[:, 2:] - gt[:, :2]).prod(axis=1)[:, None] +
                 (pred[:, 2:] - pred[:, :2]).prod(axis=1) - intersection)
        hits = intersection >= threshold * union
        counts["positive_top1_hit"] += int(hits[:, 0].any())
        counts["positive_oracle_hit"] += int(hits.any())
        g, p = linear_sum_assignment(hits.astype(int), maximize=True)
        counts["oracle_matched_gt"] += int(hits[g, p].sum())
    return counts


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("annotations", type=Path)
    cli.add_argument("predictions", type=Path)
    cli.add_argument("label")
    cli.add_argument("gt_classes", nargs="+")
    args = cli.parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    categories = {row["id"] for row in coco["categories"] if row["name"] in args.gt_classes}
    if len(categories) != len(set(args.gt_classes)):
        cli.error("unknown GT category")
    positives = {}
    for row in coco["annotations"]:
        if row["category_id"] in categories and not row.get("iscrowd") and min(row["bbox"][2:]) > 0:
            positives.setdefault(row["image_id"], []).append(row["bbox"])
    candidates = {}
    with args.predictions.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            image_id = row["image_id"]
            if image_id in candidates:
                cli.error(f"duplicate image {image_id}")
            candidates[image_id] = [p for p in row["pred"] if p["label"] == args.label]
    if candidates.keys() != {row["id"] for row in coco["images"]}:
        cli.error("prediction stream does not cover exactly the annotation images")
    print(json.dumps({f"{conf:g}": evaluate(positives, candidates, conf, .5)
                      for conf in (0., .85)}, indent=2))


if __name__ == "__main__":
    main()
