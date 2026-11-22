"""Export experimental slot-01 pose peaks as COCO-keypoints-shaped detections."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from inspect_nbg_spatial_outputs import outputs


COCO_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def source_point(x, y, geometry, output_shape):
    _, height, width = output_shape[-3:]
    iw, ih = geometry["input_size"]
    ox, oy = geometry["offset"]
    sx, sy = geometry["scale_xy"]
    return [((x + .5) * iw / width - ox) / sx,
            ((y + .5) * ih / height - oy) / sy]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("boxes_json", type=Path)
    cli.add_argument("pose_json", type=Path)
    cli.add_argument("prefix", type=Path)
    cli.add_argument("pose_log", type=Path)
    cli.add_argument("manifest", type=Path)
    cli.add_argument("photo", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    boxes = json.loads(args.boxes_json.read_text(encoding="utf-8"))
    pose = json.loads(args.pose_json.read_text(encoding="utf-8"))
    pose_tensors = outputs(args.prefix, args.pose_log)
    geometry = json.loads(args.manifest.read_text(encoding="utf-8"))[
        args.photo.stem + ".raw"]
    with Image.open(args.photo) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
    image_id, category_id = 1, 1
    annotations = []
    for annotation_id, box in enumerate(boxes["boxes"], 1):
        lookup = {int(p[0]): p for p in box["joints_grid"]}
        keypoints, visible = [], 0
        for channel in range(17):
            point = lookup.get(channel)
            if point is None:
                keypoints += [0., 0., 0]
                continue
            x, y = source_point(point[1] - .5, point[2] - .5, geometry,
                                pose_tensors[1].shape)
            score = float(point[3])
            if not (0 <= x < image.width and 0 <= y < image.height):
                keypoints += [0., 0., 0]
                continue
            keypoints += [x, y, 2 if score >= .5 else 1]
            visible += 1
        x1, y1, x2, y2 = box["xyxy_source"]
        annotations.append({
            "id": annotation_id, "image_id": image_id, "category_id": category_id,
            "keypoints": keypoints, "num_keypoints": visible,
            "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
            "iscrowd": 0, "score": float(box["center_pair_score"]),
        })
    result = {
        "info": {"description": "Experimental Sony slot-01 host postprocessing"},
        "licenses": [],
        "images": [{
            "id": image_id, "file_name": args.photo.name,
            "width": image.width, "height": image.height,
        }],
        "categories": [{
            "id": category_id, "name": "person", "supercategory": "person",
            "keypoints": list(COCO_NAMES), "skeleton": [],
        }],
        "annotations": annotations,
        "_sony_export": {
            "format": "COCO annotations JSON structure; not a COCO GT or official decoder",
            "channel_mapping": "unverified positional placeholder; not firmware-confirmed",
            "visibility_mapping": "2 at peak >= .5, 1 otherwise, 0 for absent or outside",
            "detection_threshold": boxes["threshold"],
            "pose_tensor_shapes": [list(t.shape) for t in pose_tensors],
            "input_geometry": geometry,
            "source_sha256": {
                "image": hashlib.sha256(args.photo.read_bytes()).hexdigest(),
                "pose_json": hashlib.sha256(args.pose_json.read_bytes()).hexdigest(),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(f"{args.output}: {len(annotations)} COCO-shaped detections")


if __name__ == "__main__":
    main()
