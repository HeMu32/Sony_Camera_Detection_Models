"""Collect YOLO26 and Sony host detections on the local COCO 2017 validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/coco2017-eval"
DATASET = ROOT / "datasets"
ANNOTATIONS = ("instances_val2017.json", "person_keypoints_val2017.json")
YOLO_CLASSES = {"airplane", "bird", "cat", "dog"}
SLOTS = {
    "08": ("bird", "coeff_probe_01/sony-slot08-cuda.npz"),
    "10": ("airplane", "r5_s10/sony-slot10-cuda.npz"),
}
CLIP_CONF = 0.001
PINNED = "pinned-95c9834"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(root):
    data = root / "data"
    with zipfile.ZipFile(DATASET / "annotations_trainval2017.zip") as z:
        for name in ANNOTATIONS:
            z.extract(f"annotations/{name}", data)
    images = read_json(data / "annotations" / ANNOTATIONS[0])["images"]
    if len(images) != 5000 or len({im["id"] for im in images}) != 5000:
        raise ValueError("expected 5,000 distinct COCO val2017 images")
    with zipfile.ZipFile(DATASET / "val2017.zip") as z:
        for im in images:
            name = im["file_name"]
            if name != Path(name).name or Path(name).suffix != ".jpg":
                raise ValueError(f"unexpected val2017 image: {name}")
            dst = data / "val2017" / name
            member = z.getinfo(f"val2017/{name}")
            if not dst.is_file() or dst.stat().st_size != member.file_size:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as source, dst.open("wb") as out:
                    shutil.copyfileobj(source, out)
    snapshot = root / PINNED
    if (root / "sony-release.tar").is_file():
        with tarfile.open(root / "sony-release.tar") as archive:
            for member in archive:
                if member.isdir() or not member.isfile():
                    continue
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or ".." in rel.parts or rel.parts[0] not in ("tools", "parsers"):
                    raise ValueError(f"unexpected source archive entry: {member.name}")
                target = snapshot.joinpath(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    print(f"Prepared {len(images)} images under {data}", flush=True)


def append_predictions(out, manifest, rows):
    """Each line includes an image even if it has no predictions."""
    path = out.with_suffix(".manifest.json")
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError(f"cannot resume with changed inputs: {path}")
    else:
        if out.exists():
            raise ValueError(f"existing results have no provenance: {out}")
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    done = {row["image_id"] for row in load_predictions(out)} if out.exists() else set()
    expected = set(manifest["image_ids"])
    if not done <= expected:
        raise ValueError("prediction stream contains unexpected image IDs")
    elapsed = time.perf_counter()
    with out.open("a", encoding="utf-8") as stream:
        for iid, predictions in rows(done):
            if iid in done or iid not in expected:
                raise ValueError(f"unexpected output image {iid}")
            stream.write(json.dumps({"image_id": iid, "pred": predictions}, allow_nan=False) + "\n")
            stream.flush()
            done.add(iid)
            if len(done) % 250 == 0:
                print(f"{out.stem}: {len(done)}/{len(expected)} ({time.perf_counter()-elapsed:.1f}s)",
                      flush=True)
    if done != expected:
        raise ValueError(f"incomplete results in {out}: {len(done)}/{len(expected)}")


def load_predictions(path):
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["image_id"] in ids:
                raise ValueError(f"duplicate image in {path}: {row['image_id']}")
            ids.add(row["image_id"])
            yield row


def yolo(path, task, images, root, batch, *, person_only=False):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    ann = root / "data/annotations" / ANNOTATIONS[task == "pose"]
    run_name = f"yolo-{task}" + ("-person" if person_only else "")
    manifest = {
        "backend": "ultralytics-predict", "task": task, "weights": path.name,
        "weights_sha256": digest(path), "annotation_sha256": digest(ann),
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(), "image_ids": [im["id"] for im in images],
        "imgsz": 640, "batch": batch, "rect": False, "conf": CLIP_CONF, "max_det": 300,
        "source_loader": "streamed-file-list",
        "classes_retained": ["person"] if person_only or task == "pose" else sorted(YOLO_CLASSES),
    }

    def produce(done):
        model = YOLO(path)
        if model.task != task:
            raise ValueError(f"{path} declares task {model.task}, expected {task}")
        expected = {p["file_name"]: p for p in images if p["id"] not in done}
        if not expected:
            return
        listing = root / f"{run_name}-source.txt"
        listing.write_text("".join(f"{root / 'data/val2017' / name}\n" for name in expected),
                           encoding="utf-8")
        for r in model.predict(
            source=str(listing),
            stream=True, batch=batch, imgsz=640, conf=CLIP_CONF, iou=0.7,
            max_det=300, device=0, half=False, rect=False, verbose=False,
        ):
            im = expected[Path(r.path).name]
            if r.orig_shape != (im["height"], im["width"]):
                raise ValueError(f"YOLO input dimension disagrees with annotation: {r.path}")
            boxes = r.boxes.data.cpu().numpy()
            kpt = r.keypoints.data.cpu().numpy() if task == "pose" else None
            pred = []
            for i, (x1, y1, x2, y2, confidence, cls) in enumerate(boxes.tolist()):
                name = r.names[int(cls)]
                if name not in manifest["classes_retained"]:
                    continue
                row = {"label": name, "bbox": [x1, y1, x2, y2], "score": confidence}
                if kpt is not None:
                    if kpt[i].shape != (17, 3):
                        raise ValueError("expected 17 COCO keypoints with confidences")
                    row["keypoints"] = kpt[i].tolist()
                pred.append(row)
            yield im["id"], pred

    append_predictions(root / f"{run_name}.jsonl", manifest, produce)


def anchor_boxes(slot, output, specs, geom):
    import numpy as np
    import torch
    from torchvision.ops import nms
    from render_anchor_probes import HEADS, class_scores

    head = HEADS[slot]
    tensors = []
    for i in (head.cls, head.reg):
        shape, scale, zero = specs[i]
        tensor = output[i].cpu().numpy()
        tensors.append((tensor.reshape(shape[::-1])[0].astype(np.float32) - zero) * float(scale))
    scores = class_scores(tensors[0], head.class_count)
    reg = tensors[1].reshape(6, 4, head.grid, head.grid)
    boxes, confidence = [], []
    for a, (sx, sy) in head.center_scales.items():
        y, x = np.nonzero(scores[a] >= CLIP_CONF)
        delta = reg[a][:, y, x].T
        center = (np.column_stack((x, y)) + .5) * (320 / head.grid) + delta[:, :2] * [sx, sy]
        half = 5 * np.array([sx, sy]) * np.exp(.2 * delta[:, 2:])
        boxes.extend(np.column_stack((center - half, center + half)))
        confidence.extend(scores[a, y, x])
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    confidence = np.array(confidence, dtype=np.float32)
    kept = nms(torch.from_numpy(boxes), torch.from_numpy(confidence), .35).numpy()
    xy = (boxes[kept].reshape(-1, 2, 2) - geom["offset"]) / geom["scale_xy"]
    xy = np.clip(xy, 0, geom["source_size"])
    valid = (xy[:, 1] > xy[:, 0]).all(1)
    out = [{"label": SLOTS[slot][0], "bbox": b, "score": s}
           for b, s in zip(xy[valid].reshape(-1, 4).tolist(), confidence[kept][valid].tolist())]
    return out[:300]


def sony_detect(root, images, mode, slot):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    release = root / PINNED / "tools"
    if not release.exists():
        raise ValueError("prepare the pinned Sony release snapshot first")
    sys.path.insert(0, str(release))
    from prepare_nbg_inputs import prepare_rgb
    from slot08_cuda import Slot08, output_specs as specs08
    from slot10_cuda import Slot10, output_specs as specs10

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model_type, make_specs = (Slot08, specs08) if slot == "08" else (Slot10, specs10)
    name, pack = SLOTS[slot]
    package = ROOT / "debug/cpu_raw" / pack
    graph_hash = {}
    for model in ("r5", "1m2"):
        path = ROOT / "nab_graphs" / f"{model}_slots" / f"slot{slot}.nb"
        graph_hash[model] = digest(path)
        if graph_hash[model] != model_type.model_hash:
            raise ValueError(f"wrong original Sony graph: {path}")
    manifest = {
        "backend": "sony-cuda", "task": "detect", "mode": mode, "slot": slot, "label": name,
        "source_graph_sha256": graph_hash, "package_sha256": digest(package),
        "release_archive_sha256": digest(root / "sony-release.tar"),
        "annotation_sha256": digest(root / "data/annotations" / ANNOTATIONS[0]),
        "image_ids": [im["id"] for im in images], "torch": torch.__version__,
        "device": torch.cuda.get_device_name(), "conf": CLIP_CONF, "nms_iou": 0.35,
        "max_det_per_class": 300, "input_size": [320, 320],
    }

    def produce(done):
        model = model_type(package, rounding=mode)
        specs = make_specs(model)
        for im in images:
            if im["id"] in done:
                continue
            with Image.open(root / "data/val2017" / im["file_name"]) as src:
                pic = ImageOps.exif_transpose(src).convert("RGB")
            if pic.size != (im["width"], im["height"]):
                raise ValueError(f"Sony input dimension disagrees with annotation: {im['file_name']}")
            planar, geom = prepare_rgb(pic, 320, 320, "letterbox", 128, "rgb", "chw")
            rgb = torch.from_numpy(np.frombuffer(planar, np.uint8).copy().reshape(3, 320, 320))
            yield im["id"], anchor_boxes(slot, model.infer(rgb), specs, geom)

    append_predictions(root / f"sony-{slot}-{mode}.jsonl", manifest, produce)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=RUN)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    y = sub.add_parser("yolo")
    y.add_argument("task", choices=("detect", "pose"))
    y.add_argument("--batch", type=int, default=16)
    y.add_argument("--person-only", action="store_true",
                   help="retain only person boxes from the detection model in a separate run")
    s = sub.add_parser("sony-detect")
    s.add_argument("mode", choices=("simulator", "simple"))
    s.add_argument("slot", choices=tuple(SLOTS))
    for sp in (y, s):
        sp.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.action == "yolo" and args.person_only and args.task != "detect":
        y.error("--person-only requires the detect task")
    root = args.run.resolve()
    if args.action == "prepare":
        prepare(root)
        return
    task = args.task if args.action == "yolo" else "detect"
    ann_path = root / "data/annotations" / ANNOTATIONS[task == "pose"]
    images = sorted(read_json(ann_path)["images"], key=lambda im: im["id"])[:args.limit]
    if args.action == "yolo":
        weight = "yolo26n-pose.pt" if task == "pose" else "yolo26n.pt"
        yolo(root / weight, task, images, root, args.batch, person_only=args.person_only)
    else:
        sony_detect(root, images, args.mode, args.slot)


if __name__ == "__main__":
    main()
