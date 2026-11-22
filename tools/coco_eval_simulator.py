"""Stage a reproducible COCO pilot for the pinned Sony x86 SDK simulator."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import threading
import time


MODELS = {
    "human": ("r5_slots/slot01.nb", 512, 336, 2,
              "a5d5ead9478e3b069598483bda95702283d7b0c3d825de02c67250eeea4394b6"),
    "animal-r5": ("r5_slots/slot12.nb", 416, 288, 2,
                  "6edaf8ed0e182ede460276e87f9841f6f29329d685f6237279b14dd3efc4e7bc"),
    "animal-1m2": ("1m2_slots/slot18.nb", 416, 288, 12,
                   "84a35252118fa92ed7a5e36e7f16d5f8b00f0ab6b901d5ba3d83cbc5a4a10e75"),
}
SDK_SOURCES = ("run_sony_nbg_cpu.sh", "nbg_cpu_probe.c", "vsim_nbg_mode.c", "vsim_legacy_sram_reloc.c")
SDK_OUTPUT_BYTES = {
    "human": (107520, 51072),
    "animal-r5": (74880, 35568),
    "animal-1m2": (33696, 1872, 33696, 67392, 3744, 67392, 1872, 3744, 3744, 3744, 3744, 1872),
}
SSH_CONFIG = Path("E:/Virtual Machines/Sony NBG CPU Lab/ssh_config")
SSH_EXE = Path("C:/Windows/System32/OpenSSH")
VM = "sony-nbg-cpu"
VM_SHARD = """\
from pathlib import Path, PurePosixPath
import subprocess, sys, tarfile, tempfile
with tempfile.TemporaryDirectory(prefix='sony-coco2017-') as directory:
    root = Path(directory)
    with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as archive:
        for item in archive:
            relative = PurePosixPath(item.name)
            if not item.isfile() or relative.is_absolute() or '..' in relative.parts:
                raise ValueError(item.name)
            target = root.joinpath(*relative.parts)
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(archive.extractfile(item).read())
    sdk = Path.home() / 'nbg/x86_64_linux'
    proc = subprocess.run(
        [sys.executable, 'coco_eval_simulator.py', 'vm-worker', str(sdk), sys.argv[1]],
        cwd=root, stdout=sys.stderr,
    )
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
        for stamp in (root / 'out').glob('*.done.json'):
            prefix = stamp.name.removesuffix('.done.json')
            for path in sorted(stamp.parent.glob(prefix + '.*')):
                archive.add(path, arcname=path.name)
    sys.exit(proc.returncode)
"""


def stable_take(ids, label, count):
    ordered = sorted(ids, key=lambda i: hashlib.sha256(f"sony-coco2017-v1:{label}:{i}".encode()).digest())
    if len(ordered) < count:
        raise ValueError(f"not enough images in stratum {label}")
    return ordered[:count]


def partition(instances, people, cal_n, pose_n, animal_n):
    by_person = defaultdict(list)
    any_person = set()
    dogs, cats = set(), set()
    category = {row["id"]: row["name"] for row in instances["categories"]}
    ids = {im["id"] for im in instances["images"]}
    for row in instances["annotations"]:
        name = category[row["category_id"]]
        if name == "person":
            any_person.add(row["image_id"])
        if not row["iscrowd"]:
            if name == "dog":
                dogs.add(row["image_id"])
            elif name == "cat":
                cats.add(row["image_id"])
    for row in people["annotations"]:
        if not row["iscrowd"] and row["num_keypoints"] and min(row["bbox"][2:]) > 0:
            by_person[row["image_id"]].append(row)
    calibration_pool = []
    for iid, annotations in by_person.items():
        if len(annotations) != 1:
            continue
        ann = annotations[0]
        if ann["num_keypoints"] >= 15 and sum(v == 2 for v in ann["keypoints"][2::3]) >= 13 and ann["bbox"][3] >= 180:
            calibration_pool.append(iid)
    calibration = stable_take(calibration_pool, "calibration", cal_n)
    pose_strata = {
        "one_labeled_person": stable_take(
            (i for i, a in by_person.items() if len(a) == 1 and i not in calibration), "pose-single", pose_n
        ),
        "multiple_labeled_people": stable_take(
            (i for i, a in by_person.items() if len(a) > 1), "pose-multi", pose_n
        ),
        "no_person_annotations": stable_take(ids - any_person, "pose-negative", pose_n),
    }
    animal_strata = {
        "cat_not_dog": stable_take(cats - dogs, "cat", animal_n),
        "dog_not_cat": stable_take(dogs - cats, "dog", animal_n),
        "no_noncrowd_cat_or_dog": stable_take(ids - (cats | dogs), "animal-negative", animal_n),
    }
    return {
        "calibration": calibration, "pose_strata": pose_strata, "animal_strata": animal_strata,
        "pose": sorted(i for group in pose_strata.values() for i in group),
        "animal": sorted(i for group in animal_strata.values() for i in group),
    }


def make_stage(root, cal_n, pose_n, animal_n, full, no_tar=False):
    from PIL import Image, ImageOps
    from coco_eval_infer import ANNOTATIONS, PINNED, ROOT, digest, read_json

    sys.path.insert(0, str(root / PINNED / "tools"))
    from prepare_nbg_inputs import prepare_rgb

    annotation_paths = [root / "data/annotations" / name for name in ANNOTATIONS]
    instances, people = (read_json(path) for path in annotation_paths)
    by_iid = {im["id"]: im for im in instances["images"]}
    subsets = partition(instances, people, cal_n, pose_n, animal_n)
    if full:
        subsets["pilot_strata_used_for_calibration_only"] = True
        subsets["pose"] = sorted(set(by_iid) - set(subsets["calibration"]))
        subsets["animal"] = sorted(by_iid)
    stage = root / ("sdk-full" if full else "sdk-pilot")
    stage.mkdir(exist_ok=True)
    (stage / "models").mkdir(exist_ok=True)
    (stage / "inputs").mkdir(exist_ok=True)
    geometry = {}
    jobs = []
    same_human = ROOT / "nab_graphs/1m2_slots/slot01.nb"
    if digest(same_human) != MODELS["human"][4]:
        raise ValueError("both camera human graph hashes must match")
    for model, (rel, w, h, _, sha) in MODELS.items():
        model_src = ROOT / "nab_graphs" / rel
        if digest(model_src) != sha:
            raise ValueError(f"unexpected source graph {model_src}")
        (stage / "models" / f"{model}.nb").write_bytes(model_src.read_bytes())
        ids = subsets["calibration"] + subsets["pose"] if model == "human" else subsets["animal"]
        for iid in ids:
            key = f"{w}x{h}-{iid:012d}"
            raw = stage / "inputs" / f"{key}.raw"
            if key not in geometry:
                item = by_iid[iid]
                with Image.open(root / "data/val2017" / item["file_name"]) as src:
                    pic = ImageOps.exif_transpose(src).convert("RGB")
                if pic.size != (item["width"], item["height"]):
                    raise ValueError(f"COCO dimension mismatch: {item['file_name']}")
                planar, geom = prepare_rgb(pic, w, h, "letterbox", 128, "rgb", "chw")
                sha_input = hashlib.sha256(planar).hexdigest()
                if not raw.is_file() or digest(raw) != sha_input:
                    raw.write_bytes(planar)
                geometry[key] = dict(geom, sha256=sha_input)
            jobs.append([model, iid, key])
    manifest = {
        "models": MODELS, "graph_sha256_match_human": True, "subsets": subsets, "jobs": jobs,
        "geometry": geometry, "annotations": {p.name: digest(p) for p in annotation_paths},
        "release_archive_sha256": digest(root / "sony-release.tar"),
    }
    (stage / "job.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if no_tar:
        print(f"{stage}: {len(jobs)} planned runs, {len(geometry)} local inputs; no bulk archive", flush=True)
        return
    tar_path = stage.with_suffix(".tar")
    with tarfile.open(tar_path, "w") as archive:
        for fn in SDK_SOURCES:
            archive.add(root / PINNED / "tools" / fn, arcname=fn)
        archive.add(Path(__file__).resolve(), arcname="coco_eval_simulator.py")
        for name in ("job.json", "inputs", "models"):
            archive.add(stage / name, arcname=name)
    print(f"{tar_path}: {len(jobs)} runs, {len(geometry)} inputs; subsets: {subsets}", flush=True)


def vm_worker(sdk, workers):
    stage = Path.cwd()
    job = json.loads((stage / "job.json").read_text())
    out = stage / "out"
    out.mkdir(exist_ok=True)

    def run(item):
        model, iid, key = item
        prefix = out / f"{model}-{iid:012d}"
        stamp = prefix.with_suffix(".done.json")
        if stamp.is_file():
            return
        start = time.perf_counter()
        with prefix.with_suffix(".log").open("wb") as log:
            subprocess.run(["bash", str(stage / "run_sony_nbg_cpu.sh"), str(sdk),
                            str(stage / "build" / str(threading.get_ident())),
                            str(stage / "models" / f"{model}.nb"),
                            str(prefix), str(stage / "inputs" / f"{key}.raw")],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        for i, size in enumerate(SDK_OUTPUT_BYTES[model]):
            if prefix.with_name(f"{prefix.name}.{i}.raw").stat().st_size != size:
                raise ValueError(f"invalid tensor write {prefix}.{i}.raw")
        seconds = time.perf_counter() - start
        stamp.write_text(json.dumps({"model": model, "image_id": iid, "seconds": seconds}) + "\n")
        print(f"{prefix.name}: {seconds:.1f} s", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(run, job["jobs"]):
            pass


def job_files(item):
    model, iid, _ = item
    prefix = f"{model}-{iid:012d}"
    return [f"{prefix}.log", *(f"{prefix}.{i}.raw" for i in range(len(SDK_OUTPUT_BYTES[model]))),
            f"{prefix}.done.json"]


def valid_job(files, item):
    names = job_files(item)
    if not set(names) <= files.keys():
        return False
    stamp = json.loads(files[names[-1]])
    return (stamp["model"] == item[0] and stamp["image_id"] == item[1]
            and bool(files[names[0]])
            and all(len(files[name]) == size for name, size in zip(names[1:-1], SDK_OUTPUT_BYTES[item[0]])))


def job_done(out, item):
    names = job_files(item)
    if not (out / names[-1]).is_file() or not all((out / n).is_file() for n in names):
        return False
    return valid_job({n: (out / n).read_bytes() for n in names}, item)


def build_shard(root, stage, plan, items):
    from coco_eval_infer import PINNED, digest

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in SDK_SOURCES:
            archive.add(root / PINNED / "tools" / name, arcname=name)
        archive.add(Path(__file__).resolve(), arcname="coco_eval_simulator.py")
        for kind in sorted({i[0] for i in items}):
            path = stage / "models" / f"{kind}.nb"
            if digest(path) != plan["models"][kind][4]:
                raise ValueError(f"invalid graph {path}")
            archive.add(path, arcname=f"models/{kind}.nb")
        for key in sorted({i[2] for i in items}):
            path = stage / "inputs" / f"{key}.raw"
            if digest(path) != plan["geometry"][key]["sha256"]:
                raise ValueError(f"invalid input {path}")
            archive.add(path, arcname=f"inputs/{key}.raw")
        job = json.dumps({"jobs": items}).encode()
        entry = tarfile.TarInfo("job.json")
        entry.size = len(job)
        archive.addfile(entry, io.BytesIO(job))
    return buffer.getvalue()


def seed_pilot(root, stage, plan, items):
    pilot = root / "sdk-pilot"
    if not (pilot / "job.json").is_file() or pilot == stage:
        return 0
    saved = json.loads((pilot / "job.json").read_text())
    if (saved["annotations"], saved["release_archive_sha256"], saved["models"]) != (
        plan["annotations"], plan["release_archive_sha256"], plan["models"],
    ):
        return 0
    available = {(kind, iid, key) for kind, iid, key in saved["jobs"]}
    copied = 0
    for item in items:
        if tuple(item) not in available or saved["geometry"][item[2]] != plan["geometry"][item[2]]:
            continue
        names = job_files(item)
        if job_done(stage / "out", item) or not job_done(pilot / "out", item):
            continue
        for name in names:
            (stage / "out" / name).write_bytes((pilot / "out" / name).read_bytes())
        copied += 1
    return copied


def run_shards(root, stage_name, kinds, jobs_per_shard, max_shards, workers):
    if not 1 <= jobs_per_shard <= 128 or not 1 <= workers <= jobs_per_shard or max_shards < 1:
        raise ValueError("jobs per shard must be 1..128, with 1..jobs workers and at least one shard")
    stage = root / stage_name
    plan = json.loads((stage / "job.json").read_text())
    out = stage / "out"
    out.mkdir(exist_ok=True)
    items = [i for i in plan["jobs"] if i[0] in kinds]
    copied = seed_pilot(root, stage, plan, items)
    pending = [i for i in items if not job_done(out, i)]
    print(f"{len(items) - len(pending)}/{len(items)} complete ({copied} seeded from verified pilot)", flush=True)
    ssh = [str(SSH_EXE / "ssh.exe"), "-F", str(SSH_CONFIG), VM]
    for start in range(0, min(len(pending), jobs_per_shard * max_shards), jobs_per_shard):
        shard = pending[start:start + jobs_per_shard]
        expected = {name for item in shard for name in job_files(item)}
        proc = subprocess.run(ssh + [f"python3 -c {shlex.quote(VM_SHARD)} {workers}"],
                              input=build_shard(root, stage, plan, shard), stdout=subprocess.PIPE)
        files = {}
        if proc.stdout:
            with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as incoming:
                for item in incoming:
                    if not item.isfile() or item.name not in expected or item.name in files:
                        raise ValueError(f"unexpected shard output {item.name}")
                    files[item.name] = incoming.extractfile(item).read()
        done = 0
        for item in shard:
            if valid_job(files, item):
                for name in job_files(item):
                    (out / name).write_bytes(files[name])
                done += 1
        print(f"Persisted {done}/{len(shard)} jobs in {out}", flush=True)
        proc.check_returncode()
        if done != len(shard):
            raise ValueError("shard did not return every expected tensor")


def vm_run(root, name, workers, remote=None):
    stage = root / name
    ssh = [str(SSH_EXE / "ssh.exe"), "-F", str(SSH_CONFIG), VM]
    scp = [str(SSH_EXE / "scp.exe"), "-F", str(SSH_CONFIG)]
    if remote is None:
        remote = subprocess.check_output(ssh + ["cd ~/nbg; mktemp -d outputs/coco2017.XXXXXX"],
                                         text=True).strip()
        check_remote(remote)
        (stage / "remote.txt").write_text(remote + "\n", encoding="utf-8")
        subprocess.run(scp + [str(stage.with_suffix(".tar")), f"{VM}:nbg/{remote}/input.tar"], check=True)
        subprocess.run(ssh + [f"cd ~/nbg/{shlex.quote(remote)} && tar -xf input.tar"], check=True)
    check_remote(remote)
    print(f"VM directory: ~/nbg/{remote}", flush=True)
    proc = subprocess.run(ssh + [
        f'cd ~/nbg/{shlex.quote(remote)} && python3 coco_eval_simulator.py vm-worker "$HOME/nbg/x86_64_linux" {workers}'
    ])
    vm_fetch(stage, remote)
    proc.check_returncode()


def check_remote(remote):
    parent, _, suffix = remote.rpartition("/coco2017.")
    if parent != "outputs" or len(suffix) != 6 or not suffix.isalnum():
        raise ValueError(f"unexpected remote staging path: {remote}")


def vm_fetch(stage, remote, *, partial=False):
    check_remote(remote)
    ssh = [str(SSH_EXE / "ssh.exe"), "-F", str(SSH_CONFIG), VM]
    scp = [str(SSH_EXE / "scp.exe"), "-F", str(SSH_CONFIG)]
    archive_name = "results-partial.tar" if partial else "results.tar"
    if partial:
        marshall = ('for f in *.done.json; do b=${f%.done.json}; '
                    r"""printf '%s\0' "$f" "$b.log" "$b".*.raw; done""")
        command = (f"cd ~/nbg/{shlex.quote(remote)}/out && "
                   f"({marshall}) | tar -cf ../{archive_name} --null -T -")
    else:
        command = f"cd ~/nbg/{shlex.quote(remote)} && tar -cf {archive_name} -C out ."
    subprocess.run(ssh + [command], check=True)
    results = stage / ("received-partial.tar" if partial else "received.tar")
    subprocess.run(scp + [f"{VM}:nbg/{remote}/{archive_name}", str(results)], check=True)
    with tarfile.open(results) as archive:
        for member in archive:
            if not member.isfile():
                continue
            filename = Path(member.name)
            if len(filename.parts) != 1 or filename.is_absolute() or filename.name == "..":
                raise ValueError(f"unexpected VM output entry {member.name}")
            dst = stage / "out" / filename.name
            dst.parent.mkdir(exist_ok=True)
            dst.write_bytes(archive.extractfile(member).read())


def main():
    if sys.argv[1:2] == ["vm-worker"]:
        vm_worker(Path(sys.argv[2]), int(sys.argv[3]))
        return
    from coco_eval_infer import RUN

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, default=RUN)
    verbs = cli.add_subparsers(dest="action", required=True)
    p = verbs.add_parser("stage")
    p.add_argument("--calibration", type=int, default=8)
    p.add_argument("--pose-per-stratum", type=int, default=3)
    p.add_argument("--animal-per-stratum", type=int, default=3)
    p.add_argument("--full", action="store_true")
    p.add_argument("--no-tar", action="store_true")
    s = verbs.add_parser("run-shards")
    s.add_argument("--stage", choices=("sdk-pilot", "sdk-full"), default="sdk-full")
    s.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    s.add_argument("--jobs-per-shard", type=int, default=24)
    s.add_argument("--max-shards", type=int, default=1)
    s.add_argument("--workers", type=int, default=3)
    for action in ("run", "fetch"):
        s = verbs.add_parser(action)
        s.add_argument("--stage", default="sdk-pilot", choices=("sdk-pilot", "sdk-full"))
        if action == "run":
            s.add_argument("--workers", type=int, default=3)
            s.add_argument("--resume", action="store_true")
        else:
            s.add_argument("--partial", action="store_true")
    args = cli.parse_args()
    if args.action == "stage":
        make_stage(args.run, args.calibration, args.pose_per_stratum, args.animal_per_stratum,
                   args.full, args.no_tar)
    elif args.action == "run-shards":
        run_shards(args.run, args.stage, args.models, args.jobs_per_shard, args.max_shards, args.workers)
    elif args.action == "run":
        path = args.run / args.stage / "remote.txt"
        vm_run(args.run, args.stage, args.workers, path.read_text().strip() if args.resume else None)
    else:
        stage = args.run / args.stage
        vm_fetch(stage, (stage / "remote.txt").read_text().strip(), partial=args.partial)


if __name__ == "__main__":
    main()
