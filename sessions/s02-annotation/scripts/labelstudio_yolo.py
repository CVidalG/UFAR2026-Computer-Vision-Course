# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Label Studio and a YOLO ML backend, downloaded, installed in this folder and started, in one command:

    uv run labelstudio_yolo.py path/to/images [--mode detect|segment|semantic|pose|classify|caption]

Only this file is needed (and uv, 0.8.18 or later). In its folder, the script:
1. installs Label Studio in envs/label-studio (first run);
2. downloads the ML backend framework into label-studio-ml-backend/ (zip of a fixed commit, GitHub) and installs it,
   with Ultralytics, in envs/ml-backend (first run);
3. creates the YOLO backend, yolo_backend/: label-studio-ml create, then the model.py of the course, downloaded from
   its repository, or the one already in yolo_backend/ (first run; then kept: edit it and restart to change the
   backend);
4. starts Label Studio (images served from the parent folder of the images folder, local ML backend allowed) and the
   ML backend;
5. creates the project (first run): the labeling configuration of the mode (modeled on the template of Label Studio
   of the same name), the images folder as local storage, the backend connected, its predictions shown when a task
   opens; at each run, the new images are imported.
Then (the project opens in the browser, except with --no-browser): log in, correct the predictions, Submit;
Settings > Model > Start Training (or --train-every N); the next predictions come from the new model. Data in data/
(database, password, datasets, runs, models). Ctrl-C stops both servers and prints the number of annotated tasks.
With --dashboard (a trial), a summary page of the project: progress, annotations per label, metric over the
trainings, tasks to annotate first.
"""
import argparse
import atexit
import base64
import csv
import datetime
import html
import http.client
import http.server
import io
import json
import mimetypes
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
import zipfile
from xml.sax.saxutils import quoteattr

# sentry-sdk 3 makes Label Studio 1.23.2 crash at start
LABEL_STUDIO = ["label-studio==1.23.2", "sentry-sdk<3"]
# The ML backend framework, at a fixed commit (zip archive: no git needed)
ML_BACKEND_COMMIT = "f3a2b7e0cb689bdd875be716c0cda19d3cb79dce"
ML_BACKEND_ZIP = f"https://github.com/HumanSignal/label-studio-ml-backend/archive/{ML_BACKEND_COMMIT}.zip"
# label-studio-ml-backend asks for label-studio-sdk from GitHub, without a version: a fixed release of PyPI instead
# (not 2.1.2, the one of Label Studio 1.23.2: its OpenCV 4.12 needs numpy < 2.3, which Ultralytics 8.4.216 excludes
# on macOS)
SDK = "label-studio-sdk==2.1.6"
# OpenCV 5.1 has macOS wheels for macOS 15 or later only (elsewhere, a long build from source): 5.0, with wheels for
# macOS 13 or later (Intel: 14), Linux and Windows (not Windows on Arm)
OPENCV = ["opencv-python==5.0.0.93", "opencv-python-headless==5.0.0.93"]
ULTRALYTICS = "ultralytics==8.4.216"
# --cold-start: CLIP (text encoder of YOLOE; classify), fork of Ultralytics at a fixed commit
CLIP = "clip @ https://github.com/ultralytics/CLIP/archive/7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40.zip"
# --mode caption: BLIP
TRANSFORMERS = "transformers==5.23.0"
# The YOLO backend of the course (predict(), fit() and train() for the retraining)
MODEL_PY_URL = ("https://raw.githubusercontent.com/CVidalG/UFAR2026-Computer-Vision-Course/main/"
                "sessions/s02-annotation/scripts/yolo_backend/model.py")
# Mode: template of Label Studio, control tag, default model
MODES = {
    "detect": ("Object Detection with Bounding Boxes", "RectangleLabels", "yolo26n.pt"),
    "segment": ("Semantic Segmentation with Polygons", "PolygonLabels", "yolo26n-seg.pt"),
    "semantic": ("Semantic Segmentation with Masks", "BrushLabels", "yolo26n-sem.pt"),
    "pose": ("Keypoint Labeling", "KeyPointLabels", "yolo26n-pose.pt"),
    "classify": ("Image Classification", "Choices", "yolo26n-cls.pt"),
    "caption": ("Image Captioning", "TextArea", "Salesforce/blip-image-captioning-base"),
}
PYTHON = "3.12"
# uv: --managed-python (0.6.8); --torch-backend with triton (added in 0.8.13, removed by mistake in 0.8.17, back in
# 0.8.18)
UV_MIN = (0, 8, 18)
IMAGE_FILES = r"(?i).*\.(jpe?g|png|gif|bmp|webp)$"
# Data Manager: the tab of the tasks not yet annotated, those the model is least sure of first (semantic, caption:
# predictions without score, not ordered); the default tab, with the columns Label Studio gives it
TAB = "To annotate"
UNANNOTATED = {"filter": "filter:tasks:total_annotations", "operator": "equal", "type": "Number", "value": 0}
SCORED = ("detect", "segment", "pose", "classify")
HIDDEN = ["inner_id", "annotations_results", "annotations_ids", "predictions_results", "file_upload",
          "storage_filename", "created_at", "updated_at", "updated_by", "avg_lead_time", "draft_exists"]
DEFAULT_TAB = {"title": "Default", "type": "list", "target": "tasks", "hiddenColumns": {
    "explore": [f"tasks:{c}" for c in HIDDEN + ["predictions_score", "predictions_model_versions"]],
    "labeling": [f"tasks:{c}" for c in HIDDEN + ["id", "completed_at", "cancelled_annotations", "total_predictions",
                                                 "annotators", "predictions_score", "predictions_model_versions"]]}}


def nvidia():
    # An NVIDIA GPU (Linux, Windows): its driver's nvidia-smi
    return sys.platform != "darwin" and shutil.which("nvidia-smi") is not None


def torch_backend():
    # PyTorch for this machine: with CUDA for an NVIDIA GPU (uv: the version of CUDA of the driver), otherwise for the
    # CPU (Linux, Windows: not auto, which takes XPU for an Intel GPU, integrated ones included, or ROCm for an AMD
    # one); macOS: the PyTorch of PyPI
    if sys.platform == "darwin":
        return []
    return ["--torch-backend", "auto" if nvidia() else "cpu"]


def first_run_needs():
    # The first run, mode detect (measured 2026-11-28: the wheels, packed and unpacked; on macOS, this folder): what is
    # downloaded and the disk it takes, the free disk space needed; with an NVIDIA GPU (Linux, Windows), PyTorch with
    # CUDA
    if nvidia():
        return "up to 3.5 GB to download and 7 GB of disk (PyTorch with CUDA, for the NVIDIA GPU)", 8
    return "about 0.6 to 0.8 GB to download, about 2 to 2.5 GB of disk", 4


def machine_address():
    # The address of this machine on its network: that of the interface used to go out (nothing is sent)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
        except OSError:
            return socket.gethostname()


def instructions(mode):
    # Instructions shown in Label Studio before labeling (projects without instructions); caption: no training
    order = (", those the model is least sure of first (<i>Actions</i> &gt; <i>Retrieve Predictions</i> updates the "
             "scores)") if mode in SCORED else ""
    text = ("<p>Correct the predictions of the model, then <b>Submit</b>.</p>"
            f"<p>Tab <i>{TAB}</i>: the tasks not yet annotated{order}; then <i>Label All Tasks</i> &gt; "
            "<i>Label Tasks As Displayed</i>.</p>")
    if mode != "caption":
        text += ("<p>Retraining: <i>Settings</i> &gt; <i>Model</i> &gt; (...) &gt; <i>Start Training</i>; the next "
                 "predictions come from the new model.</p>")
    return text


HERE = os.path.dirname(os.path.abspath(__file__))
BIN, EXE = ("Scripts", ".exe") if os.name == "nt" else ("bin", "")


def run(command, **kwargs):
    # Each installation command is shown, then run
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def environment(name, packages, overrides=(), torch=False):
    # A Python environment in envs/<name>, with these packages (overrides: versions imposed on their dependencies;
    # torch: PyTorch for this machine, see torch_backend()):
    # created and installed at the first run, completed when the packages change
    path = os.path.join(HERE, "envs", name)
    python = os.path.join(path, BIN, "python" + EXE)
    done = os.path.join(path, "installed.txt")
    options = torch_backend() if torch else []
    wanted = "\n".join([*packages, *overrides, *options])
    if os.path.exists(done):
        with open(done) as f:
            if f.read() == wanted:
                return path
    if uv_version() < UV_MIN:
        sys.exit(f"uv {'.'.join(map(str, UV_MIN))} or later is needed to install the environments: uv self update (or "
                 "update uv the way it was installed)")
    if not os.path.exists(python):
        # The Python of uv (downloaded once, for this processor), not one of conda, Homebrew...
        run(["uv", "venv", "--managed-python", "--python", PYTHON, path])
    command = ["uv", "pip", "install", "--python", python, *options]
    if overrides:
        file = os.path.join(HERE, "envs", name + "-overrides.txt")
        with open(file, "w") as f:
            f.write("\n".join(overrides) + "\n")
        command += ["--overrides", file]
    run(command + list(packages))
    with open(done, "w") as f:
        f.write(wanted)
    return path


def download_ml_backend():
    # The ML backend framework (label_studio_ml, 26 example backends), at a fixed commit: downloaded at the first run
    repo = os.path.join(HERE, "label-studio-ml-backend")
    if not os.path.isdir(os.path.join(repo, "label_studio_ml")):
        shutil.rmtree(repo, ignore_errors=True)
        print("$ download", ML_BACKEND_ZIP, flush=True)
        try:
            with urllib.request.urlopen(ML_BACKEND_ZIP, timeout=300) as r:
                archive = zipfile.ZipFile(io.BytesIO(r.read()))
        except (OSError, http.client.HTTPException, zipfile.BadZipFile) as e:
            sys.exit(f"Could not download the ML backend framework ({e}): {ML_BACKEND_ZIP}; run the same command "
                     "again")
        # The archive holds one folder, label-studio-ml-backend-<commit>
        top = os.path.join(HERE, archive.namelist()[0].split("/")[0])
        shutil.rmtree(top, ignore_errors=True)
        archive.extractall(HERE)
        os.rename(top, repo)
    return repo


def create_yolo_backend(python_ml, model_dir):
    # The YOLO backend: the folder of label-studio-ml create, with the model.py of the course (first run)
    backend_dir = os.path.join(HERE, "yolo_backend")
    model_py = os.path.join(backend_dir, "model.py")
    if os.path.isfile(model_py) and os.path.isfile(os.path.join(backend_dir, "_wsgi.py")):
        return backend_dir
    if os.path.isfile(model_py):
        # yolo_backend/model.py alone (e.g. in a clone of the course repository): kept
        with open(model_py, "rb") as f:
            code = f.read()
    elif os.path.exists(backend_dir):
        sys.exit(f"{backend_dir} is incomplete (no model.py): remove it and run again")
    else:
        print("$ download", MODEL_PY_URL, flush=True)
        try:
            with urllib.request.urlopen(MODEL_PY_URL, timeout=60) as r:
                code = r.read()
        except OSError as e:
            sys.exit(f"Could not download the YOLO backend ({e}): {MODEL_PY_URL}")
    # (MODEL_DIR: the cache that label_studio_ml opens at import goes to data/ml-backend, not into this folder;
    # --force: yolo_backend/ created again, its model.py written back below)
    run([os.path.join(os.path.dirname(python_ml), "label-studio-ml" + EXE), "create", "yolo_backend", "--force"],
        cwd=HERE, env=dict(os.environ, MODEL_DIR=model_dir))
    # model.py of the template replaced by the one of the course
    with open(model_py, "wb") as f:
        f.write(code)
    print("  -> yolo_backend/model.py", flush=True)
    return backend_dir


def mode_of(config):
    # Mode of a labeling configuration, from its control tags (RectangleLabels alone: detect)
    return next((mode for mode, (_, tag, _) in MODES.items() if mode != "detect" and f"<{tag} " in config), "detect")


def label_config(mode, labels, classes):
    # Labeling configuration of the mode, same names in all modes: image, label (the labels; pose: the classes of the
    # objects), keypoint (pose: the keypoints), caption
    def tags(values, tag="Label"):
        return "".join(f"    <{tag} value={quoteattr(value)}/>\n" for value in values)

    # Boxes without rotation: the YOLO labels have none
    boxes = '  <RectangleLabels name="label" toName="image" canRotate="false">\n'

    control = {
        "detect": f'{boxes}{tags(labels)}  </RectangleLabels>\n',
        "segment": f'  <PolygonLabels name="label" toName="image">\n{tags(labels)}  </PolygonLabels>\n',
        "semantic": f'  <BrushLabels name="label" toName="image">\n{tags(labels)}  </BrushLabels>\n',
        "pose": (f'{boxes}{tags(classes)}  </RectangleLabels>\n'
                 f'  <KeyPointLabels name="keypoint" toName="image">\n{tags(labels)}  </KeyPointLabels>\n'),
        "classify": f'  <Choices name="label" toName="image" choice="single">\n{tags(labels, "Choice")}  </Choices>\n',
        "caption": '  <TextArea name="caption" toName="image" rows="3" maxSubmissions="1"/>\n',
    }[mode]
    return f'<View>\n  <Image name="image" value="$image"/>\n{control}</View>'


def intel_mac():
    # macOS on an Intel processor (hw.optional.arm64 is 1 on Apple silicon, even for a Python running under Rosetta)
    if sys.platform != "darwin":
        return False
    check = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True)
    return check.stdout.strip() != "1"


def images_in(folder):
    # Images directly in the folder (those imported), and, when there is none, those of its subfolders (first level)
    def count(path):
        return sum(1 for name in os.listdir(path) if re.match(IMAGE_FILES, name))

    top = count(folder)
    below = 0 if top else sum(count(e.path) for e in os.scandir(folder) if e.is_dir() and os.access(e.path, os.R_OK))
    return top, below


def uv_version():
    # Version of uv, e.g. (0, 12, 23)
    found = re.search(r"(\d+)\.(\d+)\.(\d+)", subprocess.run(["uv", "--version"], capture_output=True,
                                                             text=True).stdout)
    return tuple(int(n) for n in found.groups()) if found else (0, 0, 0)


def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port(port):
    while port_in_use(port):
        port += 1
    return port


# --- Dashboard (--dashboard, a trial): a summary page served by this script on 127.0.0.1 (progress, annotations per
# label, metric of the trainings, tasks to annotate first), read from Label Studio (its API: the token stays in this
# script) and from the trainings of Ultralytics (data/ml-backend/runs), when the page is loaded
DASHBOARD_PORT = 8282
# Main metric of each mode: its column in results.csv (Ultralytics)
METRICS = {"detect": "metrics/mAP50(B)", "segment": "metrics/mAP50(M)", "pose": "metrics/mAP50(P)",
           "classify": "metrics/accuracy_top1", "semantic": "metrics/mIoU"}
# Fitness of an epoch, by which Ultralytics 8.4.216 keeps its best model (best.pt): the sum of these columns
# (utils/metrics.py: detect, mAP50-95 of the boxes; segment and pose, plus that of the masks or keypoints; classify,
# top-1 and top-5; semantic, mIoU)
FITNESS = {"detect": ["metrics/mAP50-95(B)"], "segment": ["metrics/mAP50-95(M)", "metrics/mAP50-95(B)"],
           "pose": ["metrics/mAP50-95(P)", "metrics/mAP50-95(B)"],
           "classify": ["metrics/accuracy_top1", "metrics/accuracy_top5"], "semantic": ["metrics/mIoU"]}
# Figures of a training (Ultralytics): curves, confusion matrix, ground truth and predictions on validation images
FIGURES = ["results.png", "confusion_matrix_normalized.png", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]
# Region types of Label Studio that carry labels (TextArea: a caption, no label)
LABEL_TYPES = ("rectanglelabels", "polygonlabels", "brushlabels", "keypointlabels", "choices")
STYLE = """
:root{--text:#1f2937;--muted:#6b7280;--line:#e5e7eb;--bg:#f6f7f9;--card:#fff;--accent:#2563eb}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:28px 20px 48px}
header{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px 24px;margin-bottom:20px}
h1{font-size:22px;margin:0}h2{font-size:15px;margin:0 0 4px}.muted{color:var(--muted)}.small{font-size:13px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.button{display:inline-block;padding:7px 14px;border:1px solid var(--line);border-radius:8px;background:var(--card)}
.button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.actions{display:flex;gap:8px}.days{display:flex;align-items:flex-end;gap:6px;height:90px;margin-top:12px}
.days div{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;height:100%}
.days div{font-size:11px;max-width:56px}
.days span.b{width:100%;background:var(--accent);border-radius:3px 3px 0 0;min-height:2px}
.days span.d{color:var(--muted);margin-top:4px}.figures{display:flex;gap:28px;flex-wrap:wrap;margin-top:12px}
.figures .value{font-size:20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:14px;margin-bottom:14px}
.card,.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.label{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.value{font-size:26px;font-weight:600;margin-top:4px;overflow-wrap:anywhere}.value.long{font-size:16px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin-top:10px}
.bar>div{height:100%;background:var(--accent)}
.rows{margin-top:12px}.row{display:grid;grid-template-columns:minmax(80px,180px) 1fr 48px;gap:10px;align-items:center}
.row{margin:7px 0;font-size:14px}.row .bar{margin:0}.num{text-align:right;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
th,td{text-align:left;padding:6px 4px;border-top:1px solid var(--line)}th{color:var(--muted);font-weight:500}
.tasks{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-top:12px}
.task{display:block;border:1px solid var(--line);border-radius:10px;overflow:hidden;color:var(--text)}
.task:hover{border-color:var(--accent);text-decoration:none}
.task img{width:100%;height:110px;object-fit:cover;display:block;background:var(--bg)}
.task div{padding:8px 10px;font-size:13px;display:flex;justify-content:space-between}
details summary{cursor:pointer;font-weight:600}
details img{max-width:100%;margin-top:12px;border:1px solid var(--line);border-radius:8px}
"""


def run_name(pid):
    # Name of a training of the project (model.py: runs/project-<id>-<date>-<time>)
    return rf"project-{pid}-\d{{8}}-\d{{6}}"


def trainings(model_dir, pid, mode):
    # The trainings of the project, oldest first: epochs done and planned, the epoch of the model kept (best fitness)
    # and its main metric, finished (its model in models/) or not
    runs_dir = os.path.join(model_dir, "runs")
    names = sorted(n for n in (os.listdir(runs_dir) if os.path.isdir(runs_dir) else [])
                   if re.fullmatch(run_name(pid), n))

    def number(row, column):
        try:
            return float(row[column])
        except (KeyError, ValueError):  # (a column missing, a line being written)
            return None

    runs = []
    for name in names:
        run = os.path.join(runs_dir, name)
        rows, planned = [], 0
        if os.path.exists(os.path.join(run, "results.csv")):
            with open(os.path.join(run, "results.csv")) as f:
                rows = [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]
        if os.path.exists(os.path.join(run, "args.yaml")):
            with open(os.path.join(run, "args.yaml")) as f:
                found = re.search(r"^epochs: (\d+)", f.read(), re.M)
            planned = int(found.group(1)) if found else 0
        fitness = [sum(number(row, c) or 0 for c in FITNESS[mode]) for row in rows]
        best = fitness.index(max(fitness)) if rows else None
        runs.append({"name": name, "epochs": len(rows), "planned": planned,
                     "best": None if best is None else best + 1,
                     "metric": None if best is None else number(rows[best], METRICS[mode]),
                     "done": os.path.exists(os.path.join(model_dir, "models", name + ".pt"))})
    return runs


def dataset_size(model_dir, pid, mode):
    # The last dataset of the project (datasets/project-<id>, written again at each training), in words: its training
    # images, and its validation images, or the training images when no task is kept for validation (model.py:
    # val: images/train; classify: val/, a copy of train/)
    base = os.path.join(model_dir, "datasets", f"project-{pid}")
    parts = ["train", "val"] if mode == "classify" else [os.path.join("images", p) for p in ("train", "val")]
    train, val = [{os.path.relpath(os.path.join(folder, name), os.path.join(base, part))
                   for folder, _, names in os.walk(os.path.join(base, part)) for name in names} for part in parts]
    if not train:
        return "no dataset yet"

    def images(n):
        return f"{n} image" + ("" if n == 1 else "s")

    return f"training {images(len(train))}, validation " + ("on the same images" if not val or val == train else
                                                            images(len(val)))


def annotations_of(api, pid):
    # The annotations not skipped of the project, read with the annotated tasks (Data Manager, by pages): not the
    # summary of the project, whose created_labels undercounts in Label Studio 1.23.2; not an export, which writes files
    annotated = {"filter": "filter:tasks:total_annotations", "operator": "greater", "type": "Number", "value": 0}
    found, page, seen = [], 1, 0
    while True:
        batch = tasks_query(api, pid, [annotated], size=100, page=page)
        found += [a for t in batch["tasks"] for a in t.get("annotations") or [] if not a.get("was_cancelled")]
        seen += len(batch["tasks"])
        if not batch["tasks"] or seen >= batch.get("total", 0):
            return found
        page += 1


def label_counts(annotations, project):
    # Annotations per label, for each label of the labeling configuration (0 included), most annotated first: the
    # regions (classify: the choices)
    labels = [label for control in (project.get("parsed_label_config") or {}).values()
              for label in control.get("labels", [])]
    counts = dict.fromkeys(labels, 0)
    for annotation in annotations:
        for region in annotation.get("result") or []:
            if region.get("type") in LABEL_TYPES:
                for label in region.get("value", {}).get(region["type"], []):
                    counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda item: -item[1])


def duration(seconds):
    # 42 -> "42 s", 75 -> "1 min 15 s", 5400 -> "1 h 30 min"
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def pace_panel(annotations, remaining):
    # Pace: the annotations of each active day (the last 14, local dates), the median time per annotation (the time
    # spent in the editor, lead_time, measured by Label Studio) over the last 20 and overall, and the time left for the
    # tasks not yet annotated at the recent median
    days = {}
    for annotation in annotations:
        try:
            when = datetime.datetime.fromisoformat((annotation.get("created_at") or "").replace("Z", "+00:00"))
        except ValueError:
            continue
        day = when.astimezone().date()
        days[day] = days.get(day, 0) + 1
    shown = sorted(days.items())[-14:]
    top = max([n for _, n in shown] or [1])
    bars = "".join(f"<div><span>{n}</span><span class='b' style='height:{70 * n / top:.0f}%'></span>"
                   f"<span class='d'>{day:%d/%m}</span></div>" for day, n in shown)
    timed = [a["lead_time"] for a in sorted(annotations, key=lambda a: a.get("created_at") or "")
             if (a.get("lead_time") or 0) > 0]
    if timed:
        recent, overall = statistics.median(timed[-20:]), statistics.median(timed)
        figures = (f"<div><div class='label'>Median time per task</div><div class='value'>{duration(recent)}</div>"
                   f"<div class='muted small'>last {min(len(timed), 20)}; all {len(timed)}: {duration(overall)}</div>"
                   f"</div><div><div class='label'>Time left</div><div class='value'>"
                   f"{'about ' + duration(recent * remaining) if remaining else 'none'}</div><div class='muted small'>"
                   f"{remaining} task(s) not yet annotated, at the recent median</div></div>")
    else:
        figures = "<p class='muted small'>No time measured yet (annotations made in the editor of Label Studio).</p>"
    return (f"<div class='panel' style='margin-bottom:14px'><h2>Pace</h2><div class='muted small'>Annotations per day "
            f"(the last 14 active days) and time spent in the editor, as measured by Label Studio</div>"
            f"{'<div class=days>' + bars + '</div>' if bars else ''}<div class='figures'>{figures}</div></div>")


def chart(values):
    # The main metric of each training (0 to 1): one point per training, numbered from 1
    if not any(v is not None for v in values):
        return "<p class='muted small'>No training yet.</p>"
    w, h, left, top, bottom = 520, 190, 34, 26, 24
    step = (w - left - 60) / max(len(values) - 1, 1)

    def y(v):
        return top + (1 - v) * (h - bottom - top)

    grid = "".join(f"<line x1='{left}' x2='{w - 8}' y1='{y(v):.0f}' y2='{y(v):.0f}' stroke='#e5e7eb'/>"
                   f"<text x='{left - 6}' y='{y(v) + 4:.0f}' text-anchor='end' fill='#6b7280'>{v:g}</text>"
                   for v in (0, 0.5, 1))
    points = [(left + 24 + i * step, y(v), v, i + 1) for i, v in enumerate(values) if v is not None]
    line = " ".join(f"{x:.0f},{py:.0f}" for x, py, _, _ in points)
    dots = "".join(f"<circle cx='{x:.0f}' cy='{py:.0f}' r='4' fill='#2563eb'/>"
                   f"<text x='{x:.0f}' y='{py - 9:.0f}' text-anchor='middle'>{v:.3f}</text>"
                   f"<text x='{x:.0f}' y='{h - 6}' text-anchor='middle' fill='#6b7280'>{n}</text>"
                   for x, py, v, n in points)
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' font-size='11' fill='#1f2937'>{grid}"
            f"<polyline points='{line}' fill='none' stroke='#2563eb' stroke-width='2'/>{dots}</svg>")


def tasks_query(api, pid, items, ordering=(), size=12, page=1):
    # Tasks of the project through the Data Manager, with their annotations (filters, ordering; the score: that of
    # the current model)
    query = urllib.parse.quote(json.dumps({"filters": {"conjunction": "and", "items": items},
                                           "ordering": list(ordering)}))
    return api("GET", f"/api/tasks?project={pid}&fields=all&page_size={size}&page={page}&query={query}")


def dashboard_page(d, ls):
    # The page; d: what main() knows (api, project, mode, paths); ls: Label Studio, at the address of this page
    api, pid, esc, mode = d["api"], d["pid"], html.escape, d["mode"]
    project = api("GET", f"/api/projects/{pid}")
    version = (api("GET", f"/api/ml?project={pid}") or [{}])[0].get("model_version") or "?"
    total, done = project["task_number"], project["num_tasks_with_annotations"]
    # Tasks with a prediction of the current model: a score (semantic, caption: 0.0)
    current = tasks_query(api, pid, [{"filter": "filter:tasks:predictions_score", "operator": "greater_or_equal",
                                      "type": "Number", "value": 0}], size=1).get("total", 0)
    runs = [] if mode == "caption" else trainings(d["model_dir"], pid, mode)
    annotations = annotations_of(api, pid)

    def card(label, value, note, ratio=None, long=False):
        bar = "" if ratio is None else f"<div class='bar'><div style='width:{100 * ratio:.0f}%'></div></div>"
        return (f"<div class='card'><div class='label'>{label}</div><div class='value{' long' if long else ''}'>"
                f"{value}</div><div class='muted small'>{note}</div>{bar}</div>")

    metric = METRICS.get(mode, "").split("/")[-1]
    finished = [r for r in runs if r["done"] and r["metric"] is not None]
    model_note = (f"{metric} {finished[-1]['metric']:.3f} (validation); {len(finished)} training(s)" if finished
                  else "no training yet" if mode != "caption" else "predictions only")
    page = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>{esc(project['title'])}: dashboard</title><style>{STYLE}</style></head><body><main>",
            f"<header><div><h1>{esc(project['title'])}</h1><div class='muted small'>Mode {mode} &middot; updated at "
            f"{time.strftime('%H:%M:%S')}</div></div><div class='actions'><a class='button primary' "
            f"href='{ls}/projects/{pid}/data' target='_blank'>Open Label Studio</a><a class='button' href='/'>Refresh"
            f"</a></div></header>",
            "<div class='cards'>",
            card("Annotated tasks", f"{done} / {total}", f"{100 * done / total:.0f} %" if total else "",
                 done / total if total else 0),
            card("Current model", esc(version), model_note, long=True),
            card("Predictions of the current model", f"{current} / {total}",
                 "tasks (Actions &gt; Retrieve Predictions, in Label Studio)", current / total if total else 0),
            "</div><div class='grid'>"]
    counts = label_counts(annotations, project) if mode != "caption" else []
    if counts:
        shown = [(label, n) for label, n in counts if n] or counts[:10]
        hidden = len(counts) - len(shown)
        top = max(n for _, n in counts) or 1
        rows = "".join(f"<div class='row'><span>{esc(label)}</span><div class='bar'><div style='width:"
                       f"{100 * n / top:.0f}%'></div></div><span class='num'>{n}</span></div>" for label, n in shown)
        rest = f"<p class='muted small'>{hidden} other label(s): no annotation yet</p>" if hidden else ""
        page.append(f"<div class='panel'><h2>Annotations per label</h2><div class='muted small'>"
                    f"{sum(n for _, n in counts)} {'images' if mode == 'classify' else 'regions'}</div>"
                    f"<div class='rows'>{rows}</div>{rest}</div>")
    if mode != "caption":
        lines = "".join(f"<tr><td>{i}</td><td>{r['name']}</td><td>{r['epochs']} / {r['planned']}</td>"
                        f"<td>{r['best'] or ''}</td><td>{'' if r['metric'] is None else format(r['metric'], '.3f')}"
                        f"</td><td class='muted'>{'' if r['done'] else 'training...'}</td></tr>"
                        for i, r in enumerate(runs, 1))
        table = (f"<table><tr><th>#</th><th>Model</th><th>Epochs</th><th>Model kept: epoch</th><th>{metric}</th>"
                 f"<th></th></tr>{lines}</table>") if runs else ""
        page.append(f"<div class='panel'><h2>{metric} over the trainings</h2><div class='muted small'>On the "
                    f"validation images, for the model kept by each training; last dataset: "
                    f"{dataset_size(d['model_dir'], pid, mode)}</div>"
                    f"{chart([r['metric'] if r['done'] else None for r in runs])}{table}</div>")
    page += ["</div>", pace_panel(annotations, total - done)]
    items = []
    unannotated = tasks_query(api, pid, [UNANNOTATED], ["tasks:predictions_score"] if mode in SCORED else [])
    for t in unannotated["tasks"]:
        image = urllib.parse.quote(str(t["data"].get("image", "")).split("?d=", 1)[-1], safe="")
        score = t.get("predictions_score") if mode in SCORED else None
        items.append(f"<a class='task' href='{ls}/projects/{pid}/data?task={t['id']}' target='_blank'>"
                     f"<img src='/image?d={image}' alt=''><div><span>Task {t['id']}</span>"
                     f"<span class='muted'>{'' if score is None else format(score, '.2f')}</span></div></a>")
    order = ", lowest prediction score first" if mode in SCORED else " (no prediction score in this mode)"
    page.append(f"<div class='panel'><h2>To annotate first</h2><div class='muted small'>Not yet annotated{order} "
                f"(tab <i>{TAB}</i> in Label Studio)</div><div class='tasks'>{''.join(items)}</div>"
                f"{'' if items else '<p class=muted>No task left to annotate.</p>'}</div>")
    if runs:
        figures = "".join(f"<img src='/figure?run={runs[-1]['name']}&f={f}' alt='{f}' title='{f}'>" for f in FIGURES
                          if os.path.exists(os.path.join(d["model_dir"], "runs", runs[-1]["name"], f)))
        if figures:
            page.append(f"<div class='panel' style='margin-top:14px'><details><summary>Figures of the last training "
                        f"({runs[-1]['name']}, Ultralytics)</summary>{figures}</details></div>")
    page.append("</main></body></html>")
    return "".join(page)


def serve_dashboard(d, host, port, key):
    # The dashboard server, in a thread: the page, the images of the folder of the project, the figures of its
    # trainings. Listens on host (127.0.0.1: this machine only); the key of the address printed by the launcher is
    # asked for, then kept in a cookie (another account of a shared server, a web page: no key); on 127.0.0.1, a Host
    # of this machine only (DNS rebinding); any port (an SSH tunnel)
    cookie = f"dashboard_{port}"
    local = host in ("127.0.0.1", "localhost", "::1")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, kind="text/html; charset=utf-8", code=200, headers=()):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            for header in headers:
                self.send_header(*header)
            self.end_headers()
            self.wfile.write(body)

        def file(self, path, base, allowed):
            # A file inside base (links followed), whose name is allowed
            try:
                path, base = os.path.realpath(path), os.path.realpath(base)
                inside = os.path.commonpath([path, base]) == base
            except ValueError:  # (a NUL character, another drive)
                inside = False
            if not inside or not allowed(os.path.basename(path)) or not os.path.isfile(path):
                return self.send(b"Not found", "text/plain", 404)
            with open(path, "rb") as f:
                self.send(f.read(), mimetypes.guess_type(path)[0] or "application/octet-stream")

        def do_GET(self):
            hostname = re.sub(r":\d+$", "", self.headers.get("Host", ""))
            if local and hostname not in ("localhost", "127.0.0.1", "[::1]"):
                return self.send(b"Forbidden", "text/plain", 403)
            path, _, query = self.path.partition("?")
            q = {k: v[0] for k, v in urllib.parse.parse_qs(query).items()}
            # (keys compared as bytes: any text accepted, a key with accents included)
            if secrets.compare_digest(q.get("key", "").encode(), key.encode()):
                # The address printed by the launcher: the key kept in a cookie, the page without it
                return self.send(b"", headers=[("Set-Cookie", f"{cookie}={key}; Path=/; HttpOnly; SameSite=Strict"),
                                               ("Location", "/")], code=303)
            jar = dict(c.strip().split("=", 1) for c in self.headers.get("Cookie", "").split(";") if "=" in c)
            if not secrets.compare_digest(jar.get(cookie, "").encode(), key.encode()):
                return self.send(b"Open the address of the dashboard printed by labelstudio_yolo.py (with its key)",
                                 "text/plain; charset=utf-8", 403)
            try:
                if path == "/":
                    self.send(dashboard_page(d, f"http://{html.escape(hostname)}:{d['ls_port']}").encode())
                elif path == "/image":
                    # (the path of an image as Label Studio has it: relative to the parent folder, quoted)
                    self.file(os.path.join(d["root"], urllib.parse.unquote(q.get("d", ""))), d["images"],
                              lambda name: re.match(IMAGE_FILES, name))
                elif path == "/figure" and re.fullmatch(run_name(d["pid"]), q.get("run", "")):
                    run = os.path.join(d["model_dir"], "runs", q["run"])
                    self.file(os.path.join(run, q.get("f", "")), run, lambda name: name in FIGURES)
                else:
                    self.send(b"Not found", "text/plain", 404)
            except OSError as e:
                self.send(f"Label Studio did not answer: {e}".encode(), "text/plain", 502)

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    parser = argparse.ArgumentParser(description="Label Studio + YOLO ML backend, installed in this folder")
    parser.add_argument("images", help="folder of the images to annotate")
    parser.add_argument("--mode", choices=list(MODES), default="detect",
                        help="detect (boxes), segment (polygons), semantic (masks), pose (keypoints), classify, "
                             "caption (default: detect)")
    parser.add_argument("--model", help="model of the mode (default: " +
                                        ", ".join(f"{m}: {d[2]}" for m, d in MODES.items()) + ")")
    parser.add_argument("--labels", help="labels of the project, comma-separated (pose: the keypoints; default: the "
                                         "classes of the model; classify with a pretrained model: needed)")
    parser.add_argument("--cold-start", action="store_true",
                        help="until the first training, predict the labels from their names (YOLOE; classify: CLIP)")
    parser.add_argument("--save-annotations", action="store_true",
                        help="write each submitted task as JSON into <images>-annotations/")
    parser.add_argument("--title", help="title of the project (default: name of the images folder, and the mode)")
    parser.add_argument("--epochs", type=int, default=10, help="epochs of each training (default: 10)")
    parser.add_argument("--train-every", type=int, default=0,
                        help="retrain after every N annotations (default: 0, only with Start Training)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Label Studio port, first run (default: 8080 or next free)")
    parser.add_argument("--ml-port", type=int, default=9090,
                        help="ML backend port, first run (default: 9090 or next free)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the project (or the dashboard) in the browser (opened on a desktop only)")
    parser.add_argument("--dashboard", action="store_true",
                        help="trial: a summary page of the project (progress, annotations per label, metric over the "
                             "trainings, pace, tasks to annotate first), port 8282 or next free")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "0.0.0.0"],
                        help="where Label Studio and the dashboard listen: 127.0.0.1 (default), this machine only; "
                             "0.0.0.0, all its addresses (e.g. a server reached at its IP)")
    args = parser.parse_args()
    if os.name == "nt":
        # The output of this script in UTF-8 (paths in any script: accents, Armenian...), even redirected to a file;
        # Ctrl-Break as Ctrl-C (the test of GitHub Actions stops the launcher this way)
        for stream in (sys.stdout, sys.stderr):
            stream.reconfigure(encoding="utf-8", errors="replace")
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    images = os.path.abspath(args.images)
    template, _, default_model = MODES[args.mode]
    # A model file given by its path: absolute, since the backend runs in its own folder
    model = args.model or default_model
    model = os.path.abspath(model) if os.path.isfile(model) else model
    if not os.path.isdir(images):
        sys.exit(f"No folder {images}")
    top, below = images_in(images)
    if not top:
        sys.exit(f"No image (jpg, png, gif, bmp, webp) in {images}" +
                 (f"; {below} in its subfolders, which are not imported: give a subfolder" if below else ""))
    if args.labels and args.mode == "caption":
        sys.exit("--labels: not used with --mode caption (a caption, no label)")
    if args.cold_start and args.mode in ("pose", "caption"):
        sys.exit("--cold-start: modes detect, segment, semantic and classify only")
    if intel_mac():
        sys.exit("Intel Macs are not supported: their last PyTorch (2.2.2) predates NumPy 2, which OpenCV 5 needs")
    if os.name == "nt" and platform.machine().upper() == "ARM64":
        sys.exit("Windows on Arm is not supported: PyTorch has no wheel for it on PyPI")
    if shutil.which("uv") is None:
        sys.exit("uv is needed: https://docs.astral.sh/uv/getting-started/installation/")
    root = os.path.dirname(images)
    data = os.path.join(HERE, "data")
    model_dir = os.path.join(data, "ml-backend")
    # --mode classify with a pretrained model (the default, or another one to download): its classes are the 1000 of
    # ImageNet, not labels for a project; without --labels, a new project is refused (no database yet: at once)
    imagenet = args.mode == "classify" and not args.labels and not os.path.isfile(model)
    classify_labels = ('--mode classify: give the classes of the project, e.g. --labels "recto,verso" (those of a '
                       'pretrained model are the 1000 classes of ImageNet)')
    if imagenet and not os.path.exists(os.path.join(data, "label-studio", "label_studio.sqlite3")):
        sys.exit(classify_labels)
    os.makedirs(model_dir, exist_ok=True)
    # data/ (database, password, token) for this account only (a shared server); the images folder: unchanged
    if os.name == "posix":
        os.chmod(data, 0o700)

    # Settings of this installation (user, password, API token, ports, account of the ML backend): chosen at the
    # first run, then kept; written for this account only
    settings_file = os.path.join(data, "settings.json")
    settings = {}
    if os.path.exists(settings_file):
        with open(settings_file) as f:
            settings = json.load(f)
    new = {"username": "annotator@example.com", "password": secrets.token_urlsafe(9), "token": secrets.token_hex(20),
           "port": settings.get("port") or free_port(args.port), "ml_port": settings.get("ml_port") or
           free_port(args.ml_port), "ml_user": "label-studio", "ml_password": secrets.token_urlsafe(12)}
    if any(k not in settings for k in new):
        settings = {**new, **settings}
        with open(os.open(settings_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600),
                  "w") as f:
            json.dump(settings, f, indent=2)
    if os.name == "posix":
        os.chmod(settings_file, 0o600)
    for port in (settings["port"], settings["ml_port"]):
        if port_in_use(port):
            sys.exit(f"Port {port} already in use (another Label Studio or ML backend?); ports are in {settings_file}")
    url = f"http://localhost:{settings['port']}"
    ml_url = f"http://localhost:{settings['ml_port']}"
    # The ML backend asks Label Studio, and this script, for its account (Basic authentication)
    ml_auth = {"Authorization": "Basic " + base64.b64encode(
        f"{settings['ml_user']}:{settings['ml_password']}".encode()).decode()}
    # Label Studio at the address of this machine on its network (--host 0.0.0.0, or an address of the machine)
    local = args.host == "127.0.0.1"
    address = "localhost" if local else machine_address()

    # First run: what it takes, if the disk has room for it
    first_run, disk_gb = first_run_needs()
    if not os.path.exists(os.path.join(HERE, "envs", "ml-backend", "installed.txt")):
        free = shutil.disk_usage(HERE).free / 1e9
        if free < disk_gb:
            sys.exit(f"First run: {first_run}; {free:.1f} GB free on this disk ({disk_gb} GB needed)")
        print(f"First run: {first_run}", flush=True)

    # Steps 1 to 3, installation (first run): stopped (network, Ctrl-C), the next run takes it up where it stopped
    stopped = "Installation stopped: run the same command again (what is installed is kept)"
    try:
        # 1. Label Studio, in its own environment
        print("[1/5] Label Studio: envs/label-studio", flush=True)
        label_studio = os.path.join(environment("label-studio", LABEL_STUDIO), BIN, "label-studio" + EXE)

        # 2. The ML backend framework, downloaded, then installed with Ultralytics (and what the mode needs) in a
        # second environment, with the label-studio-sdk of Label Studio
        print("[2/5] ML backend: label-studio-ml-backend/, envs/ml-backend", flush=True)
        repo = download_ml_backend()
        packages = ["-e", repo, ULTRALYTICS] + ([TRANSFORMERS] if args.mode == "caption" else []) + \
                   ([CLIP] if args.cold_start else [])
        python_ml = os.path.join(environment("ml-backend", packages, [SDK, *OPENCV], torch=True), BIN, "python" + EXE)

        # 3. The YOLO backend
        print("[3/5] YOLO backend: yolo_backend/", flush=True)
        backend_dir = create_yolo_backend(python_ml, model_dir)
        with open(os.path.join(backend_dir, "model.py")) as f:
            if (args.mode != "detect" or args.cold_start) and "COLD_START" not in f.read():
                sys.exit("yolo_backend/model.py has no mode (older version): remove yolo_backend/ and run again")
        # Settings and fonts of Ultralytics in data/ml-backend, not in the user's folders
        ultralytics_dir = os.path.join(model_dir, "ultralytics")

        # Labels of the project: by default the classes of the model (pose: its keypoints, for objects of its classes;
        # the weights are downloaded into data/ml-backend); classify with a pretrained model: --labels
        labels = [label.strip() for label in args.labels.split(",") if label.strip()] if args.labels else None
        classes = ["object"] if labels else []
        if labels is None and args.mode != "caption" and not imagenet:
            print(f"Classes of {model}...", flush=True)
            try:
                script = ("import json, sys; from ultralytics import YOLO; m = YOLO(sys.argv[1]); "
                          "print(json.dumps([list(m.names.values()), getattr(m.model, 'kpt_names', None)]))")
                out = subprocess.run([python_ml, "-c", script, model], cwd=model_dir,
                                     env=dict(os.environ, YOLO_CONFIG_DIR=ultralytics_dir, PYTHONUTF8="1"),
                                     capture_output=True, encoding="utf-8", errors="replace", check=True).stdout
            except subprocess.CalledProcessError as e:
                # (e.g. a misspelt --model; its error, captured: the last line)
                sys.exit(f"Could not read the classes of {model}: " +
                         ((e.stderr or "").strip().splitlines() or ["no error message"])[-1])
            classes, keypoints = json.loads(out.strip().splitlines()[-1])
            if args.mode == "pose" and not keypoints:
                sys.exit(f"{model} has no keypoint names: give them with --labels")
            labels = list(keypoints.values())[0] if args.mode == "pose" else classes
        labels = labels or []
    except subprocess.CalledProcessError:
        # (the error of the installation command: above)
        sys.exit("Installation stopped (error above): if it was the network, run the same command again (what is "
                 "installed is kept)")
    except KeyboardInterrupt:
        sys.exit("\n" + stopped)

    # 4. Label Studio: its database in data/label-studio, the images served from their parent folder;
    # a backend on this machine allowed (ML_BLOCK_LOCAL_IP: its URL; SSRF_PROTECTION_ENABLED: its webhook);
    # the first request to the backend can take minutes (model downloaded: YOLOE, CLIP, BLIP): waited for
    ls_vars = dict(os.environ,
                   # UTF-8 for its output and files (Windows: otherwise the code page of the console, cp1252...)
                   PYTHONUTF8="1",
                   LABEL_STUDIO_BASE_DATA_DIR=os.path.join(data, "label-studio"),
                   LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true",
                   LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=root,
                   ML_BLOCK_LOCAL_IP="false",
                   SSRF_PROTECTION_ENABLED="false",
                   ML_TIMEOUT_SETUP="600",
                   ML_TIMEOUT_PREDICT="600",
                   # target storage (--save-annotations): one JSON per task, the task and its annotations
                   FUTURE_SAVE_TASK_TO_STORAGE="true",
                   # new accounts with the invite link only (printed below; Organization > Add Members): otherwise
                   # anyone reaching it (another account of a server, the network with --host 0.0.0.0) could sign up
                   # and see the projects
                   LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK="true",
                   # its account, in its environment (readable by this account only), not on its command line (that
                   # of every process can be read by all the accounts of the machine)
                   LABEL_STUDIO_USERNAME=settings["username"],
                   LABEL_STUDIO_PASSWORD=settings["password"],
                   LABEL_STUDIO_USER_TOKEN=settings["token"],
                   # prediction score of a task (Data Manager: Order by Prediction score, the tab below): that of the
                   # predictions of the current model of the backend; without it (flag off by default), that of the
                   # predictions of version project.model_version, "YOLO" here (the backend, for the predictions asked
                   # for when a task opens), which none has: no score
                   ff_front_dev_1682_model_version_dropdown_070622_short="true")
    ls_log = open(os.path.join(data, "label-studio.log"), "a")
    servers = {}
    # The servers in their own process group: Ctrl-C reaches this script only, which then reads the summary from
    # Label Studio and stops them
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    dashboard = []

    def stop():
        # (a second Ctrl-C does not interrupt the stop)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for server in dashboard:
            server.shutdown()
            server.server_close()
        dashboard.clear()
        for p in servers.values():
            if p.poll() is None:
                if os.name == "posix":
                    os.killpg(p.pid, signal.SIGTERM)
                else:
                    p.terminate()
        for p in servers.values():
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()

    # Ctrl-C, end of the script, terminal closed or process killed: both servers are stopped too
    atexit.register(stop)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda *_: sys.exit(0))

    print("[4/5] Servers: Label Studio (its log: data/label-studio.log), ML backend (its log below)", flush=True)
    servers["Label Studio"] = subprocess.Popen(
        [label_studio, "start", "--no-browser", "--internal-host", args.host, "--port", str(settings["port"]),
         "--enable-legacy-api-token"],
        env=ls_vars, stdout=ls_log, stderr=subprocess.STDOUT, **group)

    # The ML backend of yolo_backend/ (its log in this terminal): reads the images from the same folder, calls Label
    # Studio with the token; its files (weights, datasets, runs, models) in data/ml-backend
    ml_vars = dict(os.environ,
                   PYTHONUTF8="1",
                   LABEL_STUDIO_URL=url,
                   LABEL_STUDIO_API_KEY=settings["token"],
                   LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=root,
                   MODE=args.mode,
                   YOLO_MODEL=model,
                   COLD_START="1" if args.cold_start else "",
                   MODEL_DIR=model_dir,
                   HF_HOME=os.path.join(model_dir, "huggingface"),
                   YOLO_CONFIG_DIR=ultralytics_dir,
                   TRAIN_EPOCHS=str(args.epochs),
                   # its account (Label Studio and this script use it: another account of the machine cannot)
                   BASIC_AUTH_USER=settings["ml_user"],
                   BASIC_AUTH_PASS=settings["ml_password"],
                   # its log written at once, even when the output of this script goes to a file
                   PYTHONUNBUFFERED="1")
    servers["ML backend"] = subprocess.Popen(
        [python_ml, os.path.join(backend_dir, "_wsgi.py"), "--port", str(settings["ml_port"]), "--host", "127.0.0.1"],
        env=ml_vars, cwd=model_dir, **group)

    def stopped_server():
        # A server that stopped, and where its log is
        for server, p in servers.items():
            if p.poll() is not None:
                return f"{server} stopped; its log: " + (ls_log.name if server == "Label Studio" else
                                                          "above, in this terminal")

    def wait(health_url, name, timeout, headers=None):
        start = time.time()
        while time.time() - start < timeout:
            if message := stopped_server():
                stop()
                sys.exit(message)
            try:
                with urllib.request.urlopen(urllib.request.Request(health_url, headers=headers or {}), timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(1)
        stop()
        sys.exit(f"{name} did not start in {timeout} s")

    def api(method, path, body=None):
        request = urllib.request.Request(url + path, method=method,
                                         data=None if body is None else json.dumps(body).encode(),
                                         headers={"Authorization": "Token " + settings["token"],
                                                  "Content-Type": "application/json"})
        # (the longest requests: the connection of the backend and the retrieval of predictions, up to 600 s)
        with urllib.request.urlopen(request, timeout=900) as r:
            content = r.read()
        return json.loads(content) if content else None

    pid = None
    annotations = images.rstrip(os.sep) + "-annotations"
    try:
        wait(url + "/health", "Label Studio", 300)
        wait(ml_url + "/health", "ML backend", 900, ml_auth)

        # 5. The project (first run): labeling configuration of the mode, images folder, backend, predictions shown
        print("[5/5] Project", flush=True)
        title = args.title or os.path.basename(images) + ("" if args.mode == "detect" else f" ({args.mode})")
        project = next((p for p in api("GET", "/api/projects?page_size=1000")["results"] if p["title"] == title), None)
        if project is None:
            if imagenet:
                sys.exit(classify_labels)
            project = api("POST", "/api/projects", {"title": title,
                                                    "label_config": label_config(args.mode, labels, classes)})
        elif mode_of(project["label_config"]) != args.mode:
            sys.exit(f"The project {title!r} has another mode than {args.mode}: choose another --title")
        elif args.labels:
            # --labels other than those of the project (pose: its keypoints): not changed here (annotations could
            # lose their label); a new project, or the labels changed in Label Studio
            control = "keypoint" if args.mode == "pose" else "label"
            current = api("GET", f"/api/projects/{project['id']}")["parsed_label_config"].get(control, {})
            if set(current.get("labels", [])) != set(labels):
                sys.exit(f"The project {title!r} has other labels than --labels: run again with --title \"<name>\" "
                         "for a new project with these labels (same images), or change its labels in Label Studio "
                         "(Settings > Labeling Interface) and run again without --labels")
        pid = project["id"]
        if not project.get("expert_instruction"):
            # Instructions shown before labeling (projects without instructions), and a description in the list of
            # the projects (projects without description)
            fields = {"expert_instruction": instructions(args.mode), "show_instruction": True}
            if not project.get("description"):
                fields["description"] = f"{args.mode}: {template}; model {model}" + (
                    "; until the first training, labels predicted from their names" if args.cold_start else "")
            api("PATCH", f"/api/projects/{pid}", fields)
        storages = api("GET", f"/api/storages/localfiles?project={pid}")
        if not storages:
            storages = [api("POST", "/api/storages/localfiles", {"project": pid, "title": os.path.basename(images),
                                                                 "path": images, "regex_filter": IMAGE_FILES,
                                                                 "use_blob_urls": True})]
        # The ML backend, with its account (projects connected before: the account added)
        account = {"auth_method": "BASIC_AUTH", "basic_auth_user": settings["ml_user"],
                   "basic_auth_pass": settings["ml_password"]}
        backends = api("GET", f"/api/ml?project={pid}")
        if not backends:
            print("Connecting the ML backend to the project (first run: the model may be downloaded)...", flush=True)
            api("POST", "/api/ml", {"project": pid, "url": ml_url, "title": "YOLO", **account})
        for backend in backends:
            if backend.get("url", "").rstrip("/") == ml_url and (backend.get("auth_method") != "BASIC_AUTH" or
                                                                 backend.get("basic_auth_user") != settings["ml_user"]):
                # (this backend only; with its URL and project: Label Studio 1.23.2 checks the backend at each change,
                # /health and /setup, with what the request gives)
                api("PATCH", f"/api/ml/{backend['id']}", {"project": pid, "url": ml_url,
                                                          "title": backend.get("title") or "YOLO", **account})
        # The webhook Label Studio adds for the backend (every event, without its account: refused, 401) turned off;
        # trainings (Start Training, --train-every) go through the connection of the backend, with its account
        for hook in api("GET", f"/api/webhooks/?project={pid}"):
            if hook.get("url", "").rstrip("/") == ml_url + "/webhook" and hook.get("is_active"):
                api("PATCH", f"/api/webhooks/{hook['id']}/", {"is_active": False})
        # Predictions of the backend shown, and asked for, when a task opens; training every N annotations (0: never)
        api("PATCH", f"/api/projects/{pid}", {"show_collab_predictions": True, "model_version": "YOLO",
                                              "min_annotations_to_start_training": args.train_every})
        # New images of the folder: the storage is synced at each run
        for storage in storages:
            api("POST", f"/api/storages/localfiles/{storage['id']}/sync")
        # Tabs of the Data Manager (first run): the default one (all the tasks), then the tasks not yet annotated,
        # lowest prediction score first (semantic, caption: no score), with the score and the model version shown
        views = api("GET", f"/api/dm/views/?project={pid}")
        if not any(v.get("data", {}).get("title") == TAB for v in views):
            if not views:
                api("POST", "/api/dm/views/", {"project": pid, "data": DEFAULT_TAB})
            hidden = [f"tasks:{c}" for c in HIDDEN + ["completed_at", "cancelled_annotations", "annotators"]]
            api("POST", "/api/dm/views/", {"project": pid, "data": {
                "title": TAB, "type": "list", "target": "tasks",
                "ordering": ["tasks:predictions_score"] if args.mode in SCORED else [],
                "filters": {"conjunction": "and", "items": [UNANNOTATED]},
                "hiddenColumns": {"explore": hidden, "labeling": hidden + ["tasks:total_predictions",
                                                                           "tasks:predictions_model_versions"]}}})
        saved = ""
        if args.save_annotations:
            # Target storage next to the images folder: each submitted task written as <task id>.json
            os.makedirs(annotations, exist_ok=True)
            exports = api("GET", f"/api/storages/export/localfiles?project={pid}")
            if not exports:
                exports = [api("POST", "/api/storages/export/localfiles",
                               {"project": pid, "title": os.path.basename(annotations), "path": annotations})]
            for storage in exports:
                api("POST", f"/api/storages/export/localfiles/{storage['id']}/sync")
            saved = f"\nAnnotations    {annotations}: <task id>.json, written at each Submit"
        tasks = api("GET", f"/api/projects/{pid}")["task_number"]
        every = f", and every {args.train_every} annotation(s)" if args.train_every else ""
        cold = ", labels predicted from their names until the first training" if args.cold_start else ""
        public = f"http://{address}:{settings['port']}"
        board = ""
        if args.dashboard:
            # The dashboard (a trial), at the same address as Label Studio, with a key of this run
            port, key = free_port(DASHBOARD_PORT), secrets.token_urlsafe(16)
            dashboard.append(serve_dashboard({"api": api, "pid": pid, "mode": args.mode, "root": root,
                                              "images": images, "model_dir": model_dir, "ls_port": settings["port"]},
                                             args.host, port, key))
            board = f"\nDashboard      http://{address}:{port}/?key={key}"
        # New accounts: the invite link (Organization > Add Members gives it too)
        accounts = "new accounts: " + public + api("GET", "/api/invite")["invite_url"]

        print(f"""
Label Studio   {public}/projects/{pid}/data{board}
Log in with    email {settings['username']}, password {settings['password']}
               (also in data/settings.json; {accounts})
Mode           {args.mode}: {template}
Images         {images}: {tasks} tasks
               (new images: {public}/projects/{pid}/settings/storage > Sync Storage, or restart){saved}
ML backend     {ml_url}: yolo_backend/model.py, model {model}{cold}; its log below
Train          {public}/projects/{pid}/settings/ml > (...) > Start Training{every}:
               the next predictions come from the new model
Installed in   {HERE}: envs/, label-studio-ml-backend/, yolo_backend/, data/
Stop           Ctrl-C
""", flush=True)
        # The project (or the dashboard) in the browser, where there is a desktop (without one, webbrowser would start
        # a text browser in this terminal)
        if not args.no_browser and (os.name == "nt" or sys.platform == "darwin" or os.environ.get("DISPLAY")
                                    or os.environ.get("WAYLAND_DISPLAY")):
            webbrowser.open(board.split()[-1] if board else f"{public}/projects/{pid}/data")
        while all(p.poll() is None for p in servers.values()):
            time.sleep(1)
        print(stopped_server(), flush=True)
    except KeyboardInterrupt:
        # What is annotated, and where (the servers, in their own session, are still running)
        try:
            if pid is not None:
                p = api("GET", f"/api/projects/{pid}")
                where = annotations + "/" if args.save_annotations else "--save-annotations"
                print(f"\n{p['title']}: {p['num_tasks_with_annotations']} of {p['task_number']} tasks annotated, in "
                      f"data/label-studio/label_studio.sqlite3 (to get them: Export in the project, or {where}); "
                      "to go on: the same command", flush=True)
        except (OSError, KeyboardInterrupt):
            pass
    finally:
        stop()


if __name__ == "__main__":
    main()
