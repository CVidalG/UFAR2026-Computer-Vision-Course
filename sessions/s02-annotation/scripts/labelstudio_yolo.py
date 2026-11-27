# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Label Studio and a YOLO ML backend, downloaded, installed in this folder and started, in one command:

    uv run labelstudio_yolo.py path/to/images [--mode detect|segment|semantic|pose|classify|caption]

Only this file is needed (and uv). In its folder, the script:
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
Then: log in, correct the predictions, Submit; Settings > Model > Start Training (or --train-every N); the next
predictions come from the new model. Data in data/ (database, password, datasets, runs, models). Ctrl-C stops both
servers.
"""
import argparse
import atexit
import io
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
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
IMAGE_FILES = r"(?i).*\.(jpe?g|png|gif|bmp|webp)$"
HERE = os.path.dirname(os.path.abspath(__file__))
BIN, EXE = ("Scripts", ".exe") if os.name == "nt" else ("bin", "")


def run(command, **kwargs):
    # Each installation command is shown, then run
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def environment(name, packages, overrides=()):
    # A Python environment in envs/<name>, with these packages (overrides: versions imposed on their dependencies):
    # created and installed at the first run, completed when the packages change
    path = os.path.join(HERE, "envs", name)
    python = os.path.join(path, BIN, "python" + EXE)
    done = os.path.join(path, "installed.txt")
    wanted = "\n".join([*packages, *overrides])
    if os.path.exists(done):
        with open(done) as f:
            if f.read() == wanted:
                return path
    if not os.path.exists(python):
        run(["uv", "venv", "--python", PYTHON, path])
    command = ["uv", "pip", "install", "--python", python]
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
        except OSError as e:
            sys.exit(f"Could not download the ML backend framework ({e}): {ML_BACKEND_ZIP}")
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


def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port(port):
    while port_in_use(port):
        port += 1
    return port


def main():
    parser = argparse.ArgumentParser(description="Label Studio + YOLO ML backend, installed in this folder")
    parser.add_argument("images", help="folder of the images to annotate")
    parser.add_argument("--mode", choices=list(MODES), default="detect",
                        help="detect (boxes), segment (polygons), semantic (masks), pose (keypoints), classify, "
                             "caption (default: detect)")
    parser.add_argument("--model", help="model of the mode (default: " +
                                        ", ".join(f"{m}: {d[2]}" for m, d in MODES.items()) + ")")
    parser.add_argument("--labels", help="labels of the project, comma-separated (pose: the keypoints; default: the "
                                         "classes of the model)")
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
    args = parser.parse_args()

    images = os.path.abspath(args.images)
    template, _, default_model = MODES[args.mode]
    # A model file given by its path: absolute, since the backend runs in its own folder
    model = args.model or default_model
    model = os.path.abspath(model) if os.path.isfile(model) else model
    if not os.path.isdir(images):
        sys.exit(f"No folder {images}")
    if args.cold_start and args.mode in ("pose", "caption"):
        sys.exit("--cold-start: modes detect, segment, semantic and classify only")
    if shutil.which("uv") is None:
        sys.exit("uv is needed: https://docs.astral.sh/uv/getting-started/installation/")
    root = os.path.dirname(images)
    data = os.path.join(HERE, "data")
    model_dir = os.path.join(data, "ml-backend")
    os.makedirs(model_dir, exist_ok=True)

    # Settings of this installation (user, password, API token, ports): chosen at the first run, then kept
    settings_file = os.path.join(data, "settings.json")
    if os.path.exists(settings_file):
        with open(settings_file) as f:
            settings = json.load(f)
    else:
        settings = {"username": "annotator@example.com", "password": secrets.token_urlsafe(9),
                    "token": secrets.token_hex(20), "port": free_port(args.port), "ml_port": free_port(args.ml_port)}
        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=2)
    for port in (settings["port"], settings["ml_port"]):
        if port_in_use(port):
            sys.exit(f"Port {port} already in use (another Label Studio or ML backend?); ports are in {settings_file}")
    url = f"http://localhost:{settings['port']}"
    ml_url = f"http://localhost:{settings['ml_port']}"

    # 1. Label Studio, in its own environment
    label_studio = os.path.join(environment("label-studio", LABEL_STUDIO), BIN, "label-studio" + EXE)

    # 2. The ML backend framework, downloaded, then installed with Ultralytics (and what the mode needs) in a second
    # environment, with the label-studio-sdk of Label Studio
    repo = download_ml_backend()
    packages = ["-e", repo, ULTRALYTICS] + ([TRANSFORMERS] if args.mode == "caption" else []) + \
               ([CLIP] if args.cold_start else [])
    python_ml = os.path.join(environment("ml-backend", packages, [SDK]), BIN, "python" + EXE)

    # 3. The YOLO backend
    backend_dir = create_yolo_backend(python_ml, model_dir)
    with open(os.path.join(backend_dir, "model.py")) as f:
        if (args.mode != "detect" or args.cold_start) and "COLD_START" not in f.read():
            sys.exit("yolo_backend/model.py has no mode (older version): remove yolo_backend/ and run again")
    # Settings and fonts of Ultralytics in data/ml-backend, not in the user's folders
    ultralytics_dir = os.path.join(model_dir, "ultralytics")

    # Labels of the project: by default the classes of the model (pose: its keypoints, for objects of its classes;
    # the weights are downloaded into data/ml-backend)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()] if args.labels else None
    classes = ["object"] if labels else []
    if labels is None and args.mode != "caption":
        print(f"Classes of {model}...", flush=True)
        out = subprocess.run([python_ml, "-c", "import json, sys; from ultralytics import YOLO; m = YOLO(sys.argv[1]); "
                                               "print(json.dumps([list(m.names.values()), "
                                               "getattr(m.model, 'kpt_names', None)]))", model],
                             cwd=model_dir, env=dict(os.environ, YOLO_CONFIG_DIR=ultralytics_dir),
                             capture_output=True, text=True, check=True).stdout
        classes, keypoints = json.loads(out.strip().splitlines()[-1])
        if args.mode == "pose" and not keypoints:
            sys.exit(f"{model} has no keypoint names: give them with --labels")
        labels = list(keypoints.values())[0] if args.mode == "pose" else classes
    labels = labels or []

    # 4. Label Studio: its database in data/label-studio, the images served from their parent folder;
    # a backend on this machine allowed (ML_BLOCK_LOCAL_IP: its URL; SSRF_PROTECTION_ENABLED: its webhook);
    # the first request to the backend can take minutes (model downloaded: YOLOE, CLIP, BLIP): waited for
    ls_vars = dict(os.environ,
                   LABEL_STUDIO_BASE_DATA_DIR=os.path.join(data, "label-studio"),
                   LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true",
                   LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=root,
                   ML_BLOCK_LOCAL_IP="false",
                   SSRF_PROTECTION_ENABLED="false",
                   ML_TIMEOUT_SETUP="600",
                   ML_TIMEOUT_PREDICT="600",
                   # target storage (--save-annotations): one JSON per task, the task and its annotations
                   FUTURE_SAVE_TASK_TO_STORAGE="true")
    ls_log = open(os.path.join(data, "label-studio.log"), "a")
    servers = {}

    def stop():
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

    servers["Label Studio"] = subprocess.Popen(
        [label_studio, "start", "--no-browser", "--internal-host", "127.0.0.1", "--port", str(settings["port"]),
         "--username", settings["username"], "--password", settings["password"],
         "--user-token", settings["token"], "--enable-legacy-api-token"],
        env=ls_vars, stdout=ls_log, stderr=subprocess.STDOUT, start_new_session=True)

    # The ML backend of yolo_backend/ (its log in this terminal): reads the images from the same folder, calls Label
    # Studio with the token; its files (weights, datasets, runs, models) in data/ml-backend
    ml_vars = dict(os.environ,
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
                   # its log written at once, even when the output of this script goes to a file
                   PYTHONUNBUFFERED="1")
    servers["ML backend"] = subprocess.Popen(
        [python_ml, os.path.join(backend_dir, "_wsgi.py"), "--port", str(settings["ml_port"]), "--host", "127.0.0.1"],
        env=ml_vars, cwd=model_dir, start_new_session=True)

    def wait(health_url, name, timeout):
        start = time.time()
        while time.time() - start < timeout:
            for server, p in servers.items():
                if p.poll() is not None:
                    stop()
                    log = ls_log.name if server == "Label Studio" else "above, in this terminal"
                    sys.exit(f"{server} stopped; its log: {log}")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as r:
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
        with urllib.request.urlopen(request) as r:
            content = r.read()
        return json.loads(content) if content else None

    try:
        wait(url + "/health", "Label Studio", 300)
        wait(ml_url + "/health", "ML backend", 900)

        # 5. The project (first run): labeling configuration of the mode, images folder, backend, predictions shown
        title = args.title or os.path.basename(images) + ("" if args.mode == "detect" else f" ({args.mode})")
        project = next((p for p in api("GET", "/api/projects?page_size=1000")["results"] if p["title"] == title), None)
        if project is None:
            project = api("POST", "/api/projects", {"title": title,
                                                    "label_config": label_config(args.mode, labels, classes)})
        elif mode_of(project["label_config"]) != args.mode:
            sys.exit(f"The project {title!r} has another mode than {args.mode}: choose another --title")
        pid = project["id"]
        storages = api("GET", f"/api/storages/localfiles?project={pid}")
        if not storages:
            storages = [api("POST", "/api/storages/localfiles", {"project": pid, "title": os.path.basename(images),
                                                                 "path": images, "regex_filter": IMAGE_FILES,
                                                                 "use_blob_urls": True})]
        if not api("GET", f"/api/ml?project={pid}"):
            api("POST", "/api/ml", {"project": pid, "url": ml_url, "title": "YOLO"})
        # Predictions of the backend shown, and asked for, when a task opens; training every N annotations (0: never)
        api("PATCH", f"/api/projects/{pid}", {"show_collab_predictions": True, "model_version": "YOLO",
                                              "min_annotations_to_start_training": args.train_every})
        # New images of the folder: the storage is synced at each run
        for storage in storages:
            api("POST", f"/api/storages/localfiles/{storage['id']}/sync")
        saved = ""
        if args.save_annotations:
            # Target storage next to the images folder: each submitted task written as <task id>.json
            annotations = images.rstrip(os.sep) + "-annotations"
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

        print(f"""
Label Studio   {url}/projects/{pid}/data
Log in with    email {settings['username']}, password {settings['password']}
               (or Sign up: a new account)
Mode           {args.mode}: {template}
Images         {images}: {tasks} tasks (new images: Settings > Cloud Storage > Sync Storage, or restart){saved}
ML backend     {ml_url}: yolo_backend/model.py, model {model}{cold}; its log below
Train          Settings > Model > (...) > Start Training{every}:
               the next predictions come from the new model
Installed in   {HERE}: envs/, label-studio-ml-backend/, yolo_backend/, data/
Stop           Ctrl-C
""", flush=True)
        while all(p.poll() is None for p in servers.values()):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop()


if __name__ == "__main__":
    main()
