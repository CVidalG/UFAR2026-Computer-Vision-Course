# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Label Studio and a YOLO ML backend, downloaded, installed in this folder and started, in one command:

    uv run labelstudio_yolo.py path/to/images

Only this file is needed. In its folder, the script:
1. installs Label Studio in envs/label-studio (first run);
2. downloads the ML backend framework into label-studio-ml-backend/ (GitHub) and installs it, with Ultralytics, in
   envs/ml-backend (first run);
3. creates the YOLO backend, yolo_backend/: label-studio-ml create, then the model.py of the course, downloaded from
   its repository, or the one already in yolo_backend/ (first run; then kept: edit it and restart to change the
   backend);
4. starts Label Studio (images served from the parent folder of the images folder, local ML backend allowed) and the
   ML backend;
5. creates the project (first run): the images folder as local storage, boxes with the classes of the model, the
   backend connected, its predictions shown when a task opens; at each run, the new images are imported.
Then: log in, correct the predictions, Submit; Settings > Model > Start Training (or --train-every N); the next
predictions come from the new model. Data in data/ (database, password, datasets, runs, models). Ctrl-C stops both
servers.
"""
import argparse
import atexit
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
from xml.sax.saxutils import quoteattr

# sentry-sdk 3 makes Label Studio 1.23.2 crash at start
LABEL_STUDIO = ["label-studio==1.23.2", "sentry-sdk<3"]
ML_BACKEND_REPO = "https://github.com/HumanSignal/label-studio-ml-backend.git"
ML_BACKEND_COMMIT = "f3a2b7e0cb689bdd875be716c0cda19d3cb79dce"
ULTRALYTICS = "ultralytics==8.4.216"
# The YOLO backend of the course (predict(), fit() and train() for the retraining)
MODEL_PY_URL = ("https://raw.githubusercontent.com/CVidalG/UFAR2026-Computer-Vision-Course/main/"
                "sessions/s02-annotation/scripts/yolo_backend/model.py")
PYTHON = "3.12"
IMAGE_FILES = r"(?i).*\.(jpe?g|png|gif|bmp|webp)$"
HERE = os.path.dirname(os.path.abspath(__file__))
BIN, EXE = ("Scripts", ".exe") if os.name == "nt" else ("bin", "")


def run(command, **kwargs):
    # Each installation command is shown, then run
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def environment(name, packages):
    # A Python environment in envs/<name>, with these packages: created and installed at the first run
    path = os.path.join(HERE, "envs", name)
    python = os.path.join(path, BIN, "python" + EXE)
    done = os.path.join(path, "installed.txt")
    if os.path.exists(done):
        with open(done) as f:
            if f.read() == "\n".join(packages):
                return path
    if not os.path.exists(python):
        run(["uv", "venv", "--python", PYTHON, path])
    run(["uv", "pip", "install", "--python", python, *packages])
    with open(done, "w") as f:
        f.write("\n".join(packages))
    return path


def download_ml_backend():
    # The ML backend framework (label_studio_ml, 26 example backends), at a fixed commit: downloaded at the first run
    repo = os.path.join(HERE, "label-studio-ml-backend")
    if not os.path.isdir(os.path.join(repo, "label_studio_ml")):
        shutil.rmtree(repo, ignore_errors=True)
        run(["git", "init", "--quiet", repo])
        run(["git", "-C", repo, "fetch", "--depth", "1", ML_BACKEND_REPO, ML_BACKEND_COMMIT])
        run(["git", "-C", repo, "checkout", "--quiet", "FETCH_HEAD"])
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
    parser.add_argument("--model", default="yolo26n.pt", help="Ultralytics detection model (default: yolo26n.pt)")
    parser.add_argument("--labels", help="labels of the project, comma-separated (default: the classes of the model)")
    parser.add_argument("--title", help="title of the project (default: name of the images folder)")
    parser.add_argument("--epochs", type=int, default=10, help="epochs of each training (default: 10)")
    parser.add_argument("--train-every", type=int, default=0,
                        help="retrain after every N annotations (default: 0, only with Start Training)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Label Studio port, first run (default: 8080 or next free)")
    parser.add_argument("--ml-port", type=int, default=9090,
                        help="ML backend port, first run (default: 9090 or next free)")
    args = parser.parse_args()

    images = os.path.abspath(args.images)
    # A model file given by its path: absolute, since the backend runs in its own folder
    model = os.path.abspath(args.model) if os.path.isfile(args.model) else args.model
    if not os.path.isdir(images):
        sys.exit(f"No folder {images}")
    for tool, help_url in (("uv", "https://docs.astral.sh/uv/getting-started/installation/"),
                           ("git", "https://git-scm.com/downloads")):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} is needed: {help_url}")
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

    # 2. The ML backend framework, downloaded, then installed with Ultralytics in a second environment
    repo = download_ml_backend()
    python_ml = os.path.join(environment("ml-backend", ["-e", repo, ULTRALYTICS]), BIN, "python" + EXE)

    # 3. The YOLO backend
    backend_dir = create_yolo_backend(python_ml, model_dir)

    # Labels of the project: by default the classes of the model (the weights are downloaded into data/ml-backend)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()] if args.labels else None
    if labels is None:
        print(f"Classes of {model}...", flush=True)
        out = subprocess.run([python_ml, "-c", "import json, sys; from ultralytics import YOLO; "
                                               "print(json.dumps(list(YOLO(sys.argv[1]).names.values())))", model],
                             cwd=model_dir, capture_output=True, text=True, check=True).stdout
        labels = json.loads(out.strip().splitlines()[-1])

    # 4. Label Studio: its database in data/label-studio, the images served from their parent folder;
    # a backend on this machine allowed (ML_BLOCK_LOCAL_IP: its URL; SSRF_PROTECTION_ENABLED: the webhook created with it)
    ls_vars = dict(os.environ,
                   LABEL_STUDIO_BASE_DATA_DIR=os.path.join(data, "label-studio"),
                   LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true",
                   LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=root,
                   ML_BLOCK_LOCAL_IP="false",
                   SSRF_PROTECTION_ENABLED="false")
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
                   YOLO_MODEL=model,
                   MODEL_DIR=model_dir,
                   TRAIN_EPOCHS=str(args.epochs))
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

        # 5. The project (first run): labeling configuration, images folder, backend, predictions shown
        title = args.title or os.path.basename(images)
        project = next((p for p in api("GET", "/api/projects?page_size=1000")["results"] if p["title"] == title), None)
        if project is None:
            config = ('<View>\n  <Image name="image" value="$image"/>\n  <RectangleLabels name="label" toName="image">\n'
                      + "".join(f"    <Label value={quoteattr(label)}/>\n" for label in labels)
                      + "  </RectangleLabels>\n</View>")
            project = api("POST", "/api/projects", {"title": title, "label_config": config})
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
        tasks = api("GET", f"/api/projects/{pid}")["task_number"]
        every = f", and every {args.train_every} annotation(s)" if args.train_every else ""

        print(f"""
Label Studio   {url}/projects/{pid}/data
Log in with    email {settings['username']}, password {settings['password']}
               (or Sign up: a new account)
Images         {images}: {tasks} tasks (new images: Settings > Cloud Storage > Sync Storage, or restart)
ML backend     {ml_url}: yolo_backend/model.py, model {model}; its log below
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
