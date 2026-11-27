# Label Studio with a YOLO ML backend: end-to-end setup

Label Studio 1.23.2 and an ML backend with YOLO (Ultralytics 8.4.216): pre-annotation (`predict()`), retraining (`fit()`).

## Installation

- Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/) and git:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Copy [`labelstudio_yolo.py`](labelstudio_yolo.py) into an empty folder:

```bash
mkdir labelstudio && cd labelstudio
curl -LO https://raw.githubusercontent.com/CVidalG/UFAR2026-Computer-Vision-Course/main/sessions/s02-annotation/scripts/labelstudio_yolo.py
```

- At the first run, the script downloads and installs into this folder (internet access required):
  - `envs/label-studio/`: Label Studio 1.23.2 (with `sentry-sdk<3`);
  - `label-studio-ml-backend/`: the ML backend framework, from [GitHub](https://github.com/HumanSignal/label-studio-ml-backend) (commit `f3a2b7e`);
  - `envs/ml-backend/`: `label-studio-ml` (from `label-studio-ml-backend/`) and `ultralytics==8.4.216`;
  - `yolo_backend/`: created by `label-studio-ml create`, with the `model.py` of this repository (`yolo_backend/model.py`);
  - `data/ml-backend/yolo26n.pt`: the weights of the model (5.5 MB).
- Disk space: about 2 GB; each command is printed before it runs.

## Run

```bash
uv run labelstudio_yolo.py path/to/images
```

| Option | Default |
|---|---|
| `--model` | `yolo26n.pt` (any Ultralytics detection model: name or file) |
| `--labels` | the classes of the model; e.g. `--labels "person,horse"` (first run) |
| `--title` | the name of the images folder |
| `--epochs` | 10 (each training) |
| `--train-every` | 0 (no automatic training) |
| `--port`, `--ml-port` | 8080, 9090, or the next free ports (first run) |

- Open the URL printed by the script (`http://localhost:8080/projects/1/data`).
- Log in with the email and password printed by the script (`annotator@example.com`, password in `data/settings.json`), or *Sign up* to create an account.
- `data/`: `settings.json` (user, password, API token, ports), `label-studio/` (database), `label-studio.log`, `ml-backend/` (datasets, training runs, trained models).
- Same command again: same project; the new images of the folder are imported.
- Stop: Ctrl-C (both servers).

## Training

1. Open a task: the boxes of the model are shown.
2. Correct them (move, resize, delete, add), then *Submit*.
3. *Settings* > *Model* > `•••` > *Start Training*; or `--train-every N`: a training every N annotations.
4. Log of the backend, in the terminal: `Dataset: … images, … boxes, … labels`, the epochs, then `New model: …/data/ml-backend/models/project-1-<date>.pt`.
5. The next predictions come from the new model: open a task, or *Actions* > *Retrieve Predictions*.

- Images: only those of the folder (Local Files storage); images added with *Import* are not read by the backend.
- With `--labels`: the classes of the model that are not labels of the project are shown as boxes without label, and left out of the training; new labels (e.g. `DropCapital`) are learnt at the first training.
- Validation on the training images: the scores of the log are optimistic.
- To change the backend: edit `yolo_backend/model.py`, then run again.
