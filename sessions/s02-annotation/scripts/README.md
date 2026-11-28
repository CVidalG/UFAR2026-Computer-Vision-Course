# Label Studio with a YOLO ML backend: end-to-end setup

Label Studio 1.23.2 and an ML backend with YOLO (Ultralytics 8.4.216): pre-annotation (`predict()`), retraining (`fit()`), in six modes.

## Installation

- Requirement: [uv](https://docs.astral.sh/uv/getting-started/installation/), 0.8.13 or later to install the environments, checked by the script (it also installs Python):

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
  - `label-studio-ml-backend/`: the ML backend framework, from [GitHub](https://github.com/HumanSignal/label-studio-ml-backend) (zip archive of commit `f3a2b7e`);
  - `envs/ml-backend/`: `label-studio-ml` (from `label-studio-ml-backend/`), `ultralytics==8.4.216`, `label-studio-sdk==2.1.6` and OpenCV 5.0.0.93 (versions fixed in `envs/ml-backend-overrides.txt`; OpenCV 5.1 has no wheel before macOS 15); PyTorch for this machine (`uv pip install --torch-backend auto`: with CUDA for an NVIDIA GPU, otherwise for the CPU); with `--mode caption`, `transformers`; with `--cold-start`, `clip`;
  - `yolo_backend/`: created by `label-studio-ml create`, with the `model.py` of this repository (`yolo_backend/model.py`);
  - `data/ml-backend/`: the weights of the model of the mode (e.g. `yolo26n.pt`, 5.5 MiB); with `--cold-start`, `yoloe-26s-seg.pt` (30 MiB) and its text encoder `mobileclip2_b.ts` (242 MiB), or CLIP ViT-B/32 in `clip/` (338 MiB, `classify`); with `--mode caption`, BLIP base in `huggingface/` (945 MiB).
- First run (mode `detect`): about 0.6 to 0.8 GB downloaded and 2 to 2.5 GB of disk; with an NVIDIA GPU (Linux, Windows: PyTorch with CUDA), up to 3.5 GB and 7 GB; up to about 1.6 GB more with `--cold-start` and `--mode caption`. At the first run, the script stops when the disk of this folder has less than 4 GB free (with an NVIDIA GPU: 8 GB). Each installation command is printed before it runs. uv keeps its own cache, and its own Python 3.12 (`--managed-python`: not the one of conda or Homebrew), in its folders.
- Installation stopped (network, Ctrl-C): run the same command again; what is installed is kept.
- Systems: tested on macOS 14 (Apple silicon) and Ubuntu 22.04 with an NVIDIA GPU; checked at each change of the scripts on Ubuntu, Windows and macOS by [GitHub Actions](../../../.github/workflows/launcher.yml) ([`test_launcher.py`](test_launcher.py)); expected, not tested: macOS 13 (Apple silicon); not supported: Intel Macs (their last PyTorch, 2.2.2, predates NumPy 2, which OpenCV 5 needs) and Windows on Arm (no PyTorch wheel for it on PyPI).

## Run

```bash
uv run labelstudio_yolo.py path/to/images --mode detect
```

| `--mode` | Labeling configuration modeled on the template | Model | `--cold-start` |
|---|---|---|---|
| `detect` (default) | Object Detection with Bounding Boxes | `yolo26n.pt` | YOLOE |
| `segment` | Semantic Segmentation with Polygons (instances) | `yolo26n-seg.pt` | YOLOE |
| `semantic` | Semantic Segmentation with Masks | `yolo26n-sem.pt` | YOLOE |
| `pose` | Keypoint Labeling (with a box per object) | `yolo26n-pose.pt` | – |
| `classify` | Image Classification | `yolo26n-cls.pt` | CLIP |
| `caption` | Image Captioning | BLIP base (predictions only) | – |

| Option | Default |
|---|---|
| `--mode` | `detect` |
| `--model` | the model of the mode (any Ultralytics model of the same task: name or file) |
| `--labels` | the classes of the model (`pose`: its keypoints); e.g. `--labels "DropCapital,MarginaliaIllu,MainIllu"` (first run); needed with `--mode classify` and a pretrained model (its classes: the 1000 of ImageNet) |
| `--cold-start` | off; on: until the first training, the labels are predicted from their names (`DropCapital`: "drop capital") |
| `--save-annotations` | off; on: each submitted task is written as `<task id>.json` into `<images>-annotations/`, next to the images folder |
| `--title` | the name of the images folder, followed by the mode except in `detect` (e.g. `images (segment)`) |
| `--epochs` | 10 (each training) |
| `--train-every` | 0 (no automatic training) |
| `--port`, `--ml-port` | 8080, 9090, or the next free ports (first run) |
| `--no-browser` | off: the project (or the dashboard) opens in the browser, on a desktop |
| `--dashboard` | off; on (a trial): a summary page of the project, see [Dashboard](#dashboard-a-trial) |
| `--host` | `127.0.0.1`: Label Studio and the dashboard for this machine only; `0.0.0.0`: at the address of the machine on its network, see [On a server](#on-a-server) |

- The project opens in the browser, on a desktop (the URL printed by the script, `http://localhost:8080/projects/1/data`).
- Instructions shown before labeling (*Settings* > *Annotation* > *Labeling Instructions*; written in a project without instructions), and a tab of the *Data Manager*, *To annotate*: the tasks not yet annotated, lowest prediction score first (`semantic`, `caption`: not ordered, the predictions having no score). The score is that of the predictions of the current model: Label Studio is started with `ff_front_dev_1682_model_version_dropdown_070622_short=true` (without it, *Order by Prediction score* sorts nothing here).
- Images: those of the folder itself (`jpg`, `png`, `gif`, `bmp`, `webp`); its subfolders are not imported.
- Log in with the email and password printed by the script (`annotator@example.com`, password in `data/settings.json`), or *Sign up* to create an account.
- `data/` (for this account only): `settings.json` (user, password, API token, ports, account of the ML backend), `label-studio/` (database), `label-studio.log`, `ml-backend/` (weights, datasets, training runs, trained models).
- Annotations: in the database, `data/label-studio/label_studio.sqlite3`. To get them: *Export* in the project (JSON, CSV, COCO, Pascal VOC, YOLO, with or without the images), or `--save-annotations` (one `<task id>.json` per task: the task and its annotations, with annotator, dates, time spent).
- Same command again: same project (its title: the name of the images folder, and the mode); the new images of the folder are imported. With `--labels` other than those of the project, the script stops: `--title` for a new project with these labels on the same images, or change its labels in Label Studio (*Settings* > *Labeling Interface*).
- One mode per run: the backend refuses the projects of another mode (made by another run), with an error in its log; run the launcher again in their mode (`--mode`).
- Stop: Ctrl-C (both servers); the script prints the number of annotated tasks, and where they are.

## Training

1. Open a task: the predictions of the model are shown.
2. Correct them, then *Submit*.
3. *Settings* > *Model* (the URL printed by the script, `.../projects/1/settings/ml`) > `(...)` > *Start Training*; or `--train-every N`: a training every N annotations.
4. Log of the backend, in the terminal: `Dataset: …`, the epochs, then `New model: …/data/ml-backend/models/project-1-<date>.pt`.
5. The next predictions come from the new model: in the label stream (*Label All Tasks*, *Label Tasks As Displayed*), or for all the tasks with *Actions* > *Retrieve Predictions*, which also updates the scores of the tab. A task opened from the list keeps the predictions it already has.

- Validation: the tasks whose id is a multiple of 5, not used for training. Without such a task (or with only those), validation on the training images: the scores of the log are then optimistic.
- Images: only those of the folder (Local Files storage). Images added with *Import* are not supported: the backend cannot read them (an error in its log, no prediction, no training).
- Predictions: the labels of the project only; the other classes of the model are left out, as in the training. With `--labels`, new labels (e.g. `DropCapital`) are learnt at the first training, or predicted from their names with `--cold-start`.
- `--cold-start`: YOLOE (or CLIP) predicts until the first training; the training then retrains the model of the mode.
- `--mode semantic`: the pixels without brush are learnt as `background`; paint every class present in the image.
- `--mode pose`: a box per object (label `object` with `--labels`); a keypoint belongs to its box when Label Studio links them (`parentID`), otherwise to the smallest box that contains it; without a box, one object per image. Keypoints of Label Studio have no visibility flag: an annotated keypoint is visible (`v = 2`).
- `semantic`, `caption`: the predictions have no score (*Order by Prediction score* does not sort them; the tab *To annotate* keeps the order of the tasks).
- `--mode caption`: BLIP proposes a caption; no training.
- To change the backend: edit `yolo_backend/model.py`, then run again. To get the current one of this repository: remove `yolo_backend/`.

## Dashboard (a trial)

`--dashboard`: a summary page of the project, served by the script on port 8282 (or the next free port), at the address printed by the script (with a key of this run), opened in the browser instead of the project; *Open Label Studio*, *Refresh*.

- Annotated tasks, the current model (with the main metric of the last training), tasks with a prediction of the current model.
- Annotations per label: the regions of the annotations not skipped (`classify`: the images), for each label of the project, read with the annotated tasks.
- The main metric over the trainings of the project (`data/ml-backend/runs/`), on the validation images, for the model kept by each training (`best.pt`: the epoch of best fitness for Ultralytics): `detect`, mAP50 of the boxes; `segment`, of the masks; `pose`, of the keypoints; `classify`, top-1 accuracy; `semantic`, mIoU. As a line (finished trainings) and a table, with the last dataset.
- Pace: the annotations of each active day, the median time per task (time spent in the editor, measured by Label Studio) over the last 20 and overall, the time left for the tasks not yet annotated.
- To annotate first: the 12 tasks not yet annotated with the lowest prediction scores, as in the tab (image, score, link to the task in Label Studio).
- The figures of Ultralytics of the last training, folded: curves, confusion matrix, ground truth and predictions on validation images.

## On a server

- Label Studio, the ML backend and the dashboard listen on `127.0.0.1` (default): from another computer, through an SSH tunnel, e.g. `ssh -L 8080:localhost:8080 -L 8282:localhost:8282 server`, then the addresses printed by the script.
- `--host 0.0.0.0`: Label Studio and the dashboard at the address of the server on its network (printed by the script), for annotators on that network; new accounts with the invite link only (*Organization* > *Add Members*; the people invited see all the projects). The ML backend stays on `127.0.0.1`. No HTTPS: on a network you trust only.
- Other accounts of the server: `data/` is for your account only; the ML backend asks for its account (in `data/settings.json`, given to Label Studio by the script); the dashboard asks for the key of its address.
- uv keeps its cache in your home folder (`UV_CACHE_DIR` to put it elsewhere).
