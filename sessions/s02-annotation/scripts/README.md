# Label Studio with a YOLO ML backend: end-to-end setup

Label Studio 1.23.2 and an ML backend with YOLO (Ultralytics 8.4.216): pre-annotation (`predict()`), retraining (`fit()`), in six modes.

## Installation

- Requirement: [uv](https://docs.astral.sh/uv/getting-started/installation/) (it also installs Python):

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
  - `envs/ml-backend/`: `label-studio-ml` (from `label-studio-ml-backend/`), `ultralytics==8.4.216`, `label-studio-sdk==2.1.6` and OpenCV 5.0.0.93 (versions fixed in `envs/ml-backend-overrides.txt`; OpenCV 5.1 has no wheel before macOS 15); with `--mode caption`, `transformers`; with `--cold-start`, `clip`;
  - `yolo_backend/`: created by `label-studio-ml create`, with the `model.py` of this repository (`yolo_backend/model.py`);
  - `data/ml-backend/`: the weights of the model of the mode (e.g. `yolo26n.pt`, 5.5 MB); with `--cold-start`, `yoloe-26s-seg.pt` (30 MB) and its text encoder `mobileclip2_b.ts` (242 MB), or CLIP ViT-B/32 in `clip/` (338 MB, `classify`); with `--mode caption`, BLIP base in `huggingface/` (945 MB).
- Disk space: about 2 GB, up to 1.5 GB more with `--cold-start` and `--mode caption` (sizes in MiB); each installation command is printed before it runs. uv keeps its own cache, and the Python it installs if needed, in its folders.

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
| `--labels` | the classes of the model (`pose`: its keypoints); e.g. `--labels "DropCapital,MarginaliaIllu,MainIllu"` (first run) |
| `--cold-start` | off; on: until the first training, the labels are predicted from their names (`DropCapital`: "drop capital") |
| `--save-annotations` | off; on: each submitted task is written as `<task id>.json` into `<images>-annotations/`, next to the images folder |
| `--title` | the name of the images folder, followed by the mode except in `detect` (e.g. `images (segment)`) |
| `--epochs` | 10 (each training) |
| `--train-every` | 0 (no automatic training) |
| `--port`, `--ml-port` | 8080, 9090, or the next free ports (first run) |

- Open the URL printed by the script (`http://localhost:8080/projects/1/data`).
- Log in with the email and password printed by the script (`annotator@example.com`, password in `data/settings.json`), or *Sign up* to create an account.
- `data/`: `settings.json` (user, password, API token, ports), `label-studio/` (database), `label-studio.log`, `ml-backend/` (weights, datasets, training runs, trained models).
- Annotations: in the database, `data/label-studio/label_studio.sqlite3`. To get them: *Export* in the project (JSON, CSV, COCO, Pascal VOC, YOLO, with or without the images), or `--save-annotations` (one `<task id>.json` per task: the task and its annotations, with annotator, dates, time spent).
- Same command again: same project; the new images of the folder are imported.
- One mode per run: the backend refuses the projects of another mode (made by another run), with an error in its log; run the launcher again in their mode (`--mode`).
- Stop: Ctrl-C (both servers).

## Training

1. Open a task: the predictions of the model are shown.
2. Correct them, then *Submit*.
3. *Settings* > *Model* > `•••` > *Start Training*; or `--train-every N`: a training every N annotations.
4. Log of the backend, in the terminal: `Dataset: …`, the epochs, then `New model: …/data/ml-backend/models/project-1-<date>.pt`.
5. The next predictions come from the new model: open a task, or *Actions* > *Retrieve Predictions*.

- Validation: the tasks whose id is a multiple of 5, not used for training. Without such a task (or with only those), validation on the training images: the scores of the log are then optimistic.
- Images: only those of the folder (Local Files storage). Images added with *Import* are not supported: the backend cannot read them (an error in its log, no prediction, no training).
- Predictions: the labels of the project only; the other classes of the model are left out, as in the training. With `--labels`, new labels (e.g. `DropCapital`) are learnt at the first training, or predicted from their names with `--cold-start`.
- `--cold-start`: YOLOE (or CLIP) predicts until the first training; the training then retrains the model of the mode.
- `--mode semantic`: the pixels without brush are learnt as `background`; paint every class present in the image.
- `--mode pose`: a box per object (label `object` with `--labels`); a keypoint belongs to its box when Label Studio links them (`parentID`), otherwise to the smallest box that contains it; without a box, one object per image. Keypoints of Label Studio have no visibility flag: an annotated keypoint is visible (`v = 2`).
- `semantic`, `caption`: the predictions have no score (*Order by Prediction score* does not apply).
- `--mode caption`: BLIP proposes a caption; no training.
- To change the backend: edit `yolo_backend/model.py`, then run again. To get the current one of this repository: remove `yolo_backend/`.
