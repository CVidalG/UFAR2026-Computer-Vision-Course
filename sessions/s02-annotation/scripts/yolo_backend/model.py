"""Label Studio ML backend: YOLO (Ultralytics), retrained from Label Studio, in one mode of annotation.

MODE: detect (boxes), segment (polygons), semantic (masks), pose (keypoints), classify (choices),
caption (text; BLIP, predictions only).
predict(): the predictions of the model for each task (labels of the project only), in the Label Studio format.
fit(): "Start Training" (Settings > Model) retrains YOLO on the annotated tasks; its best model then predicts.
COLD_START=1: until the first training, the labels are predicted from their names (YOLOE; CLIP for classify).
"""
from typing import List, Dict, Optional
from urllib.parse import unquote
from uuid import uuid4
import json
import os
import re
import shutil
import threading
import time

from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk import LabelStudio
from label_studio_sdk.converter.brush import decode_rle, mask2rle
from ultralytics import YOLO
import cv2
import numpy as np
import yaml

MODE = os.environ.get("MODE", "detect")
# Generic model of each mode (YOLO_MODEL: any Ultralytics model of the same task, file or name)
MODELS = {"detect": "yolo26n.pt", "segment": "yolo26n-seg.pt", "semantic": "yolo26n-sem.pt", "pose": "yolo26n-pose.pt",
          "classify": "yolo26n-cls.pt", "caption": "Salesforce/blip-image-captioning-base"}
MODEL = os.environ.get("YOLO_MODEL") or MODELS[MODE]
# Before the first training: predictions from the names of the labels (detect, segment, semantic; classify: CLIP)
COLD_START = os.environ.get("COLD_START") == "1"
COLD_START_MODEL = "yoloe-26s-seg.pt"
# Parent folder of the images folder (same value as for Label Studio)
DOCUMENT_ROOT = os.environ.get("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", "/absolute/path/to/data")
# Folder of the backend: its cache (cache.db), the datasets, the training runs and the trained models
MODEL_DIR = os.path.abspath(os.environ.get("MODEL_DIR", "."))
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 10))
# One training at a time
TRAINING = threading.Lock()
# Models slow to load (YOLOE with the text of the labels, CLIP, BLIP): loaded once, not at each request
LOADED = {}
# Control tag of each mode in the labeling configuration (RectangleLabels alone: detect)
TAGS = {"pose": "KeyPointLabels", "segment": "PolygonLabels", "semantic": "BrushLabels", "classify": "Choices",
        "caption": "TextArea"}


def image_path(url):
    # Local files storage: /data/local-files/?d=<path relative to DOCUMENT_ROOT>
    # (images imported with Data Import, /data/upload/...: not found, use a Local Files storage)
    return os.path.join(DOCUMENT_ROOT, unquote(url.split("?d=", 1)[-1]))


def mode_of(label_interface):
    # Mode of a project, from the control tags of its labeling configuration
    tags = {control.tag for control in label_interface.controls}
    return next((mode for mode, tag in TAGS.items() if tag in tags), "detect")


def words(label):
    # Text of a label for a text-prompted model: DropCapital, drop_capital -> "drop capital"
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label).replace("_", " ").replace("-", " ").lower()


def loaded(key, load):
    if key not in LOADED:
        LOADED[key] = load()
    return LOADED[key]


def yoloe(labels):
    # YOLOE: boxes and masks of the classes given as text; region labels: the labels of the project
    model = YOLO(COLD_START_MODEL)
    model.set_classes([words(label) for label in labels])
    model.model.names = dict(enumerate(labels))
    return model


class ZeroShotClassifier:
    """CLIP: the probability of each label, from the similarity of the image with "a photo of <label>"."""

    def __init__(self, labels):
        import clip
        import torch
        self.model, self.preprocess = clip.load("ViT-B/32", device="cpu", download_root=os.path.join(MODEL_DIR, "clip"))
        self.names = dict(enumerate(labels))
        with torch.no_grad():
            text = self.model.encode_text(clip.tokenize([f"a photo of {words(label)}" for label in labels]))
        self.text = text / text.norm(dim=-1, keepdim=True)

    def predict(self, paths):
        from PIL import Image
        from ultralytics.engine.results import Results
        import torch
        results = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            with torch.no_grad():
                image = self.model.encode_image(self.preprocess(img).unsqueeze(0))
            image = image / image.norm(dim=-1, keepdim=True)
            probs = (100 * image @ self.text.T).softmax(dim=-1)[0]
            results.append(Results(np.asarray(img)[:, :, ::-1], path=path, names=self.names, probs=probs))
        return results


class Captioner:
    """BLIP (transformers): a caption of each image."""

    def __init__(self, name):
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.processor = BlipProcessor.from_pretrained(name)
        self.model = BlipForConditionalGeneration.from_pretrained(name)

    def predict(self, paths):
        from PIL import Image
        texts = []
        for path in paths:
            inputs = self.processor(Image.open(path).convert("RGB"), return_tensors="pt")
            output = self.model.generate(**inputs, max_new_tokens=40)
            texts.append(self.processor.decode(output[0], skip_special_tokens=True))
        return texts


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def __init__(self, project_id, **kwargs):
        super(NewModel, self).__init__(project_id=project_id, **kwargs)
        # One mode per run: a project of another mode (made by another run of the launcher) is refused
        if mode_of(self.label_interface) != MODE:
            raise ValueError(f"Project {project_id}: mode {mode_of(self.label_interface)}, not {MODE}; "
                             f"run the launcher with --mode {mode_of(self.label_interface)}")
        # Best model of the last training of the project; otherwise, with COLD_START, a model that predicts the
        # labels from their names; otherwise the generic model of the mode
        model_path = self.get("model_path")
        labels = self.label_interface.get_control("label").labels if MODE != "caption" else []
        if model_path:
            self.model = YOLO(model_path)
        elif MODE == "caption":
            model_path = MODEL
            self.model = loaded(model_path, lambda: Captioner(MODEL))
        elif COLD_START and MODE == "classify":
            model_path = "clip-ViT-B-32"
            self.model = loaded((model_path, *labels), lambda: ZeroShotClassifier(labels))
        elif COLD_START and MODE in ("detect", "segment", "semantic"):
            model_path = COLD_START_MODEL
            self.model = loaded((model_path, *labels), lambda: yoloe(labels))
        else:
            model_path = MODEL
            self.model = YOLO(model_path)
        self.set("model_version", os.path.basename(model_path))
        # Conversions of the mode: prediction -> regions of Label Studio, regions -> labels of the YOLO dataset
        self.to_regions = {"detect": self.boxes, "segment": self.polygons, "semantic": self.masks,
                           "pose": self.keypoints, "classify": self.choices, "caption": self.texts}[MODE]
        self.to_yolo = {"detect": self.yolo_boxes, "segment": self.yolo_polygons,
                        "pose": self.yolo_keypoints}.get(MODE)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """

        img_path_list = []
        for task in tasks:
            img_path = image_path(task['data']['image'])
            print(img_path)
            img_path_list.append(img_path)

        results = self.model.predict(img_path_list)

        predictions = []
        for result in results:
            # The regions of the mode, with a label of the project only (the model can know other classes)
            records = [r for r in self.to_regions(result) if self.known(r)]
            scores = [r["score"] for r in records if "score" in r]
            predictions.append(
                {
                    "result": records,
                    # Score of the task: mean confidence of its regions (0 without a region);
                    # Data Manager: "Order by Prediction score", lowest first = least certain first
                    "score": float(np.mean(scores)) if scores else 0.0,
                }
            )
        return predictions

    def known(self, record):
        # A predicted region whose label is a label of the project (caption: any text)
        if record["type"] == "textarea":
            return True
        return record["value"][record["type"]][0] in self.label_interface.get_control(record["from_name"]).labels

    # --- Predictions -> regions of Label Studio (one method per mode) ---

    def boxes(self, result):
        records = []
        if result.boxes:
            orig_w, orig_h = result.orig_shape[1], result.orig_shape[0]

            for box, class_id, score in zip(result.boxes.xyxy,
                                            result.boxes.cls,
                                            result.boxes.conf):

                x1, y1, x2, y2 = box.tolist()
                normalized_x1, normalized_y1 = (x1 / orig_w) * 100, (y1 / orig_h) * 100
                normalized_x2, normalized_y2 = (x2 / orig_w) * 100, (y2 / orig_h) * 100

                width = normalized_x2 - normalized_x1
                height = normalized_y2 - normalized_y1

                class_name = result.names[int(class_id.item())]

                records.append(
                    {
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "original_width": orig_w,
                        "original_height": orig_h,
                        "image_rotation": 0,
                        "value": {
                            "x": normalized_x1,
                            "y": normalized_y1,
                            "width": width,
                            "height": height,
                            "rotation": 0,
                            "rectanglelabels": [class_name]
                        },
                        "score": float(score.item()),
                    }
                )
        return records

    def polygons(self, result):
        # Instance masks as polygons: their points, in percent of the image
        records = []
        if result.masks:
            orig_h, orig_w = result.orig_shape
            for points, class_id, score in zip(result.masks.xyn, result.boxes.cls, result.boxes.conf):
                records.append({"from_name": "label", "to_name": "image", "type": "polygonlabels",
                                "original_width": orig_w, "original_height": orig_h, "image_rotation": 0,
                                "value": {"points": (points * 100).tolist(),
                                          "polygonlabels": [result.names[int(class_id.item())]]},
                                "score": float(score.item())})
        return records

    def masks(self, result):
        # One brush mask (RLE of Label Studio) per class of the class map of the image: map of the semantic model,
        # or (YOLOE) the polygons of the instances; 255: no class
        orig_h, orig_w = result.orig_shape
        if result.semantic_mask is not None:
            class_map = result.semantic_mask.data.cpu().numpy()
        else:
            class_map = np.full((orig_h, orig_w), 255, np.uint8)
            if result.masks:
                for points, class_id in zip(result.masks.xy, result.boxes.cls):
                    cv2.fillPoly(class_map, [points.astype(np.int32)], int(class_id.item()))
        records = []
        for class_id in np.unique(class_map):
            name = result.names.get(int(class_id))
            if name is None or name == "background":
                continue
            mask = (class_map == class_id).astype(np.uint8) * 255
            records.append({"from_name": "label", "to_name": "image", "type": "brushlabels",
                            "original_width": orig_w, "original_height": orig_h, "image_rotation": 0,
                            "value": {"format": "rle", "rle": mask2rle(mask), "brushlabels": [name]}})
        return records

    def keypoints(self, result):
        # Each object: its box, then its visible keypoints, attached to the box (parentID); keypoint k of the model:
        # label k of the KeyPointLabels; an object whose class is not a label of the project is left out, keypoints too
        records = []
        if result.boxes and result.keypoints is not None:
            orig_h, orig_w = result.orig_shape
            keypoint_names = self.label_interface.get_control("keypoint").labels
            for box, class_id, score, points, visibility in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf,
                                                                 result.keypoints.xy, result.keypoints.conf):
                if result.names[int(class_id.item())] not in self.label_interface.get_control("label").labels:
                    continue
                x1, y1, x2, y2 = box.tolist()
                box_id = uuid4().hex[:10]
                records.append({"id": box_id, "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                                "original_width": orig_w, "original_height": orig_h, "image_rotation": 0,
                                "value": {"x": x1 / orig_w * 100, "y": y1 / orig_h * 100,
                                          "width": (x2 - x1) / orig_w * 100, "height": (y2 - y1) / orig_h * 100,
                                          "rotation": 0, "rectanglelabels": [result.names[int(class_id.item())]]},
                                "score": float(score.item())})
                for name, (x, y), conf in zip(keypoint_names, points.tolist(), visibility.tolist()):
                    if conf < 0.5:
                        continue
                    records.append({"id": uuid4().hex[:10], "parentID": box_id, "from_name": "keypoint",
                                    "to_name": "image", "type": "keypointlabels",
                                    "original_width": orig_w, "original_height": orig_h, "image_rotation": 0,
                                    "value": {"x": x / orig_w * 100, "y": y / orig_h * 100, "width": 0.5,
                                              "keypointlabels": [name]},
                                    "score": conf})
        return records

    def choices(self, result):
        # The most probable class of the image
        return [{"from_name": "label", "to_name": "image", "type": "choices",
                 "value": {"choices": [result.names[result.probs.top1]]}, "score": float(result.probs.top1conf)}]

    def texts(self, text):
        return [{"from_name": "caption", "to_name": "image", "type": "textarea", "value": {"text": [text]}}]

    # --- Regions of an annotation -> lines of the YOLO label file of the image ---

    def yolo_boxes(self, regions, names):
        lines = []
        for region in regions:
            v = region["value"]
            # Boxes only, with a label of the project
            if region["type"] != "rectanglelabels" or v["rectanglelabels"][0] not in names:
                continue
            # Label Studio: top-left corner and size, in percent; YOLO: center and size, in [0, 1]
            x_center = (v["x"] + v["width"] / 2) / 100
            y_center = (v["y"] + v["height"] / 2) / 100
            lines.append(f"{names.index(v['rectanglelabels'][0])} {x_center} {y_center} "
                         f"{v['width'] / 100} {v['height'] / 100}")
        return lines

    def yolo_polygons(self, regions, names):
        # YOLO-seg: the class, then the points of the polygon, in [0, 1]
        return [f"{names.index(r['value']['polygonlabels'][0])} "
                + " ".join(f"{x / 100} {y / 100}" for x, y in r["value"]["points"])
                for r in regions if r["type"] == "polygonlabels" and r["value"]["polygonlabels"][0] in names]

    def yolo_keypoints(self, regions, names):
        # YOLO pose: for each object, its class and box (center, size), then each keypoint of the project, in order:
        # x y v (v = 2: annotated, 0: absent). A keypoint belongs to its box in Label Studio (parentID: predicted, or
        # grouped by hand), otherwise to the smallest box that contains it; without any box, one object, whose box is
        # the extent of the keypoints
        keypoint_names = self.label_interface.get_control("keypoint").labels
        boxes = [r for r in regions if r["type"] == "rectanglelabels" and r["value"]["rectanglelabels"][0] in names]
        points = [r for r in regions
                  if r["type"] == "keypointlabels" and r["value"]["keypointlabels"][0] in keypoint_names]
        if not boxes and points:
            xs, ys = [p["value"]["x"] for p in points], [p["value"]["y"] for p in points]
            boxes = [{"value": {"x": min(xs), "y": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys),
                                "rectanglelabels": [names[0]]}}]

        def inside(b, p):
            return b["x"] <= p["x"] <= b["x"] + b["width"] and b["y"] <= p["y"] <= b["y"] + b["height"]

        ids = [b.get("id") for b in boxes]
        owner = {}
        for i, p in enumerate(points):
            if p.get("parentID") and p["parentID"] in ids:
                owner[i] = ids.index(p["parentID"])
                continue
            containing = [j for j, b in enumerate(boxes) if inside(b["value"], p["value"])]
            if containing:
                owner[i] = min(containing, key=lambda j: boxes[j]["value"]["width"] * boxes[j]["value"]["height"])
        boxes = [b["value"] for b in boxes]
        lines = []
        for j, b in enumerate(boxes):
            own = {points[i]["value"]["keypointlabels"][0]: points[i]["value"] for i in owner if owner[i] == j}
            values = []
            for name in keypoint_names:
                p = own.get(name)
                values += [p["x"] / 100, p["y"] / 100, 2] if p else [0, 0, 0]
            lines.append(f"{names.index(b['rectanglelabels'][0])} {(b['x'] + b['width'] / 2) / 100} "
                         f"{(b['y'] + b['height'] / 2) / 100} {b['width'] / 100} {b['height'] / 100} "
                         + " ".join(str(v) for v in values))
        return lines

    def class_map(self, regions, names, img_path):
        # Semantic mask: PNG of class ids; pixels without brush: background (id len(names))
        height, width = cv2.imread(img_path).shape[:2]
        class_map = np.full((height, width), len(names), np.uint8)
        for r in regions:
            if r["type"] == "brushlabels" and r["value"]["brushlabels"][0] in names:
                rgba = decode_rle(r["value"]["rle"]).reshape(r["original_height"], r["original_width"], 4)
                class_map[rgba[:, :, 3] > 0] = names.index(r["value"]["brushlabels"][0])
        return class_map

    def fit(self, event, data, **kwargs):
        # Events: ANNOTATION_CREATED, ANNOTATION_UPDATED, ANNOTATION_DELETED, START_TRAINING ("Start Training")
        if event != "START_TRAINING":
            return
        if MODE == "caption":
            print("caption: predictions only, no training")
            return
        if not TRAINING.acquire(blocking=False):
            print("Training already running")
            return
        # In the background: Label Studio does not wait for the end of the training
        threading.Thread(target=self.train).start()
        return {"training": "started"}

    def train(self):
        try:
            # 1. The annotated tasks, exported from Label Studio (LABEL_STUDIO_URL, LABEL_STUDIO_API_KEY)
            ls = LabelStudio(base_url=os.environ["LABEL_STUDIO_URL"])
            tasks = json.loads(b"".join(ls.projects.exports.download_sync(id=int(self.project_id), export_type="JSON")))

            # 2. YOLO dataset, in train/ and val/: images/ and labels/ (one .txt per image: class x_center y_center
            # width height, in [0, 1]), masks/ (semantic), or one folder per class (classify)
            names = self.label_interface.get_control("label").labels
            dataset = os.path.join(MODEL_DIR, "datasets", f"project-{self.project_id}")
            shutil.rmtree(dataset, ignore_errors=True)
            annotated = []
            for task in tasks:
                annotations = [a for a in task["annotations"] if not a["was_cancelled"]]
                if annotations:
                    annotated.append((task, annotations[-1]["result"]))
            if not annotated:
                print("No annotated task: no training")
                return
            # Validation: the tasks whose id is a multiple of 5 (none, or all: validation on the training images)
            val_ids = {task["id"] for task, _ in annotated if task["id"] % 5 == 0}
            if len(val_ids) == len(annotated):
                val_ids = set()
            counts = {"train": 0, "val": 0}
            n_regions = 0
            for task, regions in annotated:
                split = "val" if task["id"] in val_ids else "train"
                img_path = image_path(task["data"]["image"])
                name = f"task-{task['id']}"
                if MODE == "classify":
                    choices = [r["value"]["choices"][0] for r in regions
                               if r["type"] == "choices" and r["value"]["choices"][0] in names]
                    if not choices:
                        continue
                    folder = os.path.join(dataset, split, choices[0])
                else:
                    folder = os.path.join(dataset, "images", split)
                os.makedirs(folder, exist_ok=True)
                shutil.copy(img_path, os.path.join(folder, name + os.path.splitext(img_path)[1]))
                counts[split] += 1
                if MODE == "semantic":
                    os.makedirs(os.path.join(dataset, "masks", split), exist_ok=True)
                    cv2.imwrite(os.path.join(dataset, "masks", split, name + ".png"),
                                self.class_map(regions, names, img_path))
                    n_regions += sum(r["type"] == "brushlabels" for r in regions)
                elif MODE != "classify":
                    lines = self.to_yolo(regions, names)
                    os.makedirs(os.path.join(dataset, "labels", split), exist_ok=True)
                    with open(os.path.join(dataset, "labels", split, name + ".txt"), "w") as f:
                        f.write("".join(line + "\n" for line in lines))
                    n_regions += len(lines)
                else:
                    n_regions += 1
            if MODE == "classify":
                # One folder per label, in each split (validation on the training images: val/ = train/)
                for label in names:
                    os.makedirs(os.path.join(dataset, "train", label), exist_ok=True)
                if not val_ids:
                    shutil.copytree(os.path.join(dataset, "train"), os.path.join(dataset, "val"))
                for label in names:
                    os.makedirs(os.path.join(dataset, "val", label), exist_ok=True)
                data = dataset
            else:
                config = {"path": dataset, "train": "images/train", "val": "images/val" if val_ids else "images/train",
                          "names": names}
                if MODE == "pose":
                    keypoint_names = self.label_interface.get_control("keypoint").labels
                    config["kpt_shape"] = [len(keypoint_names), 3]
                    config["kpt_names"] = {i: keypoint_names for i in range(len(names))}
                    # Horizontal flip (augmentation): left_x and right_x swapped, the other keypoints kept
                    mirror = [n.replace("left", "\0").replace("right", "left").replace("\0", "right")
                              for n in keypoint_names]
                    config["flip_idx"] = [keypoint_names.index(m) if m in keypoint_names else i
                                          for i, m in enumerate(mirror)]
                if MODE == "semantic":
                    config["masks_dir"] = "masks"
                    config["names"] = names + ["background"]
                data = os.path.join(dataset, "data.yaml")
                with open(data, "w") as f:
                    yaml.safe_dump(config, f)
            print("Dataset:", counts["train"], "training images,", counts["val"] or "the same", "for validation,",
                  n_regions, "regions,", len(names), "labels")

            # 3. Training, from the generic model of the mode
            version = f"project-{self.project_id}-{time.strftime('%Y%m%d-%H%M%S')}"
            model = YOLO(MODEL)
            model.train(data=data, epochs=EPOCHS, workers=0,
                        project=os.path.join(MODEL_DIR, "runs"), name=version)

            # 4. Its best model (best.pt) becomes the model of the project: the next predictions come from it
            model_path = os.path.join(MODEL_DIR, "models", version + ".pt")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            shutil.copy(model.trainer.best, model_path)
            self.set("model_path", model_path)
            print("New model:", model_path)
        finally:
            TRAINING.release()
