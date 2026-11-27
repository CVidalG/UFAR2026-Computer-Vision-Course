"""Label Studio ML backend: object detection with YOLO (Ultralytics), retrained from Label Studio.

predict(): the boxes of the model for each task, in the Label Studio format.
fit(): "Start Training" (Settings > Model) retrains YOLO on the annotated tasks; its best model then predicts.
"""
from typing import List, Dict, Optional
from urllib.parse import unquote
import json
import os
import shutil
import threading
import time

from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk import LabelStudio
from ultralytics import YOLO
import numpy as np
import yaml

# Generic model: any Ultralytics detection model (file or name)
MODEL = os.environ.get("YOLO_MODEL", "yolo26n.pt")
# Parent folder of the images folder (same value as for Label Studio)
DOCUMENT_ROOT = os.environ.get("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", "/absolute/path/to/data")
# Folder of the backend: its cache (cache.db), the datasets, the training runs and the trained models
MODEL_DIR = os.path.abspath(os.environ.get("MODEL_DIR", "."))
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 10))
# One training at a time
TRAINING = threading.Lock()


def image_path(url):
    # Local files storage: /data/local-files/?d=<path relative to DOCUMENT_ROOT>
    # (images imported with Data Import, /data/upload/...: not found, use a Local Files storage)
    return os.path.join(DOCUMENT_ROOT, unquote(url.split("?d=", 1)[-1]))


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def __init__(self, project_id, **kwargs):
        super(NewModel, self).__init__(project_id=project_id, **kwargs)
        # Best model of the last training of the project, otherwise the generic model
        model_path = self.get("model_path") or MODEL
        self.model = YOLO(model_path)
        self.set("model_version", os.path.basename(model_path))

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
            predictions.append(
                {
                    "result": records,
                    # Score of the task: mean confidence of its boxes (0 without a box);
                    # Data Manager: "Order by Prediction score", lowest first = least certain first
                    "score": float(np.mean([r["score"] for r in records])) if records else 0.0,
                }
            )
        return predictions

    def fit(self, event, data, **kwargs):
        # Events: ANNOTATION_CREATED, ANNOTATION_UPDATED, ANNOTATION_DELETED, START_TRAINING ("Start Training")
        if event != "START_TRAINING":
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

            # 2. YOLO dataset: images/, labels/ (one .txt per image: class x_center y_center width height, in [0, 1])
            names = self.label_interface.get_control("label").labels
            dataset = os.path.join(MODEL_DIR, "datasets", f"project-{self.project_id}")
            shutil.rmtree(dataset, ignore_errors=True)
            os.makedirs(os.path.join(dataset, "images"))
            os.makedirs(os.path.join(dataset, "labels"))
            images, boxes = 0, 0
            for task in tasks:
                annotations = [a for a in task["annotations"] if not a["was_cancelled"]]
                if not annotations:
                    continue
                img_path = image_path(task["data"]["image"])
                name = f"task-{task['id']}"
                shutil.copy(img_path, os.path.join(dataset, "images", name + os.path.splitext(img_path)[1]))
                images += 1
                with open(os.path.join(dataset, "labels", name + ".txt"), "w") as f:
                    for region in annotations[-1]["result"]:
                        v = region["value"]
                        # Boxes only, with a label of the project
                        if region["type"] != "rectanglelabels" or v["rectanglelabels"][0] not in names:
                            continue
                        # Label Studio: top-left corner and size, in percent; YOLO: center and size, in [0, 1]
                        x_center = (v["x"] + v["width"] / 2) / 100
                        y_center = (v["y"] + v["height"] / 2) / 100
                        f.write(f"{names.index(v['rectanglelabels'][0])} {x_center} {y_center} "
                                f"{v['width'] / 100} {v['height'] / 100}\n")
                        boxes += 1
            # Validation on the same images: too few images to set some aside
            with open(os.path.join(dataset, "data.yaml"), "w") as f:
                yaml.safe_dump({"path": dataset, "train": "images", "val": "images", "names": names}, f)
            print("Dataset:", images, "images,", boxes, "boxes,", len(names), "labels")

            # 3. Training, from the generic model
            version = f"project-{self.project_id}-{time.strftime('%Y%m%d-%H%M%S')}"
            model = YOLO(MODEL)
            model.train(data=os.path.join(dataset, "data.yaml"), epochs=EPOCHS, workers=0,
                        project=os.path.join(MODEL_DIR, "runs"), name=version)

            # 4. Its best model (best.pt) becomes the model of the project: the next predictions come from it
            model_path = os.path.join(MODEL_DIR, "models", version + ".pt")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            shutil.copy(model.trainer.best, model_path)
            self.set("model_path", model_path)
            print("New model:", model_path)
        finally:
            TRAINING.release()
