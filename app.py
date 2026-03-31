import os
import uuid
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, render_template, request, url_for
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PREDICTION_DIR = BASE_DIR / "static" / "predictions"
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "best.pt"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "864"))
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

UPLOAD_DIR.mkdir(exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

_model = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{MODEL_PATH}'. Add trained YOLO weights there "
                f"or set the MODEL_PATH environment variable."
            )
        _model = YOLO(str(MODEL_PATH))
    return _model


def extract_detections(results) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    boxes = results[0].boxes
    names = results[0].names

    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "label": names.get(class_id, str(class_id)),
                "confidence": round(confidence, 4),
                "bbox": [x1, y1, x2, y2],
            }
        )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_image = None
    detections: List[Dict[str, Any]] = []
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please choose an image first."
        elif not allowed_file(file.filename):
            error = "Unsupported file type. Please upload png, jpg, jpeg, or webp."
        else:
            file_ext = file.filename.rsplit(".", 1)[1].lower()
            unique_name = f"{uuid.uuid4().hex}.{file_ext}"
            input_path = UPLOAD_DIR / unique_name
            file.save(input_path)

            try:
                Image.open(input_path).verify()
                model = get_model()
                results = model.predict(
                    source=str(input_path),
                    conf=CONF_THRESHOLD,
                    imgsz=IMG_SIZE,
                    save=False,
                    verbose=False,
                )

                detections = extract_detections(results)
                plotted = results[0].plot()

                output_name = f"pred_{uuid.uuid4().hex}.jpg"
                output_path = PREDICTION_DIR / output_name
                Image.fromarray(plotted[..., ::-1]).save(output_path, format="JPEG", quality=95)
                prediction_image = url_for("static", filename=f"predictions/{output_name}")
            except Exception as exc:
                error = str(exc)
            finally:
                if input_path.exists():
                    input_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        prediction_image=prediction_image,
        detections=detections,
        error=error,
        model_path=str(MODEL_PATH),
        conf_threshold=CONF_THRESHOLD,
        img_size=IMG_SIZE,
    )


@app.route("/health")
def health() -> tuple[dict, int]:
    model_ready = MODEL_PATH.exists()
    return {
        "status": "ok",
        "model_ready": model_ready,
        "model_path": str(MODEL_PATH),
    }, 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
