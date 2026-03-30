# YOLOv8 Flask Deployment - Glass vs Pen Detector

This project turns a trained YOLOv8 object detection model into a small web app using Flask and Docker.
Users upload an image, the model runs inference, and the app returns an annotated image with bounding boxes plus a table of detections.

## Project structure

```text
.
├── app.py
├── Dockerfile
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── predictions/
└── models/
    └── best.pt   <- add your trained YOLOv8 weights here
```
git 

## Run locally without Docker

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Run locally with Docker

Build the image:

```bash
docker build -t yolov8-glass-pen-app .
```

Run the container:

```bash
docker run -p 5000:5000 \
  -e MODEL_PATH=models/best.pt \
  -e CONF_THRESHOLD=0.25 \
  -e IMG_SIZE=864 \
  yolov8-glass-pen-app
```

Then open:

```text
http://localhost:5000
```

## Suggested GitHub repo contents

Upload these files to GitHub:
- `app.py`
- `templates/index.html`
- `requirements.txt`
- `Dockerfile`
- `README.md`
- your trained `models/best.pt` if repository size allows it, otherwise attach it in your Google Drive submission or use Git LFS

## Suggested online deployment workflow

### Render
1. Push the project to GitHub.
2. Create a new Web Service.
3. Connect the GitHub repository.
4. Keep the `Dockerfile` in the repo root so the platform can build from it.
5. Set any environment variables you want, such as `MODEL_PATH`, `CONF_THRESHOLD`, and `IMG_SIZE`.
6. Deploy and test the public URL.

## Health check

The app includes a simple health endpoint:

```text
/health
```

It returns whether the server is alive and whether the model file exists.
