# app.py
# FastAPI backend for AI-Powered Currency Detection (GBP+PKR)
# Uses YOLOv8 (Ultralytics) for detection and a classifier (MobileNetV3/EfficientNet) for denomination.
# Returns predictions and an annotated image; frontend speaks results using Web Speech API.

import os
import io
import json
import base64
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from torchvision import transforms
import timm
import logging

# -----------------------------
# CONFIG
# -----------------------------
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolo_gbp_best.pt")
CLS_WEIGHTS  = os.getenv("CLS_WEIGHTS",  "mobilenetv3_small_best.pth")  # or effnet_b0_best.pth
CLS_LABELS   = os.getenv("CLS_LABELS",   "cls_labels.json")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 224     # classifier input
CONF_THR     = float(os.getenv("CONF_THR", "0.35"))
IOU_THR      = float(os.getenv("IOU_THR", "0.50"))

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# HELPERS
# -----------------------------
def load_labels(labels_path: str) -> List[str]:
    with open(labels_path, "r") as f:
        data = json.load(f)
    # supports {"classes": [...]} or {0:"...",1:"..."}
    if isinstance(data, dict) and "classes" in data:
        return data["classes"]
    if isinstance(data, dict):
        return [v for k, v in sorted(data.items(), key=lambda kv: int(kv[0]))]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported labels format")

def build_classifier(ckpt_path: str, num_classes: int, forced_arch: Optional[str] = None):
    # infer arch from filename (fallback to param)
    arch = forced_arch
    name = Path(ckpt_path).name.lower()
    if arch is None:
        if "mobilenet" in name:
            arch = "mobilenetv3_small_100"
        else:
            arch = "efficientnet_b0"
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt.get("classes", None), arch

def to_base64_jpg(img_bgr: np.ndarray, quality: int = 90) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, enc = cv2.imencode(".jpg", img_bgr, encode_param)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(enc.tobytes()).decode("utf-8")

def pretty_label(raw: str) -> str:
    s = raw.replace("-", "_").replace(" ", "_").lower()
    if "gbp" in s or "pound" in s:
        digits = "".join([c for c in s if c.isdigit()])
        return f"GBP £{digits}" if digits else "GBP note"
    if "pkr" in s or "rs" in s or "rupee" in s:
        digits = "".join([c for c in s if c.isdigit()])
        return f"PKR Rs {digits}" if digits else "PKR note"
    return raw

# -----------------------------
# LOAD MODELS
# -----------------------------
if not Path(YOLO_WEIGHTS).exists():
    raise FileNotFoundError(f"YOLO weights not found at {YOLO_WEIGHTS}")
yolo = YOLO(YOLO_WEIGHTS)

cls_labels = load_labels(CLS_LABELS) if Path(CLS_LABELS).exists() else None
cls_model = None
cls_tf = None
cls_arch = None

if Path(CLS_WEIGHTS).exists():
    cls_model, saved_classes, cls_arch = build_classifier(CLS_WEIGHTS, num_classes=len(cls_labels))
    if saved_classes is not None and cls_labels is None:
        cls_labels = saved_classes
    mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
    cls_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

# -----------------------------
# FASTAPI
# -----------------------------
app = FastAPI(title="Currency Assist (GBP+PKR)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class Box(BaseModel):
    x1: int; y1: int; x2: int; y2: int
    score: float
    label: str

class InferResponse(BaseModel):
    boxes: List[Box]
    speech_text: str
    annotated_image: Optional[str]  # base64 JPG

@app.get("/", response_class=HTMLResponse)
def home():
    return (
        "<h2>Currency Assist API</h2>"
        "<p>POST an image to <code>/infer</code> with form field <code>file</code>.</p>"
        "<p>Open <code>index.html</code> in your browser for the UI.</p>"
    )

def classify_crop(crop_bgr: np.ndarray) -> Tuple[str, float]:
    if cls_model is None or cls_tf is None or cls_labels is None:
        return "", 0.0
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tens = cls_tf(rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = cls_model(tens)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    idx = int(np.argmax(probs))
    return cls_labels[idx], float(probs[idx])

@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    try:
        logging.info(f"Received file: {file.filename}, Content-Type: {file.content_type}")
        bytes_data = await file.read()
        logging.info(f"File size: {len(bytes_data)} bytes")

        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logging.error("Failed to decode image. The file might be corrupted or in an unsupported format.")
            raise HTTPException(status_code=400, detail="Invalid image")

        H, W = img.shape[:2]
        results = yolo.predict(img, conf=CONF_THR, iou=IOU_THR, verbose=False)

        boxes_out: List[Box] = []
        annotated = img.copy()

        for r in results:
            if r.boxes is None: 
                continue
            for b in r.boxes:
                xyxy = b.xyxy.cpu().numpy().squeeze().astype(int).tolist()
                x1,y1,x2,y2 = xyxy
                score = float(b.conf.item())
                cls_id = int(b.cls.item())
                det_name = yolo.model.names.get(cls_id, f"class_{cls_id}")
                label_text = det_name
                # refine with classifier if available
                x1c, y1c, x2c, y2c = max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)
                crop = annotated[y1c:y2c, x1c:x2c] if (x2c>x1c and y2c>y1c) else None
                if crop is not None and crop.size > 0 and cls_model is not None:
                    cname, cprob = classify_crop(crop)
                    if cname:
                        label_text = cname
                        score = cprob

                # draw
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                tag = f"{pretty_label(label_text)} ({score*100:.1f}%)"
                cv2.rectangle(annotated, (x1, max(0,y1-24)), (x1+min(320, x2-x1), y1), (0,255,0), -1)
                cv2.putText(annotated, tag, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)

                boxes_out.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, score=score, label=pretty_label(label_text)))

        # Generate speech text
        unique_labels = []
        for b in boxes_out:
            if b.label not in unique_labels:
                unique_labels.append(b.label)
        speech_text = ", ".join(unique_labels) if unique_labels else "No banknote detected"

        annotated_b64 = to_base64_jpg(annotated)
        return JSONResponse({
            "boxes": [b.model_dump() for b in boxes_out],
            "speech_text": speech_text,
            "annotated_image": annotated_b64
        })

    except Exception as e:
        logging.error("Error during inference", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
