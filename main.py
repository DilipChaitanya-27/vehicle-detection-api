from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import torch
import numpy as np
import tempfile
import os

app = FastAPI()

# Load YOLOv5 model (use yolov5s for small size and faster inference)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vehicle Detection API"}

@app.post("/detect-vehicles/")
async def detect_vehicles(file: UploadFile = File(...)):
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    total_vehicle_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            label = model.names[int(cls)]
            if label in vehicle_classes:
                total_vehicle_count += 1

    cap.release()
    os.remove(tmp_path)

    return JSONResponse(content={"vehicle_count": total_vehicle_count})

