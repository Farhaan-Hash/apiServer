import torch
import torch.nn as nn
import timm
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

app = FastAPI()

# ---- Load class map ----
with open("class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ---- Load model ----
device = "cpu"
model = timm.create_model("resnet18", pretrained=False, num_classes=len(class_to_idx))
checkpoint = torch.load("model_best.pt", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# ---- Transforms ----
IMG_SIZE = 256
val_tfms = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img = np.array(image)
    img = val_tfms(image=img)["image"].unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        predicted_class = idx_to_class[preds.item()]

    return JSONResponse({"prediction": predicted_class})
