import io
import torch
import timm
import uvicorn
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "resnet18"     # 🔴 change to your trained model name
NUM_CLASSES = 5             # 🔴 change to number of classes in your dataset
IMG_SIZE = 256              # must match training
CHECKPOINT_PATH = "model_best.pt"
CLASS_MAP_PATH = "class_to_idx.json"

# -----------------------------
# Load class map
# -----------------------------
with open(CLASS_MAP_PATH, "r") as fh:
    class_to_idx = json.load(fh)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# -----------------------------
# Build Model
# -----------------------------
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# -----------------------------
# Albumentations transform
# -----------------------------
val_tfms = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Plant Disease Model API is running 🚀"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(image)

        # apply transforms
        transformed = val_tfms(image=image)
        tensor = transformed["image"].unsqueeze(0)  # add batch dim

        # prediction
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_class = idx_to_class[pred_idx]
            confidence = probs[pred_idx].item()

        return JSONResponse({
            "class": pred_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})

# -----------------------------
# Run locally (Render will use Start Command)
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
