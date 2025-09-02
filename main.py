import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io, json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

# --- Load model + class map ---
MODEL_PATH = "model_best.pt"
CLASS_MAP_JSON = "class_to_idx.json"

with open(CLASS_MAP_JSON, "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()

# --- Albumentations transforms ---
IMG_SIZE = 256
val_tfms = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

app = FastAPI()

def preprocess(img: Image.Image):
    img = np.array(img.convert("RGB"))
    transformed = val_tfms(image=img)["image"]
    return transformed.unsqueeze(0)

@app.get("/")
def home():
    return {"status": "Plant Disease API running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))

    x = preprocess(img)
    with torch.no_grad():
        outputs = model(x)
        _, pred = torch.max(outputs, 1)
    label = idx_to_class[pred.item()]

    return {"prediction": label}
