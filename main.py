from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
from io import BytesIO
import os
import base64
import numpy as np
from collections import defaultdict

from app.services.detection_service import YoloV8Detector
from app.services.embedding_service import EmbeddingModel
from app.services.search_service import ImageSearchEngine
from app.config import (
    MODEL_YOLO_PATH,
    MODEL_EMBEDDING_PATH,
    INDEX_PATH,
    IMAGE_NAME_PATH,
    BASE_IMAGE_DIR,
    CATEGORY_PATH,
    FAISS_TOP_K,
    TOP_DISH_LIMIT,
    IMAGES_PER_DISH
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI(title="Image Search API")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load models
object_detector = YoloV8Detector(model_path=MODEL_YOLO_PATH)
embedding_model = EmbeddingModel(model_path=MODEL_EMBEDDING_PATH)
search_engine = ImageSearchEngine(
    index_path=INDEX_PATH,
    name_path=IMAGE_NAME_PATH,
    base_image_dir=BASE_IMAGE_DIR
)

def image_to_base64(image_path):
    if not os.path.exists(image_path):
        print(f"⚠️ File not found: {image_path}")
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def get_dish_code_from_filename(filename):
    return filename.split("_")[0]

def read_category_mapping():
    mapping = {}
    with open(CATEGORY_PATH, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = " ".join(parts[1:])
    return mapping

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def search_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    bboxes = object_detector.detect_objects(image)
    if not bboxes:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "results": [],
            "message": "Không phát hiện món ăn nào"
        })

    x1, y1, x2, y2 = bboxes[0]
    cropped_img = image.crop((x1, y1, x2, y2))
    query_vector = embedding_model.encode(cropped_img).reshape(1, -1).astype('float32')

    distances, indices = search_engine.index.search(query_vector, FAISS_TOP_K)
    distances = distances[0]
    indices = indices[0]
    image_paths = [os.path.join(search_engine.base_image_dir, search_engine.image_names[i]) for i in indices]
    dish_codes = [get_dish_code_from_filename(os.path.basename(p)) for p in image_paths]

    top_dish_codes = []
    for code in dish_codes[:TOP_DISH_LIMIT * 3]:
        if code not in top_dish_codes:
            top_dish_codes.append(code)
        if len(top_dish_codes) == TOP_DISH_LIMIT:
            break

    dish_to_images = defaultdict(list)
    for path, code, dist in zip(image_paths, dish_codes, distances):
        if code in top_dish_codes:
            dish_to_images[code].append((path, dist))

    category_map = read_category_mapping()
    grouped_results = []

    for code in top_dish_codes:
        dish_name = category_map.get(code, f"Món {code}")
        top_imgs = dish_to_images[code][:IMAGES_PER_DISH]
        group = {
            "dish_name": dish_name,
            "images": []
        }
        for path, _ in top_imgs:
            img_base64 = image_to_base64(path)
            if img_base64:
                group["images"].append({"image_base64": img_base64})
        grouped_results.append(group)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": grouped_results,
        "message": "Kết quả từ FAISS"
    })
