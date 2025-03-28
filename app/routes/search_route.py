from fastapi import APIRouter, UploadFile, File
from PIL import Image
from io import BytesIO

from app.services.detection_service import YoloV8Detector
from app.services.embedding_service import EmbeddingModel
from app.services.search_service import ImageSearchEngine
from app.config import MODEL_YOLO_PATH, MODEL_EMBEDDING_PATH, INDEX_PATH, IMAGE_NAME_PATH, BASE_IMAGE_DIR

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64

router = APIRouter()

# Khởi tạo các thành phần
detector = YoloV8Detector(model_path=MODEL_YOLO_PATH)
encoder = EmbeddingModel(model_path=MODEL_EMBEDDING_PATH)
search_engine = ImageSearchEngine(
    index_path=INDEX_PATH,
    name_path=IMAGE_NAME_PATH,
    base_image_dir=BASE_IMAGE_DIR
)

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


@router.post("/search")
async def search_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    bboxes = detector.detect_objects(image)

    if not bboxes:
        return {"message": "Không phát hiện món ăn nào", "results": []}

    x1, y1, x2, y2 = bboxes[0]
    cropped_img = image.crop((x1, y1, x2, y2))

    vector = encoder.encode(cropped_img)
    result_paths = search_engine.search(vector, top_k=5)

    result_images = []
    for path in result_paths:
        result_images.append({
            "path": path,
            "image_base64": image_to_base64(path)
        })

    return {
        "message": "Thành công",
        "results": result_images
    }
