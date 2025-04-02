from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
from io import BytesIO
import os
import base64

from app.services.detection_service import YoloV8Detector
from app.services.embedding_service import EmbeddingModel
from app.services.search_service import ImageSearchEngine
from app.config import MODEL_YOLO_PATH, MODEL_EMBEDDING_PATH, INDEX_PATH, IMAGE_NAME_PATH, BASE_IMAGE_DIR

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI(title="Image Search API")

# Mount static & template
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load model & search engine
detector = YoloV8Detector(model_path=MODEL_YOLO_PATH)
encoder = EmbeddingModel(model_path=MODEL_EMBEDDING_PATH)
search_engine = ImageSearchEngine(
    index_path=INDEX_PATH,
    name_path=IMAGE_NAME_PATH,
    base_image_dir=BASE_IMAGE_DIR
)

def image_to_base64(image_path):
    if not os.path.exists(image_path):
        print(f"ERROR: Tệp ảnh không tồn tại tại {image_path}")
        return None  # Hoặc có thể trả về giá trị mặc định
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def search_image(request: Request, file: UploadFile = File(...)):
    print("📥 Đang nhận ảnh...")
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    print("✅ Ảnh đã load xong")

    print("🔍 Đang detect object...")
    bboxes = detector.detect_objects(image)
    if not bboxes:
        print("❌ Không phát hiện object nào")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "results": [],
            "message": "Không phát hiện món ăn nào"
        })

    x1, y1, x2, y2 = bboxes[0]
    cropped_img = image.crop((x1, y1, x2, y2))
    print(f"✅ Đã crop ảnh tại bbox ({x1}, {y1}, {x2}, {y2})")

    print("🧠 Đang encode vector...")
    vector = encoder.encode(cropped_img)

    print("📡 Đang tìm kiếm ảnh tương tự...")
    result_paths = search_engine.search(vector, top_k=5)

    result_images = []
    for path in result_paths:
        result_images.append({
            "path": path,
            "image_base64": image_to_base64(path)
        })

    print("✅ Hoàn tất tìm kiếm")
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": result_images,
        "message": "Thành công"
    })
