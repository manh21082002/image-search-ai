from fastapi import APIRouter, UploadFile, File
from PIL import Image
from io import BytesIO

from app.services.detection_service import YoloV8Detector
from app.services.embedding_service import EmbeddingModel
from app.services.search_service import ImageSearchEngine
from app.config import MODEL_YOLO_PATH, MODEL_EMBEDDING_PATH, INDEX_PATH, IMAGE_NAME_PATH, BASE_IMAGE_DIR

import os
import base64

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    print("Bắt đầu quá trình tìm kiếm ảnh...")
    
    # Đọc file ảnh từ input
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    print("Ảnh đã được đọc và chuyển đổi thành RGB")

    # Phát hiện các vật thể trong ảnh
    print("Bắt đầu phát hiện vật thể...")
    bboxes = detector.detect_objects(image)
    if not bboxes:
        print("Không phát hiện vật thể nào trong ảnh.")
        return {"message": "Không phát hiện món ăn nào", "results": []}

    print(f"Phát hiện {len(bboxes)} vật thể trong ảnh.")
    
    # Cắt ảnh theo bounding box của vật thể
    x1, y1, x2, y2 = bboxes[0]
    cropped_img = image.crop((x1, y1, x2, y2))
    print(f"Đã cắt ảnh thành công, tọa độ bounding box: ({x1}, {y1}, {x2}, {y2})")

    # Mã hóa ảnh đã cắt
    print("Bắt đầu mã hóa ảnh...")
    vector = encoder.encode(cropped_img)
    print(f"Ảnh đã được mã hóa thành công thành vector: {vector.shape}")

    # Tìm kiếm ảnh tương tự trong cơ sở dữ liệu
    print("Bắt đầu tìm kiếm ảnh trong cơ sở dữ liệu...")
    result_paths = search_engine.search(vector, top_k=5)
    print(f"Đã tìm được {len(result_paths)} ảnh tương tự")

    # Chuyển đổi ảnh thành base64 để gửi lại cho client
    result_images = []
    for path in result_paths:
        result_images.append({
            "path": path,
            "image_base64": image_to_base64(path)
        })
    print("Đã chuyển tất cả ảnh kết quả thành base64")

    return {
        "message": "Thành công",
        "results": result_images
    }
