import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn model
MODEL_YOLO_PATH = os.path.join(BASE_DIR, "models", "yolov8_food.pt")
MODEL_EMBEDDING_PATH = os.path.join(BASE_DIR, "models", "resnet50_embedding_v2.pth")

# Đường dẫn FAISS index và metadata
INDEX_PATH = os.path.join(BASE_DIR, "models", "index.faiss")
IMAGE_NAME_PATH = os.path.join(BASE_DIR, "models", "image_names.pkl")

# Thư mục chứa ảnh gốc đã crop
BASE_IMAGE_DIR = os.path.join(BASE_DIR, "data", "cropped_images")
