import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn model
MODEL_YOLO_PATH = os.path.join(BASE_DIR, "models", "yolov8_food_v2.pt")
MODEL_EMBEDDING_PATH = os.path.join(BASE_DIR, "models", "resnet50_embedding_v2.pth")

# Đường dẫn FAISS index và metadata
INDEX_PATH = os.path.join(BASE_DIR, "models", "index.faiss")
IMAGE_NAME_PATH = os.path.join(BASE_DIR, "models", "image_names.pkl")

# Thư mục chứa ảnh gốc đã crop
BASE_IMAGE_DIR = os.path.join(BASE_DIR, "data", "processed")

# File ánh xạ mã món sang tên món
CATEGORY_PATH = "D:/KLTN_Image-Seaching_Systeam/app/data/UECFOOD256/category.txt"

# Tham số tìm kiếm
FAISS_TOP_K = 20000            # số lượng ảnh cần tìm trong FAISS
TOP_DISH_LIMIT = 10             # số lượng món ăn cần hiển thị (top 5)
IMAGES_PER_DISH = 20           # số lượng ảnh mỗi món ăn
