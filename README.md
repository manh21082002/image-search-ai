# 🚀 Image Search AI - Hệ thống Tìm kiếm Hình ảnh bằng Trí tuệ Nhân tạo

## 📝 Giới thiệu
**Image Search AI** là một hệ thống tìm kiếm hình ảnh món ăn sử dụng Machine Learning và Deep Learning như **YOLO, R-CNN, FAISS** để tìm kiếm ảnh tương tự từ cơ sở dữ liệu.

🔹 **Chức năng chính**:
- 📷 **Nhận diện vật thể** trong ảnh
- 🚀 **Mã hóa ảnh thành vector đặc trưng** để tìm kiếm nhanh
- 🔍 **Tìm kiếm kết hợp trên database nội bộ và API Google/Bing**

---

## 📂 Cấu trúc thư mục
```plaintext
image-search-system/
├── main.py
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .gitattributes
│
├── app/
│   ├── config.py
│   ├── database.py
│   ├── models/
│   │   ├── feature_vectors.pkl
│   │   ├── features.npy
│   │   ├── image_names.pkl
│   │   ├── index.faiss
│   │   ├── resnet50_embedding.pth
│   │   ├── resnet50_embedding_v2.pth
│   │   ├── yolov8_food.pt
│   │
│   ├── routes/
│   │   └── search_route.py
│   │
│   ├── services/
│   │   ├── detection_service.py
│   │   ├── embedding_service.py
│   │   └── search_service.py
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   ├── static/
│       └── favicon.ico
│
├── notebooks/
│   ├── 01_collect_data.ipynb
│   ├── 02_training_yolo.ipynb
│   ├── 03_encoding-image-and-testing.ipynb
│   ├── 04_train_model_embedding.ipynb
│   └── 05_train_model_embedding-v2.ipynb
│
├── data/                # Ảnh gốc & crop
├── logs/                # Log hệ thống
├── scripts/             # Script huấn luyện / xử lý
├── venv/                # Virtual Environment (bỏ vào .gitignore)


### 🟦 **1. Cài đặt môi trường**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn main:app --reload
