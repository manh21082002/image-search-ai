# 🚀 Image Search AI - Hệ thống Tìm kiếm Hình ảnh bằng Trí tuệ Nhân tạo

## 📝 Giới thiệu
**Image Search AI** là một hệ thống tìm kiếm hình ảnh sử dụng Machine Learning và Deep Learning như **YOLO, R-CNN, FAISS** để tìm kiếm ảnh tương tự từ cơ sở dữ liệu.

🔹 **Chức năng chính**:
- 📷 **Nhận diện vật thể** trong ảnh
- 🚀 **Mã hóa ảnh thành vector đặc trưng** để tìm kiếm nhanh
- 🔍 **Tìm kiếm kết hợp trên database nội bộ và API Google/Bing**

---

## 📂 Cấu trúc thư mục
```plaintext
image-search-system/
│── app/
│   │── models/               # Chứa mô hình AI (YOLO, R-CNN, FAISS)
│   │── services/             # Xử lý ảnh, mã hóa vector, tìm kiếm
│   │── routes/               # API endpoints
│   │── utils/                # Hàm hỗ trợ
│   │── database.py           # Quản lý cơ sở dữ liệu ảnh
│   │── config.py             # Cấu hình hệ thống
│── data/
│   │── raw/                  # Dữ liệu gốc (ảnh)
│   │── processed/            # Dữ liệu sau khi tiền xử lý
│── notebooks/                # Notebook kiểm thử mô hình
│── scripts/                  # Script train model
│── static/                   # Frontend, giao diện web
│── logs/                     # Nhật ký hoạt động
│── requirements.txt          # Danh sách thư viện
│── Dockerfile                # Cấu hình Docker
│── README.md                 # Hướng dẫn sử dụng
│── main.py                   # Khởi động FastAPI

### 🟦 **1. Cài đặt môi trường**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn main:app --reload
